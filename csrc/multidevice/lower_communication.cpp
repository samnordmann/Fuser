// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <ir/interface_nodes.h>
#include <multidevice/device_mesh.h>
#include <multidevice/lower_communication.h>
#include <multidevice/pipeline.h>

namespace nvfuser {

static inline bool isDeviceInvolved(
    DeviceIdxType device_index,
    DeviceIdxType root,
    DeviceMesh mesh) {
  return device_index == root || mesh.has(device_index);
}

static inline bool isDeviceInvolved(
    DeviceIdxType device_index,
    DeviceMesh sender_mesh,
    DeviceMesh receiver_mesh) {
  return sender_mesh.has(device_index) || receiver_mesh.has(device_index);
}

static inline at::Tensor createDummyTensor(at::Tensor reference) {
  return at::empty(reference.sizes(), reference.options());
}

// Utility function used for setting up a scatter or gather communication "comm".
// Since most  of the steps are somewhat similar/opposite in those cases,
// we gathered the two implementations into one function.
// The argument "is_scatter" allows to discriminate between scatter and gather
CommParams CreateParamsForGatherScatter(
    DeviceIdxType device_index,
    DeviceIdxType root,
    DeviceMesh mesh,     // is_scatter? receivers : senders
    at::Tensor root_buf, // is_scatter? input buf : output buf
    at::Tensor buf,      // is_scatter? output buf : input buf
    bool is_scatter) {

  CommParams params;
  params.root = root;
  params.team = mesh.vector();
  bool is_root_in_mesh = mesh.has(root);
  if (!is_root_in_mesh) {
     params.team.push_back(root);
  }

  if (mesh.has(device_index)) {
    auto sliced_buf = buf.index({0, "..."});
    ((is_scatter)? params.dst_bufs : params.src_bufs) = {sliced_buf};
  }

  if (device_index == root) {
    for (auto i : c10::irange(mesh.vector().size())) {
      ((is_scatter)? params.src_bufs : params.dst_bufs).push_back(root_buf.index({static_cast<int>(i), "..."}));
    }
    // The scatter/gather semantics imposes the root to be both
    // sender and receiver. If the root is not in the mesh, we thus
    // have to artificially make it send and receive a dummy buffer
    // Since it is an "inplace" operation, this should not cause any overhead
    if (!is_root_in_mesh) {
      at::Tensor dummy = createDummyTensor(root_buf.index({0, "..."}));
      params.src_bufs.push_back(dummy);
      params.dst_bufs.push_back(dummy);
    }
  }
  return params;
}

void lowerToScatter(
    DeviceIdxType device_index,
    DeviceMesh sender_mesh,
    DeviceMesh receiver_mesh,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    std::vector<std::shared_ptr<Communication>>& comms) {
  // we arbitrarily choose the first device of the sender mesh to be the root
  auto root = sender_mesh.vector().at(0);
  if (!isDeviceInvolved(device_index, root, receiver_mesh))
    return;
  auto params = CreateParamsForGatherScatter(
                  device_index,
                  root,
                  receiver_mesh,
                  input_tensor,
                  output_tensor,
                  true);
  comms.push_back(std::make_shared<Scatter>(params));
}

void lowerToGather(
    DeviceIdxType device_index,
    DeviceMesh sender_mesh,
    DeviceMesh receiver_mesh,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    std::vector<std::shared_ptr<Communication>>& comms) {
  // we create as many Gathers as there are devices in the receiver mesh
  for (auto root : receiver_mesh.vector()) {
    if (!isDeviceInvolved(device_index, root, sender_mesh))
      continue;
    auto params = CreateParamsForGatherScatter(
                    device_index,
                    root,
                    sender_mesh,
                    output_tensor,
                    input_tensor,
                    false);
    comms.push_back(std::make_shared<Gather>(params));
  }
}

void lowerToAllgather(
    DeviceIdxType device_index,
    DeviceMesh mesh,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    std::vector<std::shared_ptr<Communication>>& comms) {
  if (!mesh.has(device_index))
    return;

  CommParams params;
  params.team = mesh.vector();
  for (auto i : c10::irange(mesh.vector().size())) {
    params.dst_bufs.push_back(output_tensor.index({static_cast<int>(i), "..."}));
  }
  params.src_bufs = {input_tensor.index({0, "..."})};

  comms.push_back(std::make_shared<Allgather>(params));
}

void lowerToSingleBroadcast(
    DeviceIdxType device_index,
    DeviceIdxType root,
    DeviceMesh mesh, // receiver devices
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    std::vector<std::shared_ptr<Communication>>& comms) {
  if (!isDeviceInvolved(device_index, root, mesh))
    return;

  CommParams params;
  params.root = root;
  params.team = mesh.vector();
  if (!mesh.has(root)) {
    params.team.push_back(root);
  }

  if (device_index == root) {
    params.src_bufs = {input_tensor};
  } else {
    params.dst_bufs = {output_tensor};
  }

  comms.push_back(std::make_shared<Broadcast>(params));
}

// For now, we assume that this function is called only if
// the input and output have the same parallelization (given by
// the argument "is_parallelized"). Later we could support more general cases.
void lowerToBroadcast(
    DeviceIdxType device_index,
    DeviceMesh sender_mesh,
    DeviceMesh receiver_mesh,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    bool is_parallelized,
    std::vector<std::shared_ptr<Communication>>& comms) {
  if (is_parallelized) {
    // if the inputs and ouputs are parallelized,
    // we create as many Broadcast as that will be handled in parallel
    for (auto i : c10::irange(sender_mesh.vector().size())) {
      lowerToSingleBroadcast(
          device_index,
          sender_mesh.vector().at(i),
          DeviceMesh({receiver_mesh.vector().at(i)}),
          input_tensor.index({0, "..."}),
          output_tensor.index({0, "..."}),
          comms);
    }
  } else {
    // we arbitrarily choose the first device of the sender mesh to be the root
    lowerToSingleBroadcast(
        device_index,
        sender_mesh.vector().at(0),
        receiver_mesh,
        input_tensor,
        output_tensor,
        comms);
  }
}

/*
TODO:
*) Propose several lowering paths for each given communication
   and provide a logic to decide which path to take
*) Leverage replication in the source to create several communications handled in
parallel The idea would be to evenly split the destinations accross the sources
*) Leverage the topology to ensure that the senders and recerivers are close
*/
std::vector<std::shared_ptr<Communication>> lowerCommunication(
    DeviceIdxType device_index,
    PipelineCommunication* c,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  std::vector<std::shared_ptr<Communication>> comms;
  TensorView* input_tv =
      c->in()->as<PipelineVal>()->getOriginalVal()->as<TensorView>();
  TensorView* output_tv =
      c->out()->as<PipelineVal>()->getOriginalVal()->as<TensorView>();
  at::Tensor dummy;

  auto sender_mesh = c->in()->as<PipelineVal>()->getStage()->descriptor()->mesh;
  auto receiver_mesh =
      c->out()->as<PipelineVal>()->getStage()->descriptor()->mesh;

  // Stores whether the I/O has its first axis parallelized on Didx
  bool is_input_parallel_d = input_tv->isSharded();
  bool is_output_parallel_d = output_tv->isSharded();

  TORCH_INTERNAL_ASSERT(
      !is_input_parallel_d ||
          sender_mesh.vector().size() ==
              static_cast<size_t>(input_tensor.size(0)),
      "the size of the mesh",
      sender_mesh.vector().size(),
      " doesn't match the size of the tensor ",
      input_tensor.size(0));
  TORCH_INTERNAL_ASSERT(
      !is_output_parallel_d ||
          receiver_mesh.vector().size() ==
              static_cast<size_t>(output_tensor.size(0)),
      "the size of the mesh",
      receiver_mesh.vector().size(),
      " doesn't match the size of the tensor ",
      output_tensor.size(0));
  TORCH_INTERNAL_ASSERT(!sender_mesh.vector().empty(), "sender mesh is empty");
  TORCH_INTERNAL_ASSERT(
      !receiver_mesh.vector().empty(), "receiver mesh is empty");

  if (!isDeviceInvolved(device_index, sender_mesh, receiver_mesh))
    return {};

  if (!is_input_parallel_d && is_output_parallel_d) {
    lowerToScatter(
        device_index,
        sender_mesh,
        receiver_mesh,
        input_tensor,
        output_tensor,
        comms);
  } else if (is_input_parallel_d && !is_output_parallel_d) {
    if (receiver_mesh.vector() == sender_mesh.vector()) {
      lowerToAllgather(
          device_index, sender_mesh, input_tensor, output_tensor, comms);
    } else {
      lowerToGather(
          device_index,
          sender_mesh,
          receiver_mesh,
          input_tensor,
          output_tensor,
          comms);
    }
  } else {
    lowerToBroadcast(
        device_index,
        sender_mesh,
        receiver_mesh,
        input_tensor,
        output_tensor,
        is_input_parallel_d,
        comms);
  }
  return comms;
}

} // namespace nvfuser

#endif
