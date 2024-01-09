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
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <limits>

namespace nvfuser {

namespace {

inline bool isDeviceInvolved(
    DeviceIdxType my_device_index,
    DeviceIdxType root,
    const DeviceMesh& mesh) {
  return my_device_index == root || mesh.has(my_device_index);
}

inline bool isDeviceInvolved(
    DeviceIdxType my_device_index,
    const DeviceMesh& sender_mesh,
    const DeviceMesh& receiver_mesh) {
  return sender_mesh.has(my_device_index) || receiver_mesh.has(my_device_index);
}

// Adds one or zero Scatter communication to the vector 'comms'
void lowerToScatter(
    DeviceIdxType my_device_index,
    TensorView* input_tv,
    TensorView* output_tv,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    std::vector<std::shared_ptr<Communication>>& comms) {
  // we arbitrarily choose the first device of the sender mesh to be the root
  auto root = input_tv->getDeviceMesh().vector().at(0);
  if (!isDeviceInvolved(my_device_index, root, output_tv->getDeviceMesh())) {
    return;
  }
  comms.push_back(std::make_shared<Scatter>(input_tv, output_tv, input_tensor, output_tensor, 
      my_device_index, root));
}

/*
Adds zero or multiple Gather communications to the vector 'comms'

Note that since the root of a Gather collective is a destination, we possibly
need multiple Gather if the tensor is replicated in the receiver mesh.
*/
void lowerToGather(
    DeviceIdxType my_device_index,
    TensorView* input_tv,
    TensorView* output_tv,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    std::vector<std::shared_ptr<Communication>>& comms) {
  // we create as many 'Gathers' as there are devices in the receiver mesh
  for (auto root : output_tv->getDeviceMesh().vector()) {
    if (!isDeviceInvolved(my_device_index, root, input_tv->getDeviceMesh())) {
      continue;
    }
    comms.push_back(std::make_shared<Gather>(input_tv, output_tv, input_tensor, output_tensor, my_device_index, root));
  }
}

// Add one or zero Allgather communication to the vector 'comms'
void lowerToAllgather(
    DeviceIdxType my_device_index,
    TensorView* input_tv,
    TensorView* output_tv,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    std::vector<std::shared_ptr<Communication>>& comms) {
  if (!input_tv->getDeviceMesh().has(my_device_index)) {
    return;
  }
  comms.push_back(std::make_shared<Allgather>(input_tv, output_tv, input_tensor, output_tensor));
}

// Adds several Broadcast or Send/Recv communications to the vector 'comms'
// For now, we assume that this function is called only if
// the input and output have the same sharding. Later we could support more
// general cases.
void lowerToBroadcastOrP2P(
    DeviceIdxType my_device_index,
    TensorView* input_tv,
    TensorView* output_tv,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    bool is_sharded,
    std::vector<std::shared_ptr<Communication>>& comms) {
  auto sender_mesh = input_tv->getDeviceMesh();
  auto receiver_mesh = output_tv->getDeviceMesh();
  if (is_sharded) {
    // if the inputs and ouputs are parallelized,
    // we create as many Broadcast as that will be handled in parallel
    for (auto i : c10::irange(sender_mesh.vector().size())) {
      NVF_ERROR(
          sender_mesh.vector().size() == receiver_mesh.vector().size(),
          "the receiver and sender meshes have different sizes");
      auto root = sender_mesh.vector().at(i);
      auto receiver = receiver_mesh.vector().at(i);
      auto receiver_mesh_ = DeviceMesh({receiver});
      if (isDeviceInvolved(my_device_index, root, receiver_mesh_)) {
        comms.push_back(std::make_shared<SendRecv>(input_tv, output_tv, input_tensor, output_tensor, 
                        my_device_index, root, receiver));
      }
    }
  } else {
    // we arbitrarily choose the first device of the sender mesh to be the root
    auto root = sender_mesh.vector().at(0);
    if (isDeviceInvolved(my_device_index, root, receiver_mesh)) {
      std::shared_ptr<Communication> comm;
      if (receiver_mesh.vector().size() == 1) {
        comm = std::make_shared<SendRecv>(input_tv, output_tv, input_tensor, output_tensor, my_device_index, root, receiver_mesh.vector()[0]);
      } else {
        comm = std::make_shared<Broadcast>(input_tv, output_tv, input_tensor, output_tensor, my_device_index, root);
      }
      comms.push_back(comm);
    }
  }
}

void lowerToReduce(
    DeviceIdxType my_device_index,
    TensorView* input_tv,
    TensorView* output_tv,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    BinaryOpType op_type,
    std::vector<std::shared_ptr<Communication>>& comms) {
  // we create as many Reduces as there are devices in the receiver mesh
  for (auto root : output_tv->getDeviceMesh().vector()) {
    if (!isDeviceInvolved(my_device_index, root, input_tv->getDeviceMesh())) {
      continue;
    }
    comms.push_back(std::make_shared<Reduce>(input_tv, output_tv, input_tensor, output_tensor, op_type, my_device_index, root));
  }
}

void lowerToAllreduce(
    DeviceIdxType my_device_index,
    TensorView* input_tv,
    TensorView* output_tv,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    BinaryOpType op_type,
    std::vector<std::shared_ptr<Communication>>& comms) {
  if (!input_tv->getDeviceMesh().has(my_device_index)) {
    return;
  }
  comms.push_back(std::make_shared<Allreduce>(input_tv, output_tv, input_tensor, output_tensor, op_type));
}

void lowerToReduceScatter(
    DeviceIdxType my_device_index,
    TensorView* input_tv,
    TensorView* output_tv,
    at::Tensor input_tensor,
    at::Tensor output_tensor,
    BinaryOpType op_type,
    std::vector<std::shared_ptr<Communication>>& comms) {
  if (!output_tv->getDeviceMesh().has(my_device_index)) {
    return;
  }
  comms.push_back(std::make_shared<ReduceScatter>(input_tv, output_tv, input_tensor, output_tensor, op_type));
}

} // namespace

/*
TODO:
*) Propose several lowering paths for each given communication
   and provide a logic to decide which path to take
*) Leverage replication in the source to create several communications handled
   in parallel. The idea would be to evenly split the destinations accross the
   sources
*) Leverage the topology to ensure that the senders and recerivers are close
*/
std::vector<std::shared_ptr<Communication>> lowerCommunication(
    DeviceIdxType my_device_index,
    Expr* c,
    at::Tensor input_tensor,
    at::Tensor output_tensor) {
  std::vector<std::shared_ptr<Communication>> comms;
  NVF_ERROR(
      c->inputs().size() == 1 && c->inputs().at(0)->isA<TensorView>() &&
          c->outputs().size() == 1 && c->outputs().at(0)->isA<TensorView>(),
      "I/O must be TensorViews");
  TensorView* input_tv = c->inputs().at(0)->as<TensorView>();
  TensorView* output_tv = c->outputs().at(0)->as<TensorView>();
  at::Tensor dummy;

  const auto& sender_mesh = input_tv->getDeviceMesh();
  const auto& receiver_mesh = output_tv->getDeviceMesh();

  // Stores whether the I/O has its first axis parallelized on Didx
  const bool is_input_sharded =
      isSharded(input_tv) && sender_mesh.vector().size() > 1;
  const bool is_output_sharded =
      isSharded(output_tv) && receiver_mesh.vector().size() > 1;

  // int input_sharded_dim = dimWithParallelType(input_tv, ParallelType::DIDx);
  // int output_sharded_dim = dimWithParallelType(output_tv, ParallelType::DIDx);

  auto original_expr = output_tv->definition();
  NVF_ERROR(
      isLowerableToCommunication(original_expr),
      "Lowering expression ",
      original_expr->toString(),
      " to communication is not supported");
  bool is_reduction = original_expr->isA<ReductionOp>();

  // TODO: Check valid sharding.
  // NVF_ERROR(
  //     !is_input_sharded || !input_tensor.numel() ||
  //         static_cast<size_t>(input_tensor.size(0)) == 1,
  //     "Sharded dimension should have allocation size 1, but is ",
  //     input_tensor.size(0));
  // NVF_ERROR(
  //     !is_output_sharded || !output_tensor.numel() || is_reduction ||
  //         static_cast<size_t>(output_tensor.size(0)) == 1,
  //     "Sharded dimension should have allocation size 1, but is ",
  //     output_tensor.size(0));
  if (is_reduction) {
    BinaryOpType op_type =
        output_tv->definition()->as<ReductionOp>()->getReductionOpType();
    NVF_ERROR(
        is_input_sharded || sender_mesh.vector().size() == 1,
        "the comm input must be sharded in case of reduce.",
        "Insert a `set` before the reduction to reshard")
    if (is_output_sharded) {
      NVF_ERROR(
          receiver_mesh == sender_mesh,
          "ReduceScatter operation must have the same sender and receiver device mesh. "
          "Insert a Set operation before or after the reduction to reshard ot another device mesh");
      lowerToReduceScatter(
          my_device_index,
          input_tv,
          output_tv,
          input_tensor,
          output_tensor,
          op_type,
          comms);
    } else {
      if (receiver_mesh == sender_mesh) {
        lowerToAllreduce(
            my_device_index,
            input_tv,
            output_tv,
            input_tensor,
            output_tensor,
            op_type,
            comms);
      } else {
        lowerToReduce(
            my_device_index,
            input_tv,
            output_tv,
            input_tensor,
            output_tensor,
            op_type,
            comms);
      }
    }
  } else {
    if (!is_input_sharded && is_output_sharded) {
      lowerToScatter(
          my_device_index,
          input_tv,
          output_tv,
          input_tensor,
          output_tensor,
          comms);
    } else if (is_input_sharded && !is_output_sharded) {
      if (receiver_mesh == sender_mesh) {
        lowerToAllgather(
            my_device_index, input_tv, output_tv, input_tensor, output_tensor, comms);
      } else {
        lowerToGather(
            my_device_index,
            input_tv,
            output_tv,
            input_tensor,
            output_tensor,
            comms);
      }
    } else {
      lowerToBroadcastOrP2P(
          my_device_index,
          input_tv,
          output_tv,
          input_tensor,
          output_tensor,
          is_input_sharded,
          comms);
    }
  }
  return comms;
}

bool isLowerableToCommunication(Expr* expr) {
  if (expr->isA<ReductionOp>()) {
    auto out = expr->as<ReductionOp>()->out();
    NVF_ERROR(out->isA<TensorView>(), "output is not a TensorView");
    auto out_tv = out->as<TensorView>();
    NVF_ERROR(
        out_tv->domain()->nDims() ==
            TensorDomain::noReductions(out_tv->getMaybeRFactorDomain()).size() +
                1,
        "only reducing one-axis at a time is supported");
    return true;
  }
  return expr->isA<LoadStoreOp>() &&
      (expr->as<LoadStoreOp>()->opType() == LoadStoreOpType::Set);
}

} // namespace nvfuser

#endif
