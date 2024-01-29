// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#ifdef USE_C10D_NCCL
#include <torch/csrc/distributed/c10d/ProcessGroupNCCL.hpp>
#endif

#include <multidevice/communication.h>
#include <multidevice/utils.h>

namespace nvfuser {
namespace {

template <typename T>
inline T getInitialValue(BinaryOpType op) {
  switch (op) {
    case BinaryOpType::Add:
      return 0;
    case BinaryOpType::Mul:
      return 1;
    case BinaryOpType::Min:
      return std::numeric_limits<T>::min();
    case BinaryOpType::Max:
    case BinaryOpType::BitwiseAnd:
      return std::numeric_limits<T>::max();
    case BinaryOpType::BitwiseOr:
    case BinaryOpType::BitwiseXor:
      return 0;
    default:
      NVF_ERROR(false, "invalid binary op type");
      return 0;
  }
}

// TODO: handle `c10d::RedOpType::reduceOp::AVG` and
// `c10d::RedOpType::reduceOp::PREMUL_SUM`
inline c10d::ReduceOp::RedOpType getC10dReduceOpType(BinaryOpType op) {
  switch (op) {
    case BinaryOpType::Add:
      return c10d::ReduceOp::RedOpType::SUM;
    case BinaryOpType::Mul:
      return c10d::ReduceOp::RedOpType::PRODUCT;
    case BinaryOpType::Min:
      return c10d::ReduceOp::RedOpType::MIN;
    case BinaryOpType::Max:
      return c10d::ReduceOp::RedOpType::MAX;
    case BinaryOpType::BitwiseAnd:
      return c10d::ReduceOp::RedOpType::BAND;
    case BinaryOpType::BitwiseOr:
      return c10d::ReduceOp::RedOpType::BOR;
    case BinaryOpType::BitwiseXor:
      return c10d::ReduceOp::RedOpType::BXOR;
    default:
      NVF_ERROR(false, "unsupported reduction operation");
      return c10d::ReduceOp::RedOpType::UNUSED;
  }
}

inline void assertBufferCount(
    const std::vector<at::Tensor>& bufs,
    size_t count) {
  NVF_ERROR(
      bufs.size() == count,
      "there must be ",
      count,
      " buffer(s), but ",
      bufs.size(),
      " were given");
}

inline void assertBuffersHaveSameSize(
    const std::vector<at::Tensor>& bufs1,
    const std::vector<at::Tensor>& bufs2) {
  if (bufs1.empty() && bufs2.empty()) {
    return;
  }
  auto sizes = (bufs1.empty() ? bufs2 : bufs1).at(0).sizes();
  for (auto& bufs : {bufs1, bufs2}) {
    for (auto& buf : bufs) {
      NVF_ERROR(buf.sizes() == sizes, "all buffers must have the same size");
    }
  }
}

inline void post_common(Communication& self, Communicator& comm) {
  NVF_ERROR(
      std::find(
          self.params().team.begin(),
          self.params().team.end(),
          comm.deviceId()) != self.params().team.end(),
      "current device index ",
      comm.deviceId(),
      " must be present in the communication's team");
}

inline void doLocalCopy(const at::Tensor& dst, const at::Tensor& src) {
  dst.copy_(src, /* non-blocking */ true);
}

// Creates a dummy tensor for scatter/gather communications,
inline at::Tensor createDummyTensor(at::Tensor reference) {
  return at::empty_like(reference, reference.options());
}

inline at::Tensor createDummyTensor(
    at::Tensor reference,
    BinaryOpType op_type) {
  return createDummyTensor(reference).fill_(getInitialValue<float>(op_type));
}

// Returns the permutation order for a tensor with device
// dimensions pushed to the front.
std::vector<int64_t> permuteOrder(TensorView* tv) {
  std::vector<int64_t> permute_order;
  auto ids = TensorDomain::noReductions(tv->getMaybeRFactorDomain());
  int sharded_axis = dimWithParallelType(tv, ParallelType::DIDx);
  if (sharded_axis != -1) {
    permute_order.push_back(sharded_axis);
  }
  for (size_t i = 0; i < ids.size(); i++) {
    if (ids[i]->getParallelType() != ParallelType::DIDx) {
      permute_order.push_back(int64_t(i));
    }
  }
  return permute_order;
}

// Returns permutation order to undo permuteOrder.
std::vector<int64_t> unpermuteOrder(std::vector<int64_t>& permute_order) {
  std::vector<int64_t> unpremute_order(permute_order.size());
  for (size_t i = 0; i < permute_order.size(); i++) {
    unpremute_order[permute_order[i]] = i;
  }
  return unpremute_order;
}
} // namespace

Communication::Communication(std::string name, bool has_root) :
  collective_type_(std::move(name)),
  has_root_(has_root) {}

Communication::Communication(CommParams params, std::string name, bool has_root)
    : params_(std::move(params)),
      collective_type_(std::move(name)),
      has_root_(has_root) {}

void Communication::validateParams() {
  assertBuffersHaveSameSize(params_.src_bufs, params_.dst_bufs);
  NVF_ERROR(
      std::unique(params_.team.begin(), params_.team.end()) ==
          params_.team.end(),
      "the communication must not involve the same device more than once");
  NVF_ERROR(!params_.team.empty(), "the team size must be greater than 0");
  if (has_root_) {
    auto it = std::find(params_.team.begin(), params_.team.end(), params_.root);
    NVF_ERROR(
        it != params_.team.end(),
        "root (device ",
        params_.root,
        ") must be present in the communication's team");
    // pytorch's process group expects the root to be specified
    // as an integer between 0 and world_size-1. We choose it to be
    // the device's relative index within the team
    root_relative_index_ = std::distance(params_.team.begin(), it);
  }
}

std::string Communication::toString(int indent) const {
  std::stringstream ss;
  std::string ext_indent(" ", indent);
  std::string indent1 = ext_indent + "  ";
  std::string indent2 = ext_indent + "    ";

  ss << ext_indent << "Communication " << collective_type_ << ": {\n";

  if (has_root_) {
    ss << indent1 << "root: " << params_.root << ",\n";
  }
  ss << indent1 << "team: {";
  for (auto r : params_.team) {
    ss << r << ", ";
  }
  ss << indent1 << "}\n";
  ss << indent1 << "src_bufs: {";
  for (auto& t : params_.src_bufs) {
    ss << "\n" << t;
  }
  ss << "\n" << indent1 << "}\n";
  ss << ext_indent << "}";

  return ss.str();
}

at::Tensor Communication::allocateOutputTensor(TensorView* tv, at::Tensor& tensor) {
  auto original_shape = tensor.sizes();
  auto permute_order = permuteOrder(tv);
  unpermute_order_ = unpermuteOrder(permute_order);
  std::vector<int64_t> new_shape;
  for (auto i : permute_order) {
    new_shape.push_back(original_shape[i]);
  }
  contig_output_ = at::empty(new_shape, tensor.options());
  return contig_output_;
}

bool Communication::requiresRelayoutOutputTensor(TensorView* input_tv, TensorView* output_tv) {
  return !isContiguousShard(input_tv) && isContiguousShard(output_tv);
}

void Communication::relayoutOutputTensor() {
  // Permute the axis into the correct order and copy into the destination buffer.
  if (output_.numel() > 0) {
    output_.copy_(contig_output_.permute(unpermute_order_));
  }
}

Broadcast::Broadcast(CommParams params) : Communication(params, "broadcast") {}

Broadcast::Broadcast(TensorView* input_tv, TensorView* output_tv, at::Tensor input, at::Tensor output, 
            DeviceIdxType my_device_index, DeviceIdxType root) 
            : Communication("broadcast") {
  params_.root = root;
  auto mesh = output_tv->getDeviceMesh();
  params_.team = mesh.vector();
  if (!mesh.has(root)) {
    params_.team.push_back(root);
  }

  if (my_device_index == root) {
    params_.src_bufs = {input};
  }
  if (mesh.has(my_device_index)) {
    params_.dst_bufs = {output};
  }
  validateParams();
}

c10::intrusive_ptr<c10d::Work> Broadcast::post(
    Communicator& comm,
    std::optional<CommunicatorBackend> backend) {
  post_common(*this, comm);

  if (comm.deviceId() == params_.root) {
    assertBufferCount(params_.src_bufs, 1);
    if (params_.dst_bufs.size() == 1) {
      doLocalCopy(params_.dst_bufs.at(0), params_.src_bufs.at(0));
    } else {
      assertBufferCount(params_.dst_bufs, 0);
    }
  } else {
    assertBufferCount(params_.src_bufs, 0);
    assertBufferCount(params_.dst_bufs, 1);
  }

  if (params_.team.size() == 1) {
    return nullptr;
  }

  return comm.getBackendForTeam(params_.team, backend)
      ->broadcast(
          comm.deviceId() == params_.root ? params_.src_bufs : params_.dst_bufs,
          {.rootRank = root_relative_index_});
}

Gather::Gather(CommParams params) : Communication(params, "gather") {
  validateParams();
}

Gather::Gather(TensorView* input_tv, TensorView* output_tv, at::Tensor input, at::Tensor output, 
                 DeviceIdxType my_device_index, DeviceIdxType root) :
    Communication("gather") {
  params_.root = root;
  DeviceMesh mesh = input_tv->getDeviceMesh();
  params_.team = mesh.vector();
  bool is_root_in_mesh = mesh.has(root);
  if (!is_root_in_mesh) {
    params_.team.push_back(root);
  }

  if (my_device_index == root && requiresRelayoutOutputTensor(input_tv, output_tv)) {
    output_ = output;
    output = allocateOutputTensor(input_tv, output);
  }

  if (requiresRelayoutOutputTensor(input_tv, output_tv) && mesh.has(my_device_index)) {
    // Permute the device axis to the front.
    // Input tensors are not copied since the device axis are size 1.
    params_.src_bufs = {input.permute(permuteOrder(input_tv))};
  } else if (mesh.has(my_device_index)) {
    params_.src_bufs = {input};
  }

  if (my_device_index == root) {
    for (auto i : c10::irange(mesh.vector().size())) {
      std::vector<at::indexing::TensorIndex> indices(output.dim(), at::indexing::Slice());
      // Pushed the sharded dimension forward.
      indices[0] = at::indexing::Slice(i, i+1);
      params_.dst_bufs.push_back(output.index(indices));
    }
    // The gather semantics imposes the root has to be in receiver mesh
    // If the root is not in the mesh, we thus
    // have to artificially make it send and receive a dummy buffer
    // Since it is an "inplace" operation, this should not cause any overhead
    if (!is_root_in_mesh) {
      at::Tensor dummy = createDummyTensor(params_.dst_bufs[0]);
      params_.src_bufs.push_back(dummy);
      params_.dst_bufs.push_back(dummy);
    }
  }
  validateParams();
}

void Gather::validateParams() {
  Communication::validateParams();
  assertBufferCount(params_.src_bufs, 1);
  NVF_ERROR(params_.team.size() > 1, "the team size must be greater than 1");
}

c10::intrusive_ptr<c10d::Work> Gather::post(
    Communicator& comm,
    std::optional<CommunicatorBackend> backend) {
  post_common(*this, comm);
  // This is used to change the representation of the buffers to match c10d
  // ProcessGroup API
  std::vector<std::vector<at::Tensor>> buf_list;
  if (comm.deviceId() == params_.root) {
    assertBufferCount(params_.dst_bufs, params_.team.size());
    buf_list = {std::move(params_.dst_bufs)};
  } else {
    assertBufferCount(params_.dst_bufs, 0);
  }
  auto work =
      comm.getBackendForTeam(params_.team, backend)
          ->gather(
              buf_list, params_.src_bufs, {.rootRank = root_relative_index_});
  if (comm.deviceId() == params_.root) {
    params_.dst_bufs = std::move(buf_list.back());
  }
  return work;
}

Allgather::Allgather(CommParams params)
    : Communication(params, "allgather", false) {
  validateParams();
}

Allgather::Allgather(TensorView* input_tv, TensorView* output_tv, 
                     at::Tensor input, at::Tensor output) :
    Communication("allgather", false) {
  auto mesh = input_tv->getDeviceMesh();
  params_.team = mesh.vector();

  if (requiresRelayoutOutputTensor(input_tv, output_tv)) {
    output_ = output;
    output = allocateOutputTensor(input_tv, output);
  }

  for (auto i : c10::irange(mesh.vector().size())) {
    std::vector<at::indexing::TensorIndex> indices(output.dim(), at::indexing::Slice());
    indices[0] = at::indexing::Slice(i, i+1);
    params_.dst_bufs.push_back(output.index(indices));
  }
  // If output axes were permuted, view the input in the same shape.
  // Only device dimensions are moved so allocation is unaffected.
  params_.src_bufs = {input.view(params_.dst_bufs[0].sizes())};
  validateParams();
}

void Allgather::validateParams() {
  Communication::validateParams();
  assertBufferCount(params_.src_bufs, 1);
  assertBufferCount(params_.dst_bufs, params_.team.size());
  NVF_ERROR(params_.team.size() > 1, "the team size must be greater than 1");
}

c10::intrusive_ptr<c10d::Work> Allgather::post(
    Communicator& comm,
    std::optional<CommunicatorBackend> backend) {
  post_common(*this, comm);
  // This is used to change the representation of the buffers to match c10d
  // ProcessGroup API
  std::vector<std::vector<at::Tensor>> buf_list;
  buf_list = {std::move(params_.dst_bufs)};
  auto work = comm.getBackendForTeam(params_.team, backend)
                  ->allgather(buf_list, params_.src_bufs, {});
  params_.dst_bufs = std::move(buf_list.back());
  return work;
}

Scatter::Scatter(CommParams params) : Communication(params, "scatter") {
  validateParams();
}

Scatter::Scatter(TensorView* input_tv, TensorView* output_tv, at::Tensor input, at::Tensor output, 
                 DeviceIdxType my_device_index, DeviceIdxType root) : Communication("scatter") {
  params_.root = root;
  DeviceMesh mesh = output_tv->getDeviceMesh();
  params_.team = mesh.vector();
  bool is_root_in_mesh = mesh.has(root);
  if (!is_root_in_mesh) {
    params_.team.push_back(root);
  }

  if (mesh.has(my_device_index)) {
    params_.dst_bufs = {output};
  }

  int sharded_dim = dimWithParallelType(output_tv, ParallelType::DIDx);
  if (my_device_index == root) {
    for (auto i : c10::irange(mesh.vector().size())) {
      std::vector<at::indexing::TensorIndex> indices(input.dim(), at::indexing::Slice());
      indices[sharded_dim] = at::indexing::Slice(i, i+1);
      auto x = input.index(indices).contiguous();
      params_.src_bufs.push_back(x);
    }
    // The scatter semantics imposes the root to be both
    // sender and receiver. If the root is not in the mesh, we thus
    // have to artificially make it send and receive a dummy buffer
    // Since it is an "inplace" operation, this should not cause any overhead
    if (!is_root_in_mesh) {
      at::Tensor dummy = createDummyTensor(params_.src_bufs[0]);
      params_.src_bufs.push_back(dummy);
      params_.dst_bufs.push_back(dummy);
    }
  }
  validateParams();
}

void Scatter::validateParams() {
  Communication::validateParams();
  assertBufferCount(params_.dst_bufs, 1);
  NVF_ERROR(params_.team.size() > 1, "the team size must be greater than 1");
}

c10::intrusive_ptr<c10d::Work> Scatter::post(
    Communicator& comm,
    std::optional<CommunicatorBackend> backend) {
  post_common(*this, comm);
  // This is used to change the representation of the buffers to match c10d
  // ProcessGroup API
  std::vector<std::vector<at::Tensor>> buf_list;
  if (comm.deviceId() == params_.root) {
    assertBufferCount(params_.src_bufs, params_.team.size());
    buf_list = {std::move(params_.src_bufs)};
  } else {
    assertBufferCount(params_.src_bufs, 0);
  }
  auto work =
      comm.getBackendForTeam(params_.team, backend)
          ->scatter(
              params_.dst_bufs, buf_list, {.rootRank = root_relative_index_});
  if (comm.deviceId() == params_.root) {
    params_.src_bufs = std::move(buf_list.back());
  }
  return work;
}

Reduce::Reduce(CommParams params) : Communication(params, "reduce") {
  validateParams();
}

Reduce::Reduce(TensorView* input_tv, TensorView* output_tv, at::Tensor input, at::Tensor output, 
          BinaryOpType op_type, DeviceIdxType my_device_index, DeviceIdxType root) 
          : Communication("reduce") {
  params_.root = root;
  params_.redOp = getC10dReduceOpType(op_type);
  auto mesh = input_tv->getDeviceMesh();
  params_.team = mesh.vector();
  bool is_root_in_mesh = mesh.has(root);
  if (!is_root_in_mesh) {
    params_.team.push_back(root);
  }

  int sharded_dim = dimWithParallelType(input_tv, ParallelType::DIDx);
  if (mesh.has(my_device_index)) {
    params_.src_bufs = {input.squeeze(sharded_dim)};
  }

  if (my_device_index == root) {
    params_.dst_bufs = {output};
    // The reduce semantics imposes the root to be both
    // sender and receiver. If the root is not in the mesh, we thus
    // have to artificially make it send and receive a dummy buffer
    if (!is_root_in_mesh) {
      at::Tensor dummy = createDummyTensor(output, op_type);
      params_.src_bufs.push_back(dummy);
    }
  }
  validateParams();
}

void Reduce::validateParams() {
  Communication::validateParams();
  assertBuffersHaveSameSize(params_.src_bufs, params_.dst_bufs);
  assertBufferCount(params_.src_bufs, 1);
}

c10::intrusive_ptr<c10d::Work> Reduce::post(
    Communicator& comm,
    std::optional<CommunicatorBackend> backend) {
  if (comm.deviceId() == params_.root) {
    assertBufferCount(params_.dst_bufs, 1);
  } else {
    assertBufferCount(params_.dst_bufs, 0);
  }
  post_common(*this, comm);
  auto& buf =
      (comm.deviceId() == params_.root) ? params_.dst_bufs : params_.src_bufs;
  c10d::ReduceOptions options = {
      .reduceOp = params_.redOp, .rootRank = root_relative_index_};
  auto team_backend = comm.getBackendForTeam(params_.team, backend);
#ifdef USE_C10D_NCCL
  auto nccl_backend = dynamic_cast<c10d::ProcessGroupNCCL*>(team_backend.get());
  if (nccl_backend) {
    return nccl_backend->_reduce_oop(buf, params_.src_bufs, options);
  }
#endif
  if (comm.deviceId() == params_.root) {
    doLocalCopy(params_.dst_bufs.at(0), params_.src_bufs.at(0));
  }
  return team_backend->reduce(buf, options);
}

Allreduce::Allreduce(CommParams params)
    : Communication(params, "allreduce", false) {
  validateParams();
}

Allreduce::Allreduce(TensorView* input_tv, TensorView* output_tv,
                    at::Tensor input, at::Tensor output, BinaryOpType op_type) 
    : Communication("allreduce", false) {
  params_.redOp = getC10dReduceOpType(op_type);
  auto mesh = input_tv->getDeviceMesh();
  params_.team = mesh.vector();
  params_.dst_bufs = {output};
  params_.src_bufs = {input.view(output.sizes())};

  validateParams();
}

void Allreduce::validateParams() {
  Communication::validateParams();
  assertBuffersHaveSameSize(params_.src_bufs, params_.dst_bufs);
  assertBufferCount(params_.src_bufs, 1);
  assertBufferCount(params_.dst_bufs, 1);
}

c10::intrusive_ptr<c10d::Work> Allreduce::post(
    Communicator& comm,
    std::optional<CommunicatorBackend> backend) {
  post_common(*this, comm);
  doLocalCopy(params_.dst_bufs.at(0), params_.src_bufs.at(0));
  return comm.getBackendForTeam(params_.team, backend)
      ->allreduce(params_.dst_bufs, {.reduceOp = params_.redOp});
}

ReduceScatter::ReduceScatter(CommParams params)
    : Communication(params, "reduce_scatter", false) {
  validateParams();
}

ReduceScatter::ReduceScatter(TensorView* input_tv, TensorView* output_tv,
                             at::Tensor input, at::Tensor output, BinaryOpType op_type)
  : Communication("reduce_scatter", false) {
  params_.redOp = getC10dReduceOpType(op_type);
  auto mesh = output_tv->getDeviceMesh();
  params_.team = mesh.vector();
  params_.dst_bufs = {output};

  int sharded_dim = dimWithParallelType(output_tv, ParallelType::DIDx, true);
  int reduction_dim = dimWithParallelType(input_tv, ParallelType::DIDx);

  for (auto i : c10::irange(mesh.vector().size())) {
    std::vector<at::indexing::TensorIndex> indices(input.dim(), at::indexing::Slice());
    indices[sharded_dim] = at::indexing::Slice(i, i+1);
    indices[reduction_dim] = at::indexing::Slice(0,1);
    auto x = input.index(indices).squeeze(sharded_dim).contiguous();
    params_.src_bufs.push_back(x);
  }
  validateParams();
}

void ReduceScatter::validateParams() {
  Communication::validateParams();
  assertBufferCount(params_.src_bufs, params_.team.size());
  assertBufferCount(params_.dst_bufs, 1);
  NVF_ERROR(params_.team.size() > 1, "the team size must be greater than 1");
}

c10::intrusive_ptr<c10d::Work> ReduceScatter::post(
    Communicator& comm,
    std::optional<CommunicatorBackend> backend) {
  post_common(*this, comm);
  // This is used to change the representation of the buffers to match c10d
  // ProcessGroup API
  std::vector<std::vector<at::Tensor>> buf_list = {std::move(params_.src_bufs)};
  auto work = comm.getBackendForTeam(params_.team, backend)
                  ->reduce_scatter(
                      params_.dst_bufs, buf_list, {.reduceOp = params_.redOp});
  params_.src_bufs = std::move(buf_list.back());
  return work;
}

SendRecv::SendRecv(CommParams params) : Communication(params, "send/recv") {
  validateParams();
}

SendRecv::SendRecv(TensorView* input_tv, TensorView* output_tv, at::Tensor input, at::Tensor output, 
  DeviceIdxType my_device_index, DeviceIdxType root, DeviceIdxType receiver) 
    : Communication("send/recv") {
  params_.root = root;
  auto mesh = DeviceMesh({receiver});
  params_.team = mesh.vector();
  if (!mesh.has(root)) {
    params_.team.push_back(root);
  }

  if (my_device_index == root) {
    params_.src_bufs = {input};
  }
  if (mesh.has(my_device_index)) {
    params_.dst_bufs = {output};
  }
  validateParams();
}

void SendRecv::validateParams() {
  Communication::validateParams();
  assertBuffersHaveSameSize(params_.src_bufs, params_.dst_bufs);
  NVF_ERROR(
      params_.team.size() == 1 || params_.team.size() == 2,
      "the team size should be 1 or 2");
}

c10::intrusive_ptr<c10d::Work> SendRecv::post(
    Communicator& comm,
    std::optional<CommunicatorBackend> backend) {
  post_common(*this, comm);

  if (comm.deviceId() == params_.root) {
    assertBufferCount(params_.src_bufs, 1);
    if (params_.team.size() == 1) {
      assertBufferCount(params_.dst_bufs, 1);
      doLocalCopy(params_.dst_bufs.at(0), params_.src_bufs.at(0));
      return nullptr;
    } else {
      assertBufferCount(params_.dst_bufs, 0);
    }
  } else {
    assertBufferCount(params_.src_bufs, 0);
    assertBufferCount(params_.dst_bufs, 1);
  }

  return comm.sendRecv(
      (params_.team.at(0) == params_.root) ? params_.team.at(1)
                                           : params_.team.at(0),
      params_.root,
      params_.dst_bufs.empty() ? params_.src_bufs : params_.dst_bufs,
      backend);
}

} // namespace nvfuser

#endif
