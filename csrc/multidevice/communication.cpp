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
#include <torch/csrc/distributed/c10d/Types.hpp>

#include <multidevice/communication.h>

namespace nvfuser {
namespace {

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

// TODO: add `c10d::reduceOp::AVG` and `c10d::reduceOp::PREMUL_SUM`
inline c10d::reduceOp getC10dReduceOp(BinaryOpType op) {
  switch (op) {
  case BinaryOpType::Add:
    return c10d::reduceOp::SUM;
  case BinaryOpType::Mul:
    return c10d::reduceOp::PRODUCT;
  case BinaryOpType::Min:
    return c10d::reduceOp::MIN;
  case BinaryOpType::Max:
    return c10d::reduceOp::MAX;
  case BinaryOpType::BitwiseAnd:
    return c10d::reduceOp::BAND;
  case BinaryOpType::BitwiseOr:
    return c10d::reduceOp::BOR;
  case BinaryOpType::BitwiseXor:
    return c10d::reduceOp::BXOR;
  default:
    NVF_ERROR(false, "unsupported reduction operation");
    return c10d::reduceOp::UNUSED;
  }
}

} // namespace

Communication::Communication(CommParams params, std::string name, bool has_root)
    : params_(std::move(params)),
      collective_type_(std::move(name)),
      has_root_(has_root) {
  assertBuffersHaveSameSize(params_.src_bufs, params_.dst_bufs);
  NVF_ERROR(
      std::unique(params_.team.begin(), params_.team.end()) ==
          params_.team.end(),
      "the communication must not involve the same device more than once");
  NVF_ERROR(params_.team.size() > 0, "the team size must be greater than 0");
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

Broadcast::Broadcast(CommParams params) : Communication(params, "broadcast") {}

c10::intrusive_ptr<c10d::Work> Broadcast::post(Communicator& comm) {
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

  return comm.getBackendForTeam(params_.team)
      ->broadcast(
          comm.deviceId() == params_.root ? params_.src_bufs : params_.dst_bufs,
          {.rootRank = root_relative_index_});
}

Gather::Gather(CommParams params) : Communication(params, "gather") {
  assertBufferCount(params_.src_bufs, 1);
  NVF_ERROR(params_.team.size() > 1, "the team size must be greater than 1");
}

c10::intrusive_ptr<c10d::Work> Gather::post(Communicator& comm) {
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
      comm.getBackendForTeam(params_.team)
          ->gather(
              buf_list, params_.src_bufs, {.rootRank = root_relative_index_});
  if (comm.deviceId() == params_.root) {
    params_.dst_bufs = std::move(buf_list.back());
  }
  return work;
}

Allgather::Allgather(CommParams params)
    : Communication(params, "allgather", false) {
  assertBufferCount(params_.src_bufs, 1);
  assertBufferCount(params_.dst_bufs, params_.team.size());
  NVF_ERROR(params_.team.size() > 1, "the team size must be greater than 1");
}

c10::intrusive_ptr<c10d::Work> Allgather::post(Communicator& comm) {
  post_common(*this, comm);
  // This is used to change the representation of the buffers to match c10d
  // ProcessGroup API
  std::vector<std::vector<at::Tensor>> buf_list;
  buf_list = {std::move(params_.dst_bufs)};
  auto work = comm.getBackendForTeam(params_.team)
                  ->allgather(buf_list, params_.src_bufs, {});
  params_.dst_bufs = std::move(buf_list.back());
  return work;
}

Scatter::Scatter(CommParams params) : Communication(params, "scatter") {
  assertBufferCount(params_.dst_bufs, 1);
  NVF_ERROR(params_.team.size() > 1, "the team size must be greater than 1");
}

c10::intrusive_ptr<c10d::Work> Scatter::post(Communicator& comm) {
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
      comm.getBackendForTeam(params_.team)
          ->scatter(
              params_.dst_bufs, buf_list, {.rootRank = root_relative_index_});
  if (comm.deviceId() == params_.root) {
    params_.src_bufs = std::move(buf_list.back());
  }
  return work;
}

Reduce::Reduce(CommParams params) : Communication(params, "reduce") {
  assertBufferCount(params_.src_bufs, 1);
  // assertBufferCount(params_.dst_bufs, 0);
  NVF_ERROR(params_.team.size() > 1, "the team size must be greater than 1");
}

c10::intrusive_ptr<c10d::Work> Reduce::post(Communicator& comm) {
  post_common(*this, comm);
  auto backend = comm.getBackendForTeam(params_.team);
  auto nccl_backend = dynamic_cast<c10d::ProcessGroupNCCL*>(backend.get());
  auto& buf = (comm.deviceId() == params_.root)? params_.dst_bufs : params_.src_bufs;
  c10d::ReduceOptions options = {.rootRank = root_relative_index_,
                                 .reduceOp = getC10dReduceOp(params_.RedOp)};
  if (nccl_backend) {
    return nccl_backend->_reduce_oop(buf, params_.src_bufs, {.rootRank = root_relative_index_}); 
  } else {
    if (comm.deviceId() == params_.root) {
      doLocalCopy(params_.dst_bufs.at(0), params_.src_bufs.at(0));
    }
    return backend->reduce(buf, {.rootRank = root_relative_index_});
  }
// fill also .reduceOp,  https://github.com/pytorch/pytorch/blob/c36b31d5302d31746f3f3bd64ed8d9acd8e36155/torch/csrc/distributed/c10d/Types.hpp#L123
}

SendRecv::SendRecv(CommParams params) : Communication(params, "send/recv") {
  NVF_ERROR(
      params_.team.size() == 1 || params_.team.size() == 2,
      "the team size should be 1 or 2");
}

c10::intrusive_ptr<c10d::Work> SendRecv::post(Communicator& comm) {
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
      params_.dst_bufs.empty() ? params_.src_bufs : params_.dst_bufs);
}

} // namespace nvfuser

#endif
