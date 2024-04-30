// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#pragma once

#include <ir/base_nodes.h>
#include <ir/builder.h>
#include <multidevice/communicator.h>
#include <multidevice/multidevice.h>
#include <torch/csrc/distributed/c10d/Types.hpp>
#include <type.h>
#include <visibility.h>

namespace nvfuser {

enum class CommunicationType {
  Gather,
  Allgather,
  Scatter,
  Reduce,
  Allreduce,
  ReduceScatter,
  Broadcast,
  SendRecv
};

/*
  This struct gathers all the parameters necessary for the
  construction a communication
*/
struct CommParams {
  CommunicationType type;
  DeviceIdxType root = -1;
  std::vector<at::Tensor> src_bufs;
  std::vector<at::Tensor> dst_bufs;
  Team team; // should not have duplicate
  c10d::ReduceOp::RedOpType redOp = c10d::ReduceOp::RedOpType::UNUSED;
};

/*
The class "Communication" represents a MPI-style communication
communication operation to be executed on the network. The base class
Communication should not be used directly but through its derived classes:
Broadcast, Gather, Scatter, Allgather, and SendRecv. Other collectives will be
added later.

CommParams contains the arguments for the communication constructors.
Note that each process (associated with a device index given by
communicator.deviceId()) will fill CommParams with different arguments,
depending on the role they play in this communication. For example, the root of
a Gather communication will have <team_size> destination buffers, whereas
non-root will have no destination buffers. Also, the ranks not participating in
the communication should not instantiate it.

The method "post" triggers the execution of the communication. This call is
non-blocking. The communication can be posted multiple times.
It is assumed that the current device_index (given by
communicator.deviceId()) belongs to the team of the communication,
otherwise an error is thrown.

NOTE: pytorch's NCCL process group API needs <team_size> buffers on root for
scatter/gather operation.

(*) Broadcast
Copies the root's src buffer to each device's dst buffer
Requirements:
  - the root is set and belongs to the team
  - the root has one src buffer, and no or one dst buffer
  - non-roots have no src buffer and one dst buffer
  - all buffers have the same size


(*) Gather
Copies each device's source buffer to the root's respective src
buffer. The order of the sender devices matches the order of the
root's buffers.
Requirements:
  - the root is set and belongs to the team
  - the root has one src buffer and <team_size> dst buffers
  - non-roots have one src buffer and no dst buffer
  - all buffers have the same size

(*) Allgather
Copies each device's src buffer to each device's respective src
buffer. The order of the devices matches the order of the
buffers
Requirements:
  - all device have one src buffer and <team_size> dst buffers
  - all buffers have the same size

(*) Scatter
Copies each root's src buffer to each device's dst buffer.
The order of the buffers matches the order of the receiver devices
Requirements:
  - the root is set and belongs to the team
  - the root has <team_size> src buffers and one dst buffer
  - non-roots have no src buffer and one dst buffer
  - all buffers have the same size

(*) Reduce
Reduce the src buffers to the root's dst buffer.
Requirements:
  - the root is set and belongs to the team
  - the root has one src buffers and one dst buffer
  - non-roots have one src buffer and no dst buffer
  - all buffers have the same size

(*) Allreduce
Reduce the src buffers to the dst buffer.
Requirements:
  - all devices have one src buffer and one dst buffer
  - all buffers have the same size

(*) ReduceScatter
Reduce all the src buffers and shard the result to the dst buffers.
Requirements:
  - all devices have <team_size> src buffer and one dst buffer
  - all buffers have the same size

(*) SendRecv
Copies the sender's src buffers to the receiver's dst buffer
It is equivalent to a Broadcast with a team of size == 2

Requirements:
  - the team must be of size 2 or 1 (in which case the SendRecv reduces to a
local copy)
  - all buffers have the same size
  - the root is set and belongs to the team. The "root" corresponds to the
sender
  - If the team size the root has one src buffers and no dst buffer (or one in
case of a local copy)
  - If team is of size 2, the unique non-root have no src buffer and one dst
buffer
*/

class NVF_API Communication : public Expr {
 public:
  using Expr::Expr;
  Communication(IrBuilderPasskey passkey, CommParams params);
  Communication(const Communication* src, IrCloner* ir_cloner);

  Communication(const Communication& other) = delete;
  Communication& operator=(const Communication& other) = delete;
  Communication(Communication&& other) = delete;
  Communication& operator=(Communication&& other) = delete;

  NVFUSER_DECLARE_CLONE_AND_CREATE

  std::string toString(int indent_size = 0) const override;
  std::string toInlineString(int indent_size = 0) const override;
  const char* getOpString() const override {
    return "Communication";
  }

  bool sameAs(const Statement* other) const override;

  auto params() const {
    return params_;
  }

 private:
  // store the arguments of the communication
  CommParams params_;
};

// Triggers the execution of the communication. This is a non-blocking call.
// The communication can be posted multiple times
c10::intrusive_ptr<c10d::Work> postCommunication(
    Communication* communication,
    DeviceIdxType my_device_index,
    c10::intrusive_ptr<c10d::Backend> backend);

} // namespace nvfuser
