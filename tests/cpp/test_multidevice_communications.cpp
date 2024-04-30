// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <fusion.h>
#include <ir/builder.h>
#include <multidevice/communication.h>
#include <multidevice/communicator.h>
#include <tests/cpp/multidevice.h>

#include <iostream>

namespace nvfuser {

class CommunicationTest
    : public MultiDeviceTest,
      public ::testing::WithParamInterface<CommunicatorBackend> {
 protected:
  CommunicationTest();
  void SetUp() override;

  void validate(at::Tensor obtained, at::Tensor expected);
  void resetDstBuffers();

  static constexpr DeviceIdxType root = 0;
  static constexpr int tensor_size = 1024;
  static constexpr int number_of_repetitions = 8;
  static constexpr c10d::ReduceOp::RedOpType red_op =
      c10d::ReduceOp::RedOpType::SUM;
  CommParams params;
  std::vector<DeviceIdxType> all_ranks;
  c10::intrusive_ptr<c10d::Backend> backend;
  IrContainer container;
};

CommunicationTest::CommunicationTest() {
  all_ranks = std::vector<DeviceIdxType>(communicator->size());
  std::iota(all_ranks.begin(), all_ranks.end(), 0);
}

void CommunicationTest::SetUp() {
  MultiDeviceTest::SetUp();

  if (!communicator->isBackendAvailable(GetParam())) {
    GTEST_SKIP() << "Backend not available";
  }
  backend = communicator->getBackendForTeam(all_ranks, GetParam());
}

void CommunicationTest::validate(at::Tensor obtained, at::Tensor expected) {
  EXPECT_TRUE(obtained.equal(expected))
      << "Device " << communicator->deviceId() << " expected tensor:\n"
      << expected << "\nbut obtained tensor:\n"
      << obtained;
}

void CommunicationTest::resetDstBuffers() {
  for (auto& buf : params.dst_bufs) {
    buf.copy_(at::full(tensor_size, nan(""), tensor_options));
  }
}

TEST_P(CommunicationTest, Gather) {
  params.type = CommunicationType::Gather;
  params.root = root;
  params.team = all_ranks;
  params.src_bufs = {at::empty(tensor_size, tensor_options)};
  if (communicator->deviceId() == root) {
    for (int64_t i = 0; i < communicator->size(); i++) {
      params.dst_bufs.push_back(at::empty(tensor_size, tensor_options));
    }
  }
  auto communication = IrBuilder::create<Communication>(&container, params);

  for (int j : c10::irange(number_of_repetitions)) {
    resetDstBuffers();
    params.src_bufs.at(0).copy_(
        at::arange(tensor_size, tensor_options) +
        (communicator->deviceId() + 1) * j);

    auto work =
        postCommunication(communication, communicator->deviceId(), backend);
    work->wait();

    if (communicator->deviceId() == root) {
      for (int i : c10::irange(communicator->size())) {
        auto obtained = params.dst_bufs.at(i);
        auto ref = at::arange(tensor_size, tensor_options) + (i + 1) * j;
        validate(obtained, ref);
      }
    }
  }
}

TEST_P(CommunicationTest, Allgather) {
  params.type = CommunicationType::Allgather;
  params.team = all_ranks;
  params.src_bufs = {
      at::empty(tensor_size, tensor_options) * communicator->deviceId()};
  for (int64_t i = 0; i < communicator->size(); i++) {
    params.dst_bufs.push_back(at::empty(tensor_size, tensor_options));
  }
  auto communication = IrBuilder::create<Communication>(&container, params);

  for (int j : c10::irange(number_of_repetitions)) {
    resetDstBuffers();
    params.src_bufs.at(0).copy_(
        at::arange(tensor_size, tensor_options) +
        (communicator->deviceId() + 1) * j);

    auto work =
        postCommunication(communication, communicator->deviceId(), backend);
    work->wait();

    for (int i : c10::irange(communicator->size())) {
      auto obtained = params.dst_bufs.at(i);
      auto ref = at::arange(tensor_size, tensor_options) + (i + 1) * j;
      validate(obtained, ref);
    }
  }
}

TEST_P(CommunicationTest, Scatter) {
  params.type = CommunicationType::Scatter;
  params.root = root;
  params.team = all_ranks;
  if (communicator->deviceId() == root) {
    for (int64_t i = 0; i < communicator->size(); i++) {
      params.src_bufs.push_back(
          at::empty(tensor_size, tensor_options) * static_cast<int>(i));
    }
  }
  params.dst_bufs = {at::empty(tensor_size, tensor_options)};
  auto communication = IrBuilder::create<Communication>(&container, params);

  for (int j : c10::irange(number_of_repetitions)) {
    resetDstBuffers();
    for (int i : c10::irange(params.src_bufs.size())) {
      params.src_bufs.at(i).copy_(
          at::arange(tensor_size, tensor_options) + (i + 1) * j);
    }

    auto work =
        postCommunication(communication, communicator->deviceId(), backend);
    work->wait();

    auto obtained = params.dst_bufs.at(0);
    auto ref = at::arange(tensor_size, tensor_options) +
        (communicator->deviceId() + 1) * j;
    validate(obtained, ref);
  }
}

TEST_P(CommunicationTest, Broadcast) {
  params.type = CommunicationType::Broadcast;
  params.root = root;
  params.team = all_ranks;
  if (communicator->deviceId() == root) {
    params.src_bufs = {at::empty(tensor_size, tensor_options)};
  }
  params.dst_bufs = {at::empty(tensor_size, tensor_options)};

  auto communication = IrBuilder::create<Communication>(&container, params);

  for (int j : c10::irange(number_of_repetitions)) {
    resetDstBuffers();
    if (communicator->deviceId() == root) {
      params.src_bufs.at(0).copy_(at::arange(tensor_size, tensor_options) + j);
    }

    auto work =
        postCommunication(communication, communicator->deviceId(), backend);
    if (communicator->size() > 1) {
      work->wait();
    }

    auto obtained = params.dst_bufs.at(0);
    auto ref = at::arange(tensor_size, tensor_options) + j;
    validate(obtained, ref);
  }
}

TEST_P(CommunicationTest, SendRecv) {
  if (GetParam() == CommunicatorBackend::ucc) {
    GTEST_SKIP() << "Disabling because of UCC hangs, see issue #2091";
  }
  if (communicator->size() < 2 || torch::cuda::device_count() < 2) {
    GTEST_SKIP() << "This test needs at least 2 GPUs and 2 ranks.";
  }

  DeviceIdxType sender = 0;
  DeviceIdxType receiver = 1;
  if (communicator->deviceId() > 1) { // only devices 0 and 1 participate
    return;
  }

  params.type = CommunicationType::SendRecv;
  params.root = sender;
  params.team = {0, 1};
  if (communicator->deviceId() == sender) {
    params.src_bufs.push_back(at::empty(tensor_size, tensor_options));
  } else {
    params.dst_bufs.push_back(at::empty(tensor_size, tensor_options));
  }
  auto communication = IrBuilder::create<Communication>(&container, params);

  for (int j : c10::irange(number_of_repetitions)) {
    resetDstBuffers();
    if (communicator->deviceId() == sender) {
      params.src_bufs.at(0).copy_(at::arange(tensor_size, tensor_options) + j);
    }

    auto work =
        postCommunication(communication, communicator->deviceId(), backend);
    work->wait();

    if (communicator->deviceId() == receiver) {
      auto obtained = params.dst_bufs.at(0);
      auto ref = at::arange(tensor_size, tensor_options) + j;
      validate(obtained, ref);
    }
  }
}

TEST_P(CommunicationTest, SendRecvToSelf) {
  DeviceIdxType sender = 0;
  if (communicator->deviceId() > 0) { // only device 0 participates
    return;
  }

  params.type = CommunicationType::SendRecv;
  params.root = sender;
  params.team = {0};
  params.src_bufs.push_back(at::empty(tensor_size, tensor_options));
  params.dst_bufs.push_back(at::empty(tensor_size, tensor_options));
  auto communication = IrBuilder::create<Communication>(&container, params);

  for (int j : c10::irange(number_of_repetitions)) {
    resetDstBuffers();
    params.src_bufs.at(0).copy_(at::arange(tensor_size, tensor_options) + j);

    postCommunication(communication, communicator->deviceId(), backend);

    auto obtained = params.dst_bufs.at(0);
    auto ref = at::arange(tensor_size, tensor_options) + j;
    validate(obtained, ref);
  }
}

TEST_P(CommunicationTest, Reduce) {
  params.type = CommunicationType::Reduce;
  params.redOp = red_op;
  params.root = root;
  params.team = all_ranks;
  params.src_bufs = {at::empty(tensor_size, tensor_options)};
  if (communicator->deviceId() == root) {
    params.dst_bufs = {at::empty(tensor_size, tensor_options)};
  }
  auto communication = IrBuilder::create<Communication>(&container, params);

  for (int j : c10::irange(number_of_repetitions)) {
    resetDstBuffers();
    params.src_bufs.at(0).copy_(
        at::arange(tensor_size, tensor_options) +
        (communicator->deviceId() + 1) * j);

    auto work =
        postCommunication(communication, communicator->deviceId(), backend);
    work->wait();

    if (communicator->deviceId() == root) {
      auto obtained = params.dst_bufs.at(0);
      int S = communicator->size();
      auto ref =
          at::arange(tensor_size, tensor_options) * S + S * (S + 1) / 2 * j;
      validate(obtained, ref);
    }
  }
}

TEST_P(CommunicationTest, Allreduce) {
  params.type = CommunicationType::Allreduce;
  params.redOp = red_op;
  params.team = all_ranks;
  params.src_bufs = {at::empty(tensor_size, tensor_options)};
  params.dst_bufs = {at::empty(tensor_size, tensor_options)};
  auto communication = IrBuilder::create<Communication>(&container, params);

  for (int j : c10::irange(number_of_repetitions)) {
    resetDstBuffers();
    params.src_bufs.at(0).copy_(
        at::arange(tensor_size, tensor_options) +
        (communicator->deviceId() + 1) * j);

    auto work =
        postCommunication(communication, communicator->deviceId(), backend);
    work->wait();

    auto obtained = params.dst_bufs.at(0);
    int S = communicator->size();
    auto ref =
        at::arange(tensor_size, tensor_options) * S + S * (S + 1) / 2 * j;
    validate(obtained, ref);
  }
}

TEST_P(CommunicationTest, ReduceScatter) {
  params.type = CommunicationType::ReduceScatter;
  params.redOp = red_op;
  params.root = root;
  params.team = all_ranks;
  for (int64_t i = 0; i < communicator->size(); i++) {
    params.src_bufs.push_back(at::empty(tensor_size, tensor_options));
  }
  params.dst_bufs = {at::empty(tensor_size, tensor_options)};
  auto communication = IrBuilder::create<Communication>(&container, params);

  for (int j : c10::irange(number_of_repetitions)) {
    resetDstBuffers();
    for (int i : c10::irange(communicator->size())) {
      params.src_bufs.at(i).copy_(
          at::arange(tensor_size, tensor_options) +
          (communicator->deviceId() + 1) * (i + j));
    }

    auto work =
        postCommunication(communication, communicator->deviceId(), backend);
    work->wait();

    auto obtained = params.dst_bufs.at(0);
    int S = communicator->size();
    auto ref = at::arange(tensor_size, tensor_options) * S +
        S * (S + 1) / 2 * (communicator->deviceId() + j);
    validate(obtained, ref);
  }
}

INSTANTIATE_TEST_SUITE_P(
    ,
    CommunicationTest,
    testing::Values(CommunicatorBackend::nccl, CommunicatorBackend::ucc),
    testing::PrintToStringParamName());

} // namespace nvfuser
