// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <gtest/gtest.h>

#include <multidevice/communication.h>
#include <multidevice/communicator.h>
#include <test/multidevice.h>

#include <iostream>

namespace nvfuser {

static constexpr DeviceIdxType root = 0;
static constexpr int tensor_size = 1024;
static constexpr int number_of_repetitions = 8;
Communicator* MultiDeviceTest::communicator = nullptr;

TEST_F(MultiDeviceTest, Communication_Gather) {
  if (!communicator->is_available() || communicator->size() < 2) {
    GTEST_SKIP() << "This test needs at least 2 ranks";
  }
  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(communicator->device());

  CommParams params;
  params.root = root;
  params.team = std::vector<DeviceIdxType>(communicator->size());
  std::iota(params.team.begin(), params.team.end(), 0);
  params.src_bufs = {at::empty(tensor_size, options)};
  if (communicator->deviceId() == root) {
    for (uint64_t i = 0; i < communicator->size(); i++) {
      params.dst_bufs.push_back(at::empty(tensor_size, options));
    }
  }
  auto communication = Gather(params);

  for (int j : c10::irange(number_of_repetitions)) {
    params.src_bufs.at(0).copy_(
        at::arange(tensor_size, options) + (communicator->deviceId() + 1) * j);
    for (auto& buf : params.dst_bufs) {
      buf.copy_(at::zeros(tensor_size, options));
    }

    auto work = communication.post(*communicator);
    work->wait();

    if (communicator->deviceId() == root) {
      for (int i : c10::irange(communicator->size())) {
        auto obtained = params.dst_bufs.at(i);
        auto ref = at::arange(tensor_size, options) + (i + 1) * j;
        NVF_ERROR(
            at::equal(obtained, ref),
            "Device ",
            communicator->deviceId(),
            " expected tensor:\n",
            ref,
            "\nbut obtained tensor:\n",
            obtained);
      }
    }
  }
  communicator->barrier();
}

TEST_F(MultiDeviceTest, Communication_Allgather) {
  if (!communicator->is_available() || communicator->size() < 2) {
    GTEST_SKIP() << "This test needs at least 2 ranks";
  }
  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(communicator->device());

  CommParams params;
  params.team = std::vector<DeviceIdxType>(communicator->size());
  std::iota(params.team.begin(), params.team.end(), 0);
  params.src_bufs = {at::empty(tensor_size, options) * communicator->deviceId()};
  for (uint64_t i = 0; i < communicator->size(); i++) {
    params.dst_bufs.push_back(at::empty(tensor_size, options));
  }
  auto communication = Allgather(params);

  for (int j : c10::irange(number_of_repetitions)) {
    params.src_bufs.at(0).copy_(
        at::arange(tensor_size, options) + (communicator->deviceId() + 1) * j);
    for (auto& buf : params.dst_bufs) {
      buf.copy_(at::zeros(tensor_size, options));
    }

    auto work = communication.post(*communicator);
    work->wait();

    for (int i : c10::irange(communicator->size())) {
      auto obtained = params.dst_bufs.at(i);
      auto ref = at::arange(tensor_size, options) + (i + 1) * j;
      NVF_ERROR(
          obtained.equal(ref),
          "Device",
          communicator->deviceId(),
          " expected tensor:\n",
          ref,
          "\nbut obtained tensor:\n",
          obtained);
    }
  }
  communicator->barrier();
}

TEST_F(MultiDeviceTest, Communication_Scatter) {
  if (!communicator->is_available() || communicator->size() < 2) {
    GTEST_SKIP() << "This test needs at least 2 ranks";
  }
  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(communicator->device());

  CommParams params;
  params.root = root;
  params.team = std::vector<DeviceIdxType>(communicator->size());
  std::iota(params.team.begin(), params.team.end(), 0);
  if (communicator->deviceId() == root) {
    for (uint64_t i = 0; i < communicator->size(); i++) {
      params.src_bufs.push_back(
          at::empty(tensor_size, options) * static_cast<int>(i));
    }
  }
  params.dst_bufs = {at::empty(tensor_size, options)};
  auto communication = Scatter(params);

  for (int j : c10::irange(number_of_repetitions)) {
    params.dst_bufs.at(0).copy_(at::zeros(tensor_size, options));
    for (int i : c10::irange(params.src_bufs.size())) {
      params.src_bufs.at(i).copy_(
          at::arange(tensor_size, options) + (i + 1) * j);
    }

    auto work = communication.post(*communicator);
    work->wait();

    auto obtained = params.dst_bufs.at(0);
    auto ref = at::arange(tensor_size, options) + (communicator->deviceId() + 1) * j;
    NVF_ERROR(
        obtained.equal(ref),
        "Device",
        communicator->deviceId(),
        " expected tensor:\n",
        ref,
        "\nbut obtained tensor:\n",
        obtained);
  }
  communicator->barrier();
}

TEST_F(MultiDeviceTest, Communication_Broadcast) {
  if (!communicator->is_available()) {
    GTEST_SKIP() << "This test needs distributed setting";
  }
  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(communicator->device());

  CommParams params;
  params.root = root;
  params.team = std::vector<DeviceIdxType>(communicator->size());
  std::iota(params.team.begin(), params.team.end(), 0);
  if (communicator->deviceId() == root) {
    params.src_bufs = {at::empty(tensor_size, options)};
  }
  params.dst_bufs = {at::empty(tensor_size, options)};

  auto communication = Broadcast(params);

  for (int j : c10::irange(number_of_repetitions)) {
    if (communicator->deviceId() == root) {
      params.src_bufs.at(0).copy_(at::arange(tensor_size, options) + j);
    }
    params.dst_bufs.at(0).copy_(at::zeros(tensor_size, options));

    auto work = communication.post(*communicator);
    if (communicator->size() > 1) {
      work->wait();
    }

    auto obtained = params.dst_bufs.at(0);
    auto ref = at::arange(tensor_size, options) + j;
    NVF_ERROR(
        obtained.equal(ref),
        "Device",
        communicator->deviceId(),
        " expected tensor:\n",
        ref,
        "\nbut obtained tensor:\n",
        obtained);
  }
  communicator->barrier();
}

TEST_F(MultiDeviceTest, Communication_SendRecv) {
  DeviceIdxType sender = 0;
  DeviceIdxType receiver = 1;
  if (!communicator->is_available() || communicator->size() < 2) {
    GTEST_SKIP() << "This test needs at least 2 ranks";
  }
  if (communicator->deviceId() > 1) { // only devices 0 and 1 participate
    communicator->barrier();
    return;
  }
  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(communicator->device());

  CommParams params;
  params.root = sender;
  params.team = {0, 1};
  if (communicator->deviceId() == sender) {
    params.src_bufs.push_back(at::empty(tensor_size, options));
  } else {
    params.dst_bufs.push_back(at::empty(tensor_size, options));
  }
  auto communication = SendRecv(params);

  for (int j : c10::irange(number_of_repetitions)) {
    if (communicator->deviceId() == sender) {
      params.src_bufs.at(0).copy_(at::arange(tensor_size, options) + j);
    } else {
      params.dst_bufs.at(0).copy_(at::zeros(tensor_size, options));
    }

    auto work = communication.post(*communicator);
    work->wait();

    if (communicator->deviceId() == receiver) {
      auto obtained = params.dst_bufs.at(0);
      auto ref = at::arange(tensor_size, options) + j;
      NVF_ERROR(
          obtained.equal(ref),
          "Device",
          communicator->deviceId(),
          " expected tensor:\n",
          ref,
          "\nbut obtained tensor:\n",
          obtained);
    }
  }
  communicator->barrier();
}

TEST_F(MultiDeviceTest, Communication_SendRecvToSelf) {
  DeviceIdxType sender = 0;
  if (!communicator->is_available()) {
    GTEST_SKIP() << "This test needs distributed setting";
  }
  if (communicator->deviceId() > 0) { // only device 0 participates
    communicator->barrier();
    return;
  }
  c10::TensorOptions options =
      at::TensorOptions().dtype(at::kFloat).device(communicator->device());

  CommParams params;
  params.root = sender;
  params.team = {0};
  params.src_bufs.push_back(at::empty(tensor_size, options));
  params.dst_bufs.push_back(at::empty(tensor_size, options));
  auto communication = SendRecv(params);

  for (int j : c10::irange(number_of_repetitions)) {
    params.src_bufs.at(0).copy_(at::arange(tensor_size, options) + j);
    params.dst_bufs.at(0).copy_(at::zeros(tensor_size, options));

    communication.post(*communicator);

    auto obtained = params.dst_bufs.at(0);
    auto ref = at::arange(tensor_size, options) + j;
    NVF_ERROR(
        obtained.equal(ref),
        "Device",
        communicator->deviceId(),
        " expected tensor:\n",
        ref,
        "\nbut obtained tensor:\n",
        obtained);
  }
  communicator->barrier();
}

} // namespace nvfuser

#endif
