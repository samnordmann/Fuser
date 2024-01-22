// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <disjoint_set.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <gtest/gtest.h>
#include <multidevice/executor.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <test/multidevice.h>
#include <test/validator.h>

namespace nvfuser {
namespace {

inline at::Tensor shardInputTensor(at::Tensor tensor, std::vector<int64_t>& devices, int deviceId) {
  int i = 0;
  auto it = find (devices.begin(), devices.end(), deviceId);
  if (it != devices.end()) {
    i = *it;
  }
  return tensor.index({at::indexing::Slice(i, i+1), "..."});
}
} // namespace

class ShardingTest
    : public MultiDeviceTest,
      public ::testing::WithParamInterface<std::tuple<bool, int>> {
};

TEST_P(ShardingTest, UnshardedGlobalInput) {
  auto [concreteTV, sharded_dim] = GetParam();
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int num_devices = communicator->size();
  std::vector<int64_t> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  DeviceMesh mesh(devices);
  std::vector<int64_t> input_size = {3, 3};
  input_size[sharded_dim] = num_devices;

  TensorView* tv0 = concreteTV ? makeConcreteTensor(input_size) : makeContigTensor(2);
  TensorView* tv1 = add(tv0, tv0);
  TensorView* tv2 = set(tv1);
  TensorView* tv3 = add(tv2, tv2);
  TensorView* tv4 = set(tv3);
  TensorView* tv5 = sum(tv4, {0});
  fusion->addInput(tv0);
  fusion->addOutput(tv5);

  // TODO: split
  // tv2->split(sharded_dim, num_devices, false);
  tv2->axis(sharded_dim)->parallelize(ParallelType::DIDx);
  // tv3->split(sharded_dim, num_devices, false);
  tv3->axis(sharded_dim)->parallelize(ParallelType::DIDx);

  std::vector<TensorView*> tvs = {tv0, tv1, tv2, tv3, tv4, tv5};
  for (auto tv : tvs) {
    tv->setDeviceMesh(mesh);
  }

  auto x = at::randn(input_size, tensor_options);
  std::vector<c10::IValue> inputs = {x};
  auto ref_outputs = at::sum(x * 4, {0});

  MultiDeviceExecutor runtime(std::move(fusion), *communicator);
  auto outputs = runtime.runWithInput(inputs);
  testValidate(
      runtime.fusion(), outputs, inputs, {ref_outputs}, __LINE__, __FILE__);
}

TEST_P(ShardingTest, ShardGlobalInput) {
  auto [concreteTV, sharded_dim] = GetParam();
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int num_devices = communicator->size();
  std::vector<int64_t> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  DeviceMesh mesh(devices);
  std::vector<int64_t> unsharded_input_size = {num_devices, 3, 2};

  TensorView* tv0 = concreteTV ? makeConcreteTensor(unsharded_input_size) : makeContigTensor(2);
  TensorView* tv1 = set(tv0);
  TensorView* tv2 = add(tv1, tv1);
  fusion->addInput(tv0);
  fusion->addOutput(tv2);

  tv0->axis(sharded_dim)->parallelize(ParallelType::DIDx);

  std::vector<TensorView*> tvs = {tv0, tv1, tv2};
  for (auto tv : tvs) {
    tv->setDeviceMesh(mesh);
  }

  auto x = at::randn(unsharded_input_size, tensor_options);
  std::vector<c10::IValue> inputs = {
      shardInputTensor(x, devices, communicator->deviceId())};
  auto ref_outputs = x * 2;

  MultiDeviceExecutor runtime(std::move(fusion), *communicator);
  auto outputs = runtime.runWithInput(inputs);
  testValidate(
      runtime.fusion(), outputs, inputs, {ref_outputs}, __LINE__, __FILE__);
}

INSTANTIATE_TEST_SUITE_P(
    NonoutermostAxis,
    ShardingTest,
    ::testing::Combine(
      ::testing::Values(true),
      ::testing::Values(1))
);


} // namespace nvfuser
#endif
