// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef NVFUSER_DISTRIBUTED
#include <gtest/gtest.h>

#include <codegen.h>
#include <device_lower/lower2device.h>
#include <disjoint_set.h>
#include <executor.h>
#include <executor_params.h>
#include <expr_evaluator.h>
#include <fusion.h>
#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <ir/graphviz.h>
#include <ir/iostream.h>
#include <ir/printer.h>
#include <ir/utils.h>
#include <iter_visitor.h>
#include <kernel_cache.h>
#include <kernel_ir.h>
#include <mma_type.h>
#include <ops/all_ops.h>
#include <root_domain_map.h>
#include <scheduler/all_schedulers.h>
#include <scheduler/reduction_utils.h>
#include <scheduler/utils.h>
#include <test/multidevice.h>
#include <test/validator.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <transform_replay.h>
#include <transform_rfactor.h>

#include <algorithm>
#include <iostream>

namespace nvfuser {

using namespace torch::jit::fuser::cuda;
using namespace at::indexing;

/* To run the following tests on several devices, pytorch must be installed
   with the flag NVFUSER_DISTRIBUTED=1 and nccl support.
   Then simply run the tests on several processes, for example using mpirun
   on a node having at least 6 GPUs,
   e.g.: mpirun -np 6 build/nvfuser_tests
   --gtest_filter=PipelineTest.Pipeline
*/

TEST_F(PipelineTest, Pipeline) {
  const std::vector<int64_t> input_shape1 = {6, 7};
  const std::vector<int64_t> input_shape2 = {3, 5, 2};
  // ===========================================================
  //        FUSION
  // ===========================================================
  FusionGuard fg(fusion.get());

  TensorView* tv0_ = makeConcreteTensor(input_shape1);
  fusion->addInput(tv0_);
  TensorView* tv1_ = sum(tv0_, {0});

  TensorView* tv2_ = set(tv1_);
  TensorView* tv3_ = sum(tv2_, {0});
  fusion->addOutput(tv3_);

  TensorView* tv0 = makeConcreteTensor(input_shape2);
  fusion->addInput(tv0);
  TensorView* tv1 = sum(tv0, {0});

  TensorView* tv2 = set(tv1);
  TensorView* tv2a = set(tv2);
  TensorView* tv3 = sum(tv2a, {0});

  TensorView* tv4 = set(tv1);
  TensorView* tv4a = set(tv4);
  TensorView* tv5 = sum(tv4a, {0});

  TensorView* tv6 = set(tv2a);
  TensorView* tv7 = sum(tv6, {0});
  fusion->addOutput(tv7);

  TensorView* tv8 = set(tv3);
  TensorView* tv9 = set(tv5);
  TensorView* tv10 = set(tv7);
  TensorView* tv11 = add(tv8, tv9);
  TensorView* tv12 = add(tv11, tv10);
  TensorView* tv13 = sum(tv12, {0});
  fusion->addOutput(tv13);

  DeviceMesh mesh0_({5});
  DeviceMesh mesh1_({2, 4});
  DeviceMesh mesh0({0});
  DeviceMesh mesh1({0, 1, 4});
  DeviceMesh mesh2({1, 3});
  DeviceMesh mesh3({2});
  DeviceMesh mesh4({4, 5});

  tv0_->setDeviceMesh(mesh0_);
  tv1_->setDeviceMesh(mesh0_);
  tv2_->setDeviceMesh(mesh1_);
  tv3_->setDeviceMesh(mesh1_);
  tv0->setDeviceMesh(mesh0);
  tv1->setDeviceMesh(mesh0);
  tv2->setDeviceMesh(mesh1);
  tv2a->setDeviceMesh(mesh1);
  tv3->setDeviceMesh(mesh1);
  tv4->setDeviceMesh(mesh2);
  tv4a->setDeviceMesh(mesh2);
  tv5->setDeviceMesh(mesh2);
  tv6->setDeviceMesh(mesh3);
  tv7->setDeviceMesh(mesh3);
  tv8->setDeviceMesh(mesh4);
  tv9->setDeviceMesh(mesh4);
  tv10->setDeviceMesh(mesh4);
  tv11->setDeviceMesh(mesh4);
  tv12->setDeviceMesh(mesh4);
  tv13->setDeviceMesh(mesh4);

  // Create input tensors.
  // Note: each process is binded to a different GPU
  // Note: the concrete values are only used at the relevant ranks
  unsharded_inputs = {
      at::randn(input_shape1, tensor_options),
      at::randn(input_shape2, tensor_options)};

  executeAndValidate();
}

//(backend type, first stage's mesh, second stage's mesh (if not null), is first
// stage sharded?, is second
// stage sharded?, do_reduction?)
using PipelineTestTwoStagesParams =
    std::tuple<CommunicatorBackend, DeviceMesh, DeviceMesh, bool, bool, bool>;
class PipelineTestTwoStages
    : public PipelineTest,
      public ::testing::WithParamInterface<PipelineTestTwoStagesParams> {};

TEST_P(PipelineTestTwoStages, Communication) {
  auto
      [backend,
       mesh0,
       mesh1,
       is_stage0_sharded,
       is_stage1_sharded,
       do_reduction] = GetParam();
  if (!disable_skip && !communicator->isBackendAvailable(backend)) {
    GTEST_SKIP() << "Backend not available";
  }
  communicator->setDefaultBackend(backend);

  if (mesh1.vector().empty()) {
    mesh1 = mesh0;
  }

  int first_axis_extent = 3;
  if (is_stage0_sharded) {
    first_axis_extent = mesh0.vector().size();
  } else if (is_stage1_sharded) {
    first_axis_extent = mesh1.vector().size();
  }
  int second_axis_extent = 2;
  if (is_stage1_sharded && do_reduction) {
    GTEST_ASSERT_EQ(mesh0.vector().size(), mesh1.vector().size());
    second_axis_extent = mesh1.vector().size();
  }
  std::vector<int64_t> unsharded_input_sizes = {
      first_axis_extent, second_axis_extent, 3, 5};

  FusionGuard fg(fusion.get());
  TensorView* tv0 = makeConcreteTensor(unsharded_input_sizes);
  TensorView* tv1 = sum(tv0, {3});
  TensorView* tv2 = do_reduction ? sum(tv1, {0}) : set(tv1);
  TensorView* tv3 = sum(tv2, {1});
  fusion->addInput(tv0);
  fusion->addOutput(tv3);

  tv0->setDeviceMesh(mesh0);
  tv1->setDeviceMesh(mesh0);
  tv2->setDeviceMesh(mesh1);
  tv3->setDeviceMesh(mesh1);
  if (is_stage0_sharded) {
    tv0->axis(0)->parallelize(ParallelType::DIDx);
    tv1->axis(0)->parallelize(ParallelType::DIDx);
  }
  if (is_stage1_sharded) {
    // in case of reduction, axis(0) of tv2 is a reduction axis, except if it
    // was initially of size 1, in which case it is simply removed.
    int tv2_outmost_axis = (do_reduction && second_axis_extent > 1) ? 1 : 0;
    tv2->axis(tv2_outmost_axis)->parallelize(ParallelType::DIDx);
    tv3->axis(0)->parallelize(ParallelType::DIDx);
  }

  unsharded_inputs = {at::randn(unsharded_input_sizes, tensor_options)};

  executeAndValidate();
}

namespace {
auto all_backends =
    ::testing::Values(CommunicatorBackend::nccl, CommunicatorBackend::ucc);

DeviceMesh mesh_null;
DeviceMesh mesh0({0});
DeviceMesh mesh1({1});
DeviceMesh mesh2({0, 1, 2, 3});
DeviceMesh mesh3({0, 2, 3});
DeviceMesh mesh4({1, 0, 2});
auto all_meshes = ::testing::Values(mesh0, mesh1, mesh2, mesh3, mesh4);
auto all_nontrivial_meshes = ::testing::Values(mesh2, mesh3, mesh4);

} // namespace

INSTANTIATE_TEST_SUITE_P(
    Gather,
    PipelineTestTwoStages,
    ::testing::Combine(
        all_backends,
        all_meshes,
        all_meshes,
        ::testing::Values(true),
        ::testing::Values(false),
        ::testing::Values(false)));

INSTANTIATE_TEST_SUITE_P(
    Scatter,
    PipelineTestTwoStages,
    ::testing::Combine(
        all_backends,
        all_meshes,
        all_meshes,
        ::testing::Values(false),
        ::testing::Values(true),
        ::testing::Values(false)));

INSTANTIATE_TEST_SUITE_P(
    Bcast,
    PipelineTestTwoStages,
    ::testing::Combine(
        all_backends,
        all_meshes,
        all_meshes,
        ::testing::Values(false),
        ::testing::Values(false),
        ::testing::Values(false)));

INSTANTIATE_TEST_SUITE_P(
    Bcast_sharded,
    PipelineTestTwoStages,
    ::testing::Combine(
        all_backends,
        ::testing::Values(mesh3, mesh4),
        ::testing::Values(mesh3, mesh4),
        ::testing::Values(true),
        ::testing::Values(true),
        ::testing::Values(false)));

INSTANTIATE_TEST_SUITE_P(
    Bcast_sharded_same_mesh,
    PipelineTestTwoStages,
    ::testing::Combine(
        all_backends,
        ::testing::Values(mesh0, mesh1),
        ::testing::Values(mesh_null), // the same mesh is used for all tensors
        ::testing::Values(true),
        ::testing::Values(true),
        ::testing::Values(false)));

INSTANTIATE_TEST_SUITE_P(
    Reduce,
    PipelineTestTwoStages,
    ::testing::Combine(
        all_backends,
        all_nontrivial_meshes,
        all_meshes,
        ::testing::Values(true),
        ::testing::Values(false),
        ::testing::Values(true)));

INSTANTIATE_TEST_SUITE_P(
    ReduceScatter,
    PipelineTestTwoStages,
    ::testing::Combine(
        all_backends,
        all_nontrivial_meshes,
        ::testing::Values(mesh_null), // the same mesh is used for all tensors
        ::testing::Values(true),
        ::testing::Values(true),
        ::testing::Values(true)));

TEST_F(PipelineTest, Overlap) {
  // In this example we demonstrate how we can apply the optimization
  // described in
  // Overlap Communication with Dependent Computation via Decomposition in Large
  // Deep Learning Models (acm.org)
  // https://dl.acm.org/doi/pdf/10.1145/3567955.3567959
  // We simplify the setting as much as possible by considering a multi-device
  // Pipeline with a simple "Gather" followed by a dependent compute. The paper
  // suggest to slice those two operation and to interleave them to achieve
  // better overlap. Consider the following Pipeline:

  // /* Stage 0 */
  // TensorView* tv0 = makeContigTensor(3);
  // fusion->addInput(tv0);
  // TensorView* tv1 = sum(tv0, {2});
  // /* Stage 1 */
  // TensorView* tv2 = set(tv1); // is lowered to a "Gather" communication
  // TensorView* tv3 = sum(tv2, {1});
  // fusion->addOutput(tv3);

  // tv0->axis(0)->parallelize(ParallelType::DIDx);
  // tv1->axis(0)->parallelize(ParallelType::DIDx);

  // PipelineStageDescriptor stage0(false), stage1(false);
  // stage0.addVal({tv0, tv1});
  // stage1.addVal({tv2, tv3});

  // stage0.mesh = {0, 1, 2, 3, 4, 5, 6, 7};
  // stage1.mesh = {0};

  const int64_t number_of_devices = communicator->size();
  constexpr int64_t number_of_slices = 4;
  constexpr int64_t extent_of_axis2 = 1024;
  constexpr int64_t extent_of_slice = extent_of_axis2 / number_of_slices;
  const std::vector<int64_t> input_extents = {
      number_of_devices, 7, extent_of_axis2, 3};
  assert(!(extent_of_axis2 % number_of_slices)); // for simplicity

  FusionGuard fg(fusion.get());

  // containers used later for adding ranges of tvs directly to the stages
  std::unordered_set<Val*> from_stage0, from_stage1;
  std::vector<Val*> to_stage0, to_stage1;

  TensorView* tv0 = makeConcreteTensor(input_extents);
  fusion->addInput(tv0);
  TensorView* tv1 = sum(tv0, {3});
  from_stage0.insert(tv0);
  tv0->axis(0)->parallelize(ParallelType::DIDx);
  tv1->axis(0)->parallelize(ParallelType::DIDx);

  TensorView *tv1x, *tv2x, *tv3x;
  std::vector<TensorView*> tv3_slices;
  std::vector<Slice> slices{3};
  for (int i = 0; i < number_of_slices; i++) {
    slices.at(2).start = IrBuilder::create<Val>(i * extent_of_slice);
    slices.at(2).stop = IrBuilder::create<Val>((i + 1) * extent_of_slice);
    tv1x = slice(tv1, slices);
    tv1x->axis(0)->parallelize(ParallelType::DIDx);
    to_stage0.push_back(tv1x);

    tv2x = set(tv1x);
    from_stage1.insert(tv2x);
    tv3x = sum(tv2x, {1});
    tv3_slices.push_back(tv3x);
  }
  TensorView* tv3 = cat(tv3_slices, 1);
  fusion->addOutput(tv3);
  to_stage1.push_back(tv3);

  // instead of using "slice/cat" it would be nicer to split the dimension and
  // use "select/stack", but "stack" is not implemented in nvFuser at the moment

  // stage0.addRange(from_stage0, to_stage0);
  // stage1.addRange(from_stage1, to_stage1);
  // std::vector<DeviceIdxType> devices(number_of_devices);
  // std::iota(devices.begin(), devices.end(), 0);
  // stage0.mesh = devices;
  // stage1.mesh = {0};

  // PipelineDescriptor descriptor {
  //     .stage_descriptors{std::move(stage0), std::move(stage1)}};

  // pipeline = std::make_unique<Pipeline>(std::move(fusion),
  // std::move(descriptor));

  // inputs = {at::ones(input_extents, tensor_options) *
  // (communicator->deviceId() + 1)};

  // executeAndValidate();
}

TensorView* MatrixMultiplication(TensorView* a, TensorView* b) {
  auto a_b = broadcast(a, {false, false, true}); // (x,y,b)
  auto b_b = broadcast(b, {true, false, false}); // (b,y,z)

  auto c = mul(a_b, b_b); // (x,y,z)
  auto d = sum(c, {1}); // (x,r,z)
  return d;
}

TEST_F(PipelineTest, matmul_summa) {
  // use NMK for matrix dimensions instead (by convention)
  if (communicator->deviceId()) {
    return;
  }
  // Matrices dimensions
  // a's shape=[x,y]
  // b's shape=[y,z]
  constexpr int64_t x = 12;
  constexpr int64_t y = 18;
  constexpr int64_t z = 24;
  const std::vector<int64_t> a_extents = {x, y};
  const std::vector<int64_t> b_extents = {y, z};

  // Device Mesh
  // [ 0 1 2
  //   3 4 5 ]
  constexpr int64_t N = 2;
  constexpr int64_t M = 3;
  DeviceMesh mesh({0, 1, 2, 3, 4, 5});
  mesh.reshape({N, M});

  auto fusion = std::make_unique<Fusion>();
  auto fg = std::make_unique<FusionGuard>(fusion.get());

  // a {DIDx{N}, x/N, DIDy{M}, y/M}
  // b {DIDx{N}, y/N, DIDy{M}, z/M}
  auto a = makeConcreteTensor(a_extents);
  auto b = makeConcreteTensor(b_extents);
  a->split(0, N, false);
  a->split(2, M, false);
  b->split(0, N, false);
  b->split(2, M, false);
  a->setDeviceMesh(mesh);
  b->setDeviceMesh(mesh);
  a->axis(0)->parallelize(ParallelType::DIDx);
  a->axis(2)->parallelize(ParallelType::DIDy);
  b->axis(0)->parallelize(ParallelType::DIDx);
  b->axis(2)->parallelize(ParallelType::DIDy);
  fusion->addInput(a);
  fusion->addInput(b);

  // a2 {DIDx{N}, x/N, y}
  // b2 {y, DIDy{M}, z/M}
  auto a2 = set(a);
  auto b2 = set(b);
  a2->split(0, N, false);
  b2->split(1, M, false);
  a2->setDeviceMesh(mesh);
  b2->setDeviceMesh(mesh);
  a2->axis(0)->parallelize(ParallelType::DIDx);
  b2->axis(1)->parallelize(ParallelType::DIDy);

  // a3 {DIDx{N}, x/N, y, b,  b }
  // b3 {b,  b , y, DIDy{M}, z/M}
  auto a3 = broadcast(a2, {false, false, true, true});
  auto b3 = broadcast(b2, {true, true, false, false});
  a3->split(0, N, false);
  b3->split(3, M, false);
  a3->setDeviceMesh(mesh);
  b3->setDeviceMesh(mesh);
  a3->axis(0)->parallelize(ParallelType::DIDx);
  b3->axis(3)->parallelize(ParallelType::DIDy);

  // c {DIDx{N}, x/N, y, DIDy{M}, z/M}
  auto c = mul(a3, b3);
  c->setDeviceMesh(mesh);
  c->axis(0)->parallelize(ParallelType::DIDx);
  c->axis(3)->parallelize(ParallelType::DIDy);

  // d {DIDx{N}, x/N, r{y}, DIDy{M}, z/M}
  auto d = sum(c, {2});
  d->setDeviceMesh(mesh);
  d->axis(0)->parallelize(ParallelType::DIDx);
  d->axis(3)->parallelize(ParallelType::DIDy);
  fusion->addOutput(d);

  // fusion->print();

  inputs = {
      at::randn(a_extents, tensor_options),
      at::randn(b_extents, tensor_options)};

  FusionExecutor fe;
  fe.compileFusion(fusion.get(), inputs);
  auto ref_outputs = fe.runFusion(inputs);

  // std::cout << "a (concrete inputs): \n"
  //           << inputs.at(0) << "\nb (concrete inputs): \n"
  //           << inputs.at(1) << std::endl;
  // for (auto t : c10::irange(ref_outputs.size())) {
  //   std::cout << "\noutput " << t << ":\n" << ref_outputs.at(t);
  // }
  // std::cout << std::endl;
}

// 1D tensor parallel
// Inputs: X[N,K], A^T[M, K], B[M,0]
// Compute: Y = Matmul(X, A), Z = Matmul(Y, B)
// sharding: A, B are row-wise sharded (M)
TEST_F(PipelineTest, matmuls_megatron_mlp) {
  std::unique_ptr<Fusion> fusion = std::make_unique<Fusion>();
  FusionGuard fg(fusion.get());
  int num_devices = communicator->size();
  std::vector<int64_t> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  DeviceMesh mesh(devices);

  int N = 8;
  int K = 12;
  int M = num_devices;
  int O = 5;

  TensorView* tvx = makeConcreteTensor({N, K}); // (N,K)
  TensorView* tva = makeConcreteTensor({M, K}); // (M,K)
  TensorView* tvb = makeConcreteTensor({M, O}); // (M,O)

  // Matmul 1: tvx x tva -> y (N,M)
  TensorView* tvx_b = broadcast(tvx, {true, false, false}); // (M,N,K)
  TensorView* tva_b = broadcast(tva, {false, true, false}); // (M,N,K)
  TensorView* tvxa = mul(tvx_b, tva_b); // (M,N,K)
  TensorView* tvy = sum(tvxa, {2}); // (M,N) yT

  // Matmul 2: y^T (M, N) x tvb (M, O) -> z (N,O)
  TensorView* tvy_b = broadcast(tvy, {false, false, true}); // (M,N,O)
  TensorView* tvb_b = broadcast(tvb, {false, true, false}); // (M,N,O)
  TensorView* tvyb = mul(tvy_b, tvb_b); // (M,N,O)
  TensorView* tvz = sum(tvyb, {0}); // (N,O)

  // Tensor sharded along the M axis. All other tensors are replicated.
  // TODO: propogate sharding from inputs
  auto sharded_tvs = {tva, tvx_b, tva_b, tvxa, tvy, tvb, tvy_b, tvb_b, tvyb};
  for (auto tv : sharded_tvs) {
    // tensor->split(0, num_devices, false); // split outer-dimension by D
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  fusion->addInput(tvx);
  fusion->addInput(tva);
  fusion->addInput(tvb);
  fusion->addOutput(tvy);
  fusion->addOutput(tvz);

  std::vector<TensorView*> tvs = {
      tvx,
      tva,
      tvb,
      tvx_b,
      tva_b,
      tvxa,
      tvy,
      tvy_b,
      tvb_b,
      tvyb,
      tvz,
  };
  for (auto tv : tvs) {
    tv->setDeviceMesh(mesh);
  }

  auto x = at::randn({N, K}, tensor_options);
  auto aT = at::randn({M, K}, tensor_options);
  auto b = at::randn({M, O}, tensor_options);
  auto myDevice = communicator->deviceId();
  inputs = {x, shardTensor(aT, mesh, myDevice), shardTensor(b, mesh, myDevice)};
  auto y = at::matmul(x, aT.transpose(0, 1));
  auto z = at::matmul(y, b);
  auto expected_outputs = {shardTensor(y.transpose(0, 1), mesh, myDevice), z};

  MultiDeviceExecutor runtime(std::move(fusion), *communicator);
  auto outputs = runtime.runWithInput(inputs);
  testValidate(
      runtime.completeFusion(),
      outputs,
      inputs,
      expected_outputs,
      __LINE__,
      __FILE__);
}

TEST_F(PipelineTest, matmul_megatron_attention) {
  FusionGuard fg(fusion.get());
  int num_devices = communicator->size();
  std::vector<int64_t> devices(num_devices);
  std::iota(devices.begin(), devices.end(), 0);
  DeviceMesh mesh(devices);

  int H = num_devices; // number of attention heads
  int S = 8; // sequence length
  int D_model = 5;
  int Dv = 4;
  int Dk = 4;

  TensorView* q = makeConcreteTensor({H, S, Dk});
  TensorView* k = makeConcreteTensor({H, S, Dk});
  TensorView* v = makeConcreteTensor({H, S, Dv});
  TensorView* w = makeConcreteTensor({H, Dv, D_model});
  auto tv_inputs = {q, k, v, w};
  for (auto i : tv_inputs) {
    fusion->addInput(i);
  }

  // scaled dot product attention
  TensorView* q_b = broadcast(q, {false, false, true, false}); // (H, S, S, Dk)
  TensorView* k_b = broadcast(k, {false, true, false, false}); // (H, S, S, Dk)
  TensorView* qk_ = mul(q_b, k_b); // (H, S, S, Dk)
  TensorView* qk = sum(qk_, {3}); // (H, S, S)

  TensorView* qk_b =
      broadcast(qk, {false, false, false, true}); // (H, S, S, Dv)
  TensorView* v_b = broadcast(v, {false, true, false, false}); // (H, S, S, Dv)
  TensorView* qkv_ = mul(qk_b, v_b);
  TensorView* qkv = sum(qkv_, {2}); // (H, S, Dv)

  // linear
  TensorView* qkv_b =
      broadcast(qkv, {false, false, false, true}); // (H, S, Dv, Dmodel)
  TensorView* w_b =
      broadcast(w, {false, true, false, false}); // (H, S, Dv, Dmodel)
  TensorView* y0 = mul(qkv_b, w_b); // (H, S, Dv, Dmodel)
  TensorView* y1 = sum(y0, {2}); // (H, S, Dmodel)
  TensorView* y = sum(y1, {0}); // (S, Dmodel)
  fusion->addOutput(qk);
  fusion->addOutput(qkv);
  fusion->addOutput(y);

  // All tensorviews that are sharded along the head dimension
  auto h_sharded = {
      q,
      k,
      v,
      w,
      q_b,
      k_b,
      qk_b,
      qk_,
      qk,
      qk_b,
      v_b,
      qkv_,
      qkv,
      qkv_b,
      w_b,
      y0,
      y1};
  for (auto tv : h_sharded) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
    tv->setDeviceMesh(mesh);
  }

  // Only y (output) is not sharded.
  y->setDeviceMesh(mesh);

  auto aten_q = at::randn({H, S, Dk}, tensor_options);
  auto aten_k = at::randn({H, S, Dk}, tensor_options);
  auto aten_v = at::randn({H, S, Dv}, tensor_options);
  auto aten_w = at::randn({H, Dv, D_model}, tensor_options);
  auto deviceId = communicator->deviceId();
  inputs = {
      shardTensor(aten_q, mesh, deviceId),
      shardTensor(aten_k, mesh, deviceId),
      shardTensor(aten_v, mesh, deviceId),
      shardTensor(aten_w, mesh, deviceId)};
  auto aten_qk = at::matmul(
      aten_q, aten_k.permute({0, 2, 1})); // H, S, Dk x H, Dk, S -> H, S, S
  auto aten_qkv = at::matmul(aten_qk, aten_v); // H, S, S x H, S, Dv -> H, S, Dv
  auto aten_qkv_concat = aten_qkv.permute({1, 0, 2}).flatten(1, 2); // S, H*Dv
  auto aten_w_ = aten_w.flatten(0, 1); // H*Dv, Dmodel
  auto out = at::matmul(
      aten_qkv_concat, aten_w_); // S, H*Dv x H*Dv, Dmodel -> S, Dmodel
  auto expected_outputs = {
      shardTensor(aten_qk, mesh, deviceId),
      shardTensor(aten_qkv, mesh, deviceId),
      out};

  MultiDeviceExecutor runtime(std::move(fusion), *communicator);
  auto outputs = runtime.runWithInput(inputs);
  testValidate(
      runtime.completeFusion(),
      outputs,
      inputs,
      expected_outputs,
      __LINE__,
      __FILE__);
}
} // namespace nvfuser

#endif
