// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <fusion.h>
#include <fusion_executor/executor_kernel_arg.h>
#include <fusion_segmenter.h>
#include <ir/all_nodes.h>
#include <ir/builder.h>
#include <multidevice/device_mesh.h>
#include <multidevice/lower_communication.h>
#include <multidevice/utils.h>
#include <ops/all_ops.h>
#include <preseg_passes/insert_reshardings.h>
#include <preseg_passes/reorder_sharded_axis.h>
#include <tests/cpp/utils.h>

#include <algorithm>
#include <iostream>

namespace nvfuser {

using ReshardingTestParams =
    std::tuple<DeviceMesh, DeviceMesh, DeviceMesh, bool, bool, bool>;

class ReshardingTest : public NVFuserFixtureParamTest<ReshardingTestParams> {
 protected:
  void SetUp() override {
    fusion_ = std::make_unique<Fusion>();
    fg_ = std::make_unique<FusionGuard>(fusion_.get());
  }
  void validate() {
    // TODO(wujingyue): after preseg passes are integrated to
    // FusionExecutorCache, simplify validation by using
    // FusionExecutorCache::getMostRecentKernelRuntime()->fusionSegments()->groups().
    for (auto expr : fusion_->exprs()) {
      EXPECT_TRUE(!isResharding(expr) || isLowerableToCommunication(expr))
          << "on expr=" << expr;
    }

    SegmentCandidateFinderOptions options{
        .run_translate_welford = false,
        .run_combine_reductions = false,
        .run_herrmann_merge = true,
        .run_final_merge = true,
        .only_segment_resharding_exprs = true};

    auto segmented_fusion =
        SegmentCandidateFinder::segment(std::move(fusion_), nullptr, options);

    for (SegmentedGroup* group : segmented_fusion->groups()) {
      // TODO: use EXPECT_THAT.
      EXPECT_TRUE(
          std::none_of(
              group->exprs().begin(),
              group->exprs().end(),
              [](auto expr) { return isResharding(expr); }) ||
          (group->exprs().size() == 1 && isResharding(group->exprs().at(0))));
    }
    // checks that the segments are disjoints and that the graph of segment is
    // acyclic
    segmented_fusion->validate();
  }

  std::unique_ptr<Fusion> fusion_;
  std::unique_ptr<FusionGuard> fg_;
};

TEST_F(ReshardingTest, Set_SameMesh_NoParallelTypes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(3);
  in->setDeviceMesh({0, 1});
  TensorView* out = set(in);

  EXPECT_FALSE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Set_DifferentMeshes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(3);
  TensorView* out = set(in);
  in->setDeviceMesh({0, 1});
  out->setDeviceMesh({0, 2});

  EXPECT_TRUE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Set_DifferentParallelTypes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(3);
  in->setDeviceMesh({0, 1, 2});
  TensorView* out = set(in);
  out->axis(0)->parallelize(ParallelType::DIDx);

  EXPECT_TRUE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Set_SameMesh_SameParallelType) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(3);
  in->setDeviceMesh({0, 1, 2});
  in->axis(0)->parallelize(ParallelType::DIDx);
  TensorView* out = set(in);

  EXPECT_FALSE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Sum_SameMesh_NoParallelTypes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(3);
  in->setDeviceMesh({0, 1, 2});
  TensorView* out = sum(in, {0});

  EXPECT_FALSE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Sum_DifferentParallelTypes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(3);
  in->setDeviceMesh({0, 1, 2});
  TensorView* out = sum(in, {0});
  out->axis(0)->parallelize(ParallelType::DIDx);

  EXPECT_TRUE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Sum_DifferentMeshes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(3);
  TensorView* out = sum(in, {0});

  in->setDeviceMesh({0, 1});
  out->setDeviceMesh({0, 1, 2});

  EXPECT_TRUE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Sum_ParallelizeDifferentAxes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(3);
  in->setDeviceMesh({0, 1, 2});
  in->axis(0)->parallelize(ParallelType::DIDx);
  TensorView* out = sum(in, {0});
  out->axis(1)->parallelize(ParallelType::DIDx);

  EXPECT_TRUE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Sum_ParallelizeSameAxis) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(3);
  in->setDeviceMesh({0, 1, 2});
  in->axis(0)->parallelize(ParallelType::DIDx);
  TensorView* out = sum(in, {1});

  EXPECT_FALSE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Sum_AllReduce) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* in = makeContigTensor(3);
  in->setDeviceMesh({0, 1, 2});
  in->axis(0)->parallelize(ParallelType::DIDx);
  TensorView* out = sum(in, {0});

  EXPECT_TRUE(isResharding(out->definition()));
}

TEST_F(ReshardingTest, Add_SameMesh_NoParallelTypes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeContigTensor(3);
  TensorView* y = makeContigTensor(3);
  TensorView* z = add(x, y);

  for (auto* tv : {x, y, z}) {
    tv->setDeviceMesh({0, 1});
  }

  EXPECT_FALSE(isResharding(z->definition()));
}

TEST_F(ReshardingTest, Add_DifferentMeshes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeContigTensor(3);
  TensorView* y = makeContigTensor(3);
  TensorView* z = add(x, y);

  for (auto* tv : {x, y}) {
    tv->setDeviceMesh({0, 1});
  }
  z->setDeviceMesh({0, 1, 2});

  EXPECT_TRUE(isResharding(z->definition()));
}

TEST_F(ReshardingTest, Add_OnlyOutputParallelized) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeContigTensor(3);
  TensorView* y = makeContigTensor(3);
  TensorView* z = add(x, y);

  for (auto* tv : {x, y, z}) {
    tv->setDeviceMesh({0, 1});
  }
  z->axis(0)->parallelize(ParallelType::DIDx);

  EXPECT_TRUE(isResharding(z->definition()));
}

TEST_F(ReshardingTest, Add_OnlyInputsParallelized) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeContigTensor(3);
  TensorView* y = makeContigTensor(3);
  TensorView* z = add(x, y);

  for (auto* tv : {x, y, z}) {
    tv->setDeviceMesh({0, 1});
  }
  for (auto* tv : {x, y}) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  EXPECT_TRUE(isResharding(z->definition()));
}

TEST_F(ReshardingTest, Add_SameMesh_SameParallelTypes) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeContigTensor(3);
  TensorView* y = makeContigTensor(3);
  TensorView* z = add(x, y);

  for (auto* tv : {x, y, z}) {
    tv->setDeviceMesh({0, 1});
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  EXPECT_FALSE(isResharding(z->definition()));
}

TEST_F(ReshardingTest, Add_InputsParallelizedDifferently) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* x = makeContigTensor(3);
  TensorView* y = makeContigTensor(3);
  TensorView* z = add(x, y);

  for (auto* tv : {x, y, z}) {
    tv->setDeviceMesh({0, 1, 2});
  }
  for (auto* tv : {x, z}) {
    tv->axis(0)->parallelize(ParallelType::DIDx);
  }

  EXPECT_TRUE(isResharding(z->definition()));
}

TEST_F(ReshardingTest, InsertResharding_Before) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* a = makeSymbolicTensor(3);
  TensorView* b = makeSymbolicTensor(3);
  TensorView* c = add(a, b);
  fusion.addInput(a);
  fusion.addInput(b);
  fusion.addOutput(c);

  DeviceMesh mesh0({0, 1});
  DeviceMesh mesh1({2});
  a->setDeviceMesh(mesh0);
  b->setDeviceMesh(mesh0);
  c->setDeviceMesh(mesh1);

  a->axis(0)->parallelize(ParallelType::DIDx);
  c->axis(1)->parallelize(ParallelType::DIDx);

  preseg_passes::OptimizationPass<
      preseg_passes::InsertReshardingsPass>::runPass(&fusion);
  std::vector<Val*> outputs = fusion.outputs();

  c = outputs[0]->as<TensorView>();
  std::vector<TensorView*> inputs(c->definition()->inputs().size());
  for (auto i : c10::irange(c->definition()->inputs().size())) {
    inputs[i] = c->definition()->input(i)->as<TensorView>();
  }
  EXPECT_TRUE(getTvsWithDifferentSharding(c, inputs).empty());
}

TEST_F(ReshardingTest, InsertResharding_After) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* a = makeSymbolicTensor(3);
  TensorView* b = relu(a);
  fusion.addInput(a);
  fusion.addOutput(b);

  DeviceMesh mesh0({0, 1});
  DeviceMesh mesh1({2});
  a->setDeviceMesh(mesh0);
  b->setDeviceMesh(mesh1);

  a->axis(0)->parallelize(ParallelType::DIDx);
  b->axis(1)->parallelize(ParallelType::DIDx);

  preseg_passes::OptimizationPass<
      preseg_passes::InsertReshardingsPass>::runPass(&fusion);
  std::vector<Val*> outputs = fusion.outputs();

  b = outputs[0]->as<TensorView>();
  Expr* expr = b->definition();
  EXPECT_TRUE(expr->isA<LoadStoreOp>());
  EXPECT_EQ(expr->as<LoadStoreOp>()->opType(), LoadStoreOpType::Set);
  std::vector<TensorView*> tvs = {expr->inputs()[0]->as<TensorView>()};
  EXPECT_THAT(getTvsWithDifferentSharding(a, tvs), ::testing::IsEmpty());
}

TEST_F(ReshardingTest, InsertShardedAxisReordering) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  TensorView* a = makeSymbolicTensor(3);
  TensorView* b = relu(a);
  TensorView* c = add(a, b);
  fusion.addInput(a);
  fusion.addOutput(c);

  DeviceMesh mesh({0, 1});
  a->setDeviceMesh(mesh);
  b->setDeviceMesh(mesh);
  c->setDeviceMesh(mesh);

  b->axis(1)->parallelize(ParallelType::DIDx);
  c->axis(1)->parallelize(ParallelType::DIDx);

  preseg_passes::OptimizationPass<
      preseg_passes::InsertReshardingsPass>::runPass(&fusion);
  int num_inner_reshardings = 0;
  for (auto expr : fusion.exprs()) {
    if (isResharding(expr) && isInnerResharding(expr)) {
      num_inner_reshardings++;
    }
  }
  EXPECT_GT(num_inner_reshardings, 0);

  preseg_passes::OptimizationPass<
      preseg_passes::ReorderShardedAxisPass>::runPass(&fusion);
  for (auto expr : fusion.exprs()) {
    if (isResharding(expr)) {
      EXPECT_FALSE(isInnerResharding(expr));
    }
  }
}

TEST_P(ReshardingTest, Insert) {
  if (!distributedEnabled()) { // Test only works with distributed
    GTEST_SKIP() << "Requires distributed API";
  }
  auto
      [mesh0,
       mesh1,
       mesh2,
       is_tv0_tv3_tv5_sharded,
       is_tv1_tv4_sharded,
       is_tv2_sharded] = GetParam();
  int sharded_axis = 1;

  TensorView* tv0 = makeContigTensor(3);
  TensorView* tv1 = binaryOp(BinaryOpType::Mul, tv0, tv0);
  TensorView* tv2 = binaryOp(BinaryOpType::Add, tv0, tv1);
  TensorView* tv3 = sum(tv2, {1});
  TensorView* tv4 = broadcast(tv3, {false, true, false});
  TensorView* tv5 = binaryOp(BinaryOpType::Mul, tv2, tv4);

  tv0->setDeviceMesh(mesh0);
  tv1->setDeviceMesh(mesh1);
  tv2->setDeviceMesh(mesh2);
  tv3->setDeviceMesh(mesh0);
  tv4->setDeviceMesh(mesh1);
  tv5->setDeviceMesh(mesh0);
  fusion_->addInput(tv0);
  fusion_->addOutput(tv1);
  fusion_->addOutput(tv5);

  if (is_tv0_tv3_tv5_sharded) {
    tv0->axis(sharded_axis)->parallelize(ParallelType::DIDx);
    tv3->axis(sharded_axis)->parallelize(ParallelType::DIDx);
    tv5->axis(sharded_axis)->parallelize(ParallelType::DIDx);
  }
  if (is_tv1_tv4_sharded) {
    tv1->axis(sharded_axis)->parallelize(ParallelType::DIDx);
    tv4->axis(sharded_axis)->parallelize(ParallelType::DIDx);
  }
  if (is_tv2_sharded) {
    tv2->axis(sharded_axis)->parallelize(ParallelType::DIDx);
  }

  preseg_passes::OptimizationPass<
      preseg_passes::InsertReshardingsPass>::runPass(fusion_.get());
  preseg_passes::OptimizationPass<
      preseg_passes::ReorderShardedAxisPass>::runPass(fusion_.get());
  validate();
}

namespace {

DeviceMesh Mesh0({0});
DeviceMesh Mesh1({1, 2});
DeviceMesh Mesh2({0, 1, 2, 3});

} // namespace

INSTANTIATE_TEST_SUITE_P(
    ,
    ReshardingTest,
    ::testing::Combine(
        ::testing::Values(Mesh0, Mesh2),
        ::testing::Values(Mesh1, Mesh2),
        ::testing::Values(Mesh2),
        ::testing::Bool(),
        ::testing::Bool(),
        ::testing::Bool()));

} // namespace nvfuser
