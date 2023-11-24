// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#include <gtest/gtest.h>

#include <test/utils.h>
#include <test/validator.h>

#include <executor.h>
#include <fusion.h>
#include <ir/all_nodes.h>
#include <ops/all_ops.h>
#include <scheduler/mma_utils.h>

namespace nvfuser {

class TuringMmaTest : public NVFuserTest {
  void SetUp() override {
    // requires Hopper or newer
    if (cudaArchGuardShouldSkip(7, 5)) {
      GTEST_SKIP() << "skipping tests on pre-Turing GPUs";
    }
    NVFuserTest::SetUp();
  }
};

// MMA unit test on Turing
TEST_F(TuringMmaTest, TN) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [M, K]
  auto tv0 = makeConcreteTensor({16, 16}, DataType::Half);
  // [N, K]
  auto tv1 = makeConcreteTensor({8, 16}, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [M, N, K]
  auto tv0b = broadcast(tv0, {false, true, false});
  auto tv1b = broadcast(tv1, {true, false, false});

  // Leaving both sets of mma inputs for volta outside
  //  currently since they need to be swizzled.
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {2});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(16, 8, 16);
  gemm_tile.warp_tile = GemmTile(16, 8, 16);
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

  auto mma_builder =
      MmaBuilder(MmaOptions::MacroType::Turing_16_8_16, gemm_tile)
          .layout(MmaOptions::MmaLayout::TN);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_builder.configureMma(mma_ops.front());

  auto tv0cw = tv0b->cacheAfter();
  auto tv0cr = tv0cw->cacheAfter(LoadStoreOpType::LdMatrix);
  auto tv1cw = tv1b->cacheAfter();
  auto tv1cr = tv1cw->cacheAfter(LoadStoreOpType::LdMatrix);

  auto tv2c = tv2->cacheBefore();
  mma_builder.accumulatorTv(tv2c);

  // [M, N, K] -> [N, M, K]
  tv0cr->reorder({{-2, -3}, {-3, -2}});
  tv0cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv0cr);
  tv1cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv1cr, true);
  tv2c->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());
  tv2->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());

  tv0cw->setMemoryType(MemoryType::Shared);
  tv1cw->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 16}, options);
  auto t1 = at::randn({8, 16}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams);

  auto cg_outputs = fe.runFusion({t0, t1});

  auto tref = t0.to(at::kFloat).matmul(t1.t().to(at::kFloat));

  testValidate(&fusion, cg_outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

// MMA unit test on Turing
TEST_F(TuringMmaTest, TT) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [M, K]
  auto tv0 = makeConcreteTensor({16, 16}, DataType::Half);
  // [K, N]
  auto tv1 = makeConcreteTensor({16, 8}, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [M, N, K]
  auto tv0b = broadcast(tv0, {false, true, false});
  // [M, K, N]
  auto tv1b = broadcast(tv1, {true, false, false});
  // [M, N, K]
  auto tv1t = transpose(tv1b, 1, 2);

  auto tv2 = fusedMultiplySum(tv0b, tv1t, {2});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(16, 8, 16);
  gemm_tile.warp_tile = GemmTile(16, 8, 16);
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

  auto mma_builder =
      MmaBuilder(MmaOptions::MacroType::Turing_16_8_16, gemm_tile)
          .layout(MmaOptions::MmaLayout::TT);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_builder.configureMma(mma_ops.front());

  auto tv0cw = tv0b->cacheAfter();
  auto tv0cr = tv0cw->cacheAfter(LoadStoreOpType::LdMatrix);
  auto tv1cw = tv1b->cacheAfter();
  auto tv1cr = tv1t;
  tv1cr->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::LdMatrixTranspose);

  auto tv2c = tv2->cacheBefore();
  mma_builder.accumulatorTv(tv2c);

  // [M, N, K] -> [N, M, K]
  tv0cr->reorder({{-2, -3}, {-3, -2}});
  tv0cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv0cr);
  tv1cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv1cr, true);
  tv2c->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());
  tv2->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());

  tv0cw->setMemoryType(MemoryType::Shared);
  tv1cw->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 16}, options);
  auto t1 = at::randn({16, 8}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams);

  auto cg_outputs = fe.runFusion({t0, t1});

  auto tref = t0.to(at::kFloat).matmul(t1.to(at::kFloat));

  testValidate(&fusion, cg_outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

// MMA unit test on Turing
TEST_F(TuringMmaTest, NT) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [K, M]
  auto tv0 = makeConcreteTensor({16, 16}, DataType::Half);
  // [K, N]
  auto tv1 = makeConcreteTensor({16, 8}, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [K, M, N]
  auto tv0b = broadcast(tv0, {false, false, true});
  auto tv1b = broadcast(tv1, {false, true, false});

  // [M, N, K]
  auto tv0t = permute(tv0b, {1, 2, 0});
  auto tv1t = permute(tv1b, {1, 2, 0});
  auto tv2 = fusedMultiplySum(tv0t, tv1t, {2});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(16, 8, 16);
  gemm_tile.warp_tile = GemmTile(16, 8, 16);
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

  auto mma_builder =
      MmaBuilder(MmaOptions::MacroType::Turing_16_8_16, gemm_tile)
          .layout(MmaOptions::MmaLayout::NT);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_builder.configureMma(mma_ops.front());

  auto tv0cw = tv0b->cacheAfter();
  auto tv0cr = tv0t;
  tv0cr->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::LdMatrixTranspose);
  auto tv1cw = tv1b->cacheAfter();
  auto tv1cr = tv1t;
  tv1cr->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::LdMatrixTranspose);

  auto tv2c = tv2->cacheBefore();
  mma_builder.accumulatorTv(tv2c);

  // [K,M,N] -> [N,M,K]
  tv0cr->reorder({{-2, -3}, {-3, -2}});
  tv0cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv0cr);
  tv1cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv1cr, true);
  tv2c->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());
  tv2->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());

  tv0cw->setMemoryType(MemoryType::Shared);
  tv1cw->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 16}, options);
  auto t1 = at::randn({16, 8}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams);

  auto cg_outputs = fe.runFusion({t0, t1});

  auto tref = t0.t().to(at::kFloat).matmul(t1.to(at::kFloat));

  testValidate(&fusion, cg_outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

// MMA unit test on Ampere
TEST_F(TuringMmaTest, NN) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [K, M]
  auto tv0 = makeConcreteTensor({16, 16}, DataType::Half);
  // [N, K]
  auto tv1 = makeConcreteTensor({8, 16}, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [K, M, N]
  auto tv0b = broadcast(tv0, {false, false, true});
  // [M, N, K]
  auto tv1b = broadcast(tv1, {true, false, false});

  // [M, N, K]
  auto tv0t = permute(tv0b, {1, 2, 0});
  auto tv2 = fusedMultiplySum(tv0t, tv1b, {2});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(16, 8, 16);
  gemm_tile.warp_tile = GemmTile(16, 8, 16);
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

  auto mma_builder =
      MmaBuilder(MmaOptions::MacroType::Turing_16_8_16, gemm_tile)
          .layout(MmaOptions::MmaLayout::NN);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_builder.configureMma(mma_ops.front());

  auto tv0cw = tv0b->cacheAfter();
  auto tv0cr = tv0t;
  tv0cr->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::LdMatrixTranspose);
  auto tv1cw = tv1b->cacheAfter();
  auto tv1cr = tv1cw->cacheAfter(LoadStoreOpType::LdMatrix);

  auto tv2c = tv2->cacheBefore();
  mma_builder.accumulatorTv(tv2c);

  // [M, N, K] -> [N, M, K]
  tv0cr->reorder({{-2, -3}, {-3, -2}});
  tv0cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv0cr);
  tv1cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv1cr, true);
  tv2c->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());
  tv2->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());

  tv0cw->setMemoryType(MemoryType::Shared);
  tv1cw->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 16}, options);
  auto t1 = at::randn({8, 16}, options);

  FusionExecutor fe;
  fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams);
  auto cg_outputs = fe.runFusion({t0, t1});

  auto tref = t0.t().to(at::kFloat).matmul(t1.t().to(at::kFloat));

  testValidate(&fusion, cg_outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

class AmpereMmaTest : public NVFuserTest {
  void SetUp() override {
    // requires Hopper or newer
    if (!deviceMajorMinorCheck(8)) {
      GTEST_SKIP() << "skipping tests on pre-Ampere GPUs";
    }
    NVFuserTest::SetUp();
  }
};

// MMA unit test on Ampere
TEST_F(AmpereMmaTest, TN) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [M, K]
  auto tv0 = makeConcreteTensor({16, 16}, DataType::Half);
  // [N, K]
  auto tv1 = makeConcreteTensor({8, 16}, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [M, N, K]
  auto tv0b = broadcast(tv0, {false, true, false});
  auto tv1b = broadcast(tv1, {true, false, false});

  // Leaving both sets of mma inputs for volta outside
  //  currently since they need to be swizzled.
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {2});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(16, 8, 16);
  gemm_tile.warp_tile = GemmTile(16, 8, 16);
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

  auto mma_builder =
      MmaBuilder(MmaOptions::MacroType::Ampere_16_8_16, gemm_tile)
          .layout(MmaOptions::MmaLayout::TN);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_builder.configureMma(mma_ops.front());

  auto tv0cw = tv0b->cacheAfter();
  auto tv0cr = tv0cw->cacheAfter(LoadStoreOpType::LdMatrix);
  auto tv1cw = tv1b->cacheAfter();
  auto tv1cr = tv1cw->cacheAfter(LoadStoreOpType::LdMatrix);

  auto tv2c = tv2->cacheBefore();
  mma_builder.accumulatorTv(tv2c);

  // [M, N, K] -> [N, M, K]
  tv0cr->reorder({{-2, -3}, {-3, -2}});
  tv0cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());
  tv1cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv0cr);
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv1cr, true);
  tv2c->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());
  tv2->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());

  tv0cw->setMemoryType(MemoryType::Shared);
  tv1cw->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 16}, options);
  auto t1 = at::randn({8, 16}, options);

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams));
  auto cg_outputs = fe.runFusion({t0, t1});

  auto tref = t0.to(at::kFloat).matmul(t1.t().to(at::kFloat));

  testValidate(&fusion, cg_outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

// MMA unit test on Ampere
TEST_F(AmpereMmaTest, TT) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [M, K]
  auto tv0 = makeConcreteTensor({16, 16}, DataType::Half);
  // [K, N]
  auto tv1 = makeConcreteTensor({16, 8}, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [M, N, K]
  auto tv0b = broadcast(tv0, {false, true, false});
  // [M, K, N]
  auto tv1b = broadcast(tv1, {true, false, false});
  // [M, N, K]
  auto tv1t = transpose(tv1b, 1, 2);

  auto tv2 = fusedMultiplySum(tv0b, tv1t, {2});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(16, 8, 16);
  gemm_tile.warp_tile = GemmTile(16, 8, 16);
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

  auto mma_builder =
      MmaBuilder(MmaOptions::MacroType::Ampere_16_8_16, gemm_tile)
          .layout(MmaOptions::MmaLayout::TT);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_builder.configureMma(mma_ops.front());

  auto tv0cw = tv0b->cacheAfter();
  auto tv0cr = tv0cw->cacheAfter(LoadStoreOpType::LdMatrix);
  auto tv1cw = tv1b->cacheAfter();
  auto tv1cr = tv1t;
  tv1cr->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::LdMatrixTranspose);

  auto tv2c = tv2->cacheBefore();
  mma_builder.accumulatorTv(tv2c);

  // [M, N, K] -> [N, M, K]
  tv0cr->reorder({{-2, -3}, {-3, -2}});
  tv0cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());
  tv1cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv0cr);
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv1cr, true);
  tv2c->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());
  tv2->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());

  tv0cw->setMemoryType(MemoryType::Shared);
  tv1cw->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 16}, options);
  auto t1 = at::randn({16, 8}, options);

  FusionExecutor fe;

  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams));

  auto cg_outputs = fe.runFusion({t0, t1});

  auto tref = t0.to(at::kFloat).matmul(t1.to(at::kFloat));

  testValidate(&fusion, cg_outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

// MMA unit test on Ampere
TEST_F(AmpereMmaTest, NT) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [K, M]
  auto tv0 = makeConcreteTensor({16, 16}, DataType::Half);
  // [K, N]
  auto tv1 = makeConcreteTensor({16, 8}, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [K, M, N]
  auto tv0b = broadcast(tv0, {false, false, true});
  auto tv1b = broadcast(tv1, {false, true, false});

  // [M, N, K]
  auto tv0t = permute(tv0b, {1, 2, 0});
  auto tv1t = permute(tv1b, {1, 2, 0});
  auto tv2 = fusedMultiplySum(tv0t, tv1t, {2});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(16, 8, 16);
  gemm_tile.warp_tile = GemmTile(16, 8, 16);
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

  auto mma_builder =
      MmaBuilder(MmaOptions::MacroType::Ampere_16_8_16, gemm_tile)
          .layout(MmaOptions::MmaLayout::NT);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_builder.configureMma(mma_ops.front());

  auto tv0cw = tv0b->cacheAfter();
  auto tv0cr = tv0t;
  tv0cr->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::LdMatrixTranspose);
  auto tv1cw = tv1b->cacheAfter();
  auto tv1cr = tv1t;
  tv1cr->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::LdMatrixTranspose);

  auto tv2c = tv2->cacheBefore();
  mma_builder.accumulatorTv(tv2c);

  // [M, N, K] -> [N, M, K]
  tv0cr->reorder({{-2, -3}, {-3, -2}});
  tv0cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());
  tv1cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv0cr);
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv1cr, true);
  tv2c->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());
  tv2->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());

  tv0cw->setMemoryType(MemoryType::Shared);
  tv1cw->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 16}, options);
  auto t1 = at::randn({16, 8}, options);

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams));
  auto cg_outputs = fe.runFusion({t0, t1});

  auto tref = t0.t().to(at::kFloat).matmul(t1.to(at::kFloat));

  testValidate(&fusion, cg_outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

// MMA unit test on Ampere
TEST_F(AmpereMmaTest, NN) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [K, M]
  auto tv0 = makeConcreteTensor({16, 16}, DataType::Half);
  // [N, K]
  auto tv1 = makeConcreteTensor({8, 16}, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [K, M, N]
  auto tv0b = broadcast(tv0, {false, false, true});
  // [M, N, K]
  auto tv1b = broadcast(tv1, {true, false, false});

  // [M, N, K]
  auto tv0t = permute(tv0b, {1, 2, 0});
  auto tv2 = fusedMultiplySum(tv0t, tv1b, {2});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(16, 8, 16);
  gemm_tile.warp_tile = GemmTile(16, 8, 16);
  gemm_tile.instruction_tile = GemmTile(16, 8, 16);

  auto mma_builder =
      MmaBuilder(MmaOptions::MacroType::Ampere_16_8_16, gemm_tile)
          .layout(MmaOptions::MmaLayout::NN);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_builder.configureMma(mma_ops.front());

  auto tv0cw = tv0b->cacheAfter();
  auto tv0cr = tv0t;
  tv0cr->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::LdMatrixTranspose);
  auto tv1cw = tv1b->cacheAfter();
  auto tv1cr = tv1cw->cacheAfter(LoadStoreOpType::LdMatrix);

  auto tv2c = tv2->cacheBefore();
  mma_builder.accumulatorTv(tv2c);

  // [M, N, K] -> [N, M, K]
  tv0cr->reorder({{-2, -3}, {-3, -2}});
  tv0cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());
  tv1cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv0cr);
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv1cr, true);
  tv2c->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());
  tv2->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());

  tv0cw->setMemoryType(MemoryType::Shared);
  tv1cw->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 16}, options);
  auto t1 = at::randn({8, 16}, options);

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams));
  auto cg_outputs = fe.runFusion({t0, t1});

  auto tref = t0.t().to(at::kFloat).matmul(t1.t().to(at::kFloat));

  testValidate(&fusion, cg_outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

// MMA unit test on Ampere
TEST_F(AmpereMmaTest, LargeTN) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [M, K]
  auto tv0 = makeConcreteTensor({16, 16}, DataType::Half);
  // [N, K]
  auto tv1 = makeConcreteTensor({16, 16}, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [M, N, K]
  auto tv0b = broadcast(tv0, {false, true, false});
  auto tv1b = broadcast(tv1, {true, false, false});

  // Leaving both sets of mma inputs for volta outside
  //  currently since they need to be swizzled.
  auto tv2 = fusedMultiplySum(tv0b, tv1b, {2});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(16, 16, 16);
  gemm_tile.warp_tile = GemmTile(16, 16, 16);
  gemm_tile.instruction_tile = GemmTile(16, 16, 16);

  auto mma_builder =
      MmaBuilder(MmaOptions::MacroType::Ampere_16_16_16, gemm_tile)
          .layout(MmaOptions::MmaLayout::TN);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_builder.configureMma(mma_ops.front());

  auto tv0cw = tv0b->cacheAfter();
  auto tv0cr = tv0cw->cacheAfter(LoadStoreOpType::LdMatrix);
  auto tv1cw = tv1b->cacheAfter();
  auto tv1cr = tv1cw->cacheAfter(LoadStoreOpType::LdMatrix);

  auto tv2c = tv2->cacheBefore();
  mma_builder.accumulatorTv(tv2c);

  // [M, N, K] -> [N, M, K]
  tv0cr->reorder({{-2, -3}, {-3, -2}});
  tv0cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());
  tv1cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv0cr);
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv1cr, true);
  tv2c->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());
  tv2->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());

  tv0cw->setMemoryType(MemoryType::Shared);
  tv1cw->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 16}, options);
  auto t1 = at::randn({16, 16}, options);

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams));
  auto cg_outputs = fe.runFusion({t0, t1});

  auto tref = t0.to(at::kFloat).matmul(t1.t().to(at::kFloat));

  testValidate(&fusion, cg_outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

// MMA unit test on Ampere
TEST_F(AmpereMmaTest, LargeTT) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [M, K]
  auto tv0 = makeConcreteTensor({16, 16}, DataType::Half);
  // [K, N]
  auto tv1 = makeConcreteTensor({16, 16}, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [M, N, K]
  auto tv0b = broadcast(tv0, {false, true, false});
  // [M, K, N]
  auto tv1b = broadcast(tv1, {true, false, false});
  // [M, N, K]
  auto tv1t = transpose(tv1b, 1, 2);

  auto tv2 = fusedMultiplySum(tv0b, tv1t, {2});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(16, 16, 16);
  gemm_tile.warp_tile = GemmTile(16, 16, 16);
  gemm_tile.instruction_tile = GemmTile(16, 16, 16);

  auto mma_builder =
      MmaBuilder(MmaOptions::MacroType::Ampere_16_16_16, gemm_tile)
          .layout(MmaOptions::MmaLayout::TT);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_builder.configureMma(mma_ops.front());

  auto tv0cw = tv0b->cacheAfter();
  auto tv0cr = tv0cw->cacheAfter(LoadStoreOpType::LdMatrix);
  auto tv1cw = tv1b->cacheAfter();
  auto tv1cr = tv1t;
  tv1cr->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::LdMatrixTranspose);

  auto tv2c = tv2->cacheBefore();
  mma_builder.accumulatorTv(tv2c);

  // [M, N, K] -> [N, M, K]
  tv0cr->reorder({{-2, -3}, {-3, -2}});
  tv0cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());
  tv1cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv0cr);
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv1cr, true);
  tv2c->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());
  tv2->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());

  tv0cw->setMemoryType(MemoryType::Shared);
  tv1cw->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 16}, options);
  auto t1 = at::randn({16, 16}, options);

  FusionExecutor fe;

  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams));

  auto cg_outputs = fe.runFusion({t0, t1});

  auto tref = t0.to(at::kFloat).matmul(t1.to(at::kFloat));

  testValidate(&fusion, cg_outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

// MMA unit test on Ampere
TEST_F(AmpereMmaTest, LargeNT) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [K, M]
  auto tv0 = makeConcreteTensor({16, 16}, DataType::Half);
  // [K, N]
  auto tv1 = makeConcreteTensor({16, 16}, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [K, M, N]
  auto tv0b = broadcast(tv0, {false, false, true});
  auto tv1b = broadcast(tv1, {false, true, false});

  // [M, N, K]
  auto tv0t = permute(tv0b, {1, 2, 0});
  auto tv1t = permute(tv1b, {1, 2, 0});
  auto tv2 = fusedMultiplySum(tv0t, tv1t, {2});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(16, 16, 16);
  gemm_tile.warp_tile = GemmTile(16, 16, 16);
  gemm_tile.instruction_tile = GemmTile(16, 16, 16);

  auto mma_builder =
      MmaBuilder(MmaOptions::MacroType::Ampere_16_16_16, gemm_tile)
          .layout(MmaOptions::MmaLayout::NT);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_builder.configureMma(mma_ops.front());

  auto tv0cw = tv0b->cacheAfter();
  auto tv0cr = tv0t;
  tv0cr->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::LdMatrixTranspose);
  auto tv1cw = tv1b->cacheAfter();
  auto tv1cr = tv1t;
  tv1cr->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::LdMatrixTranspose);

  auto tv2c = tv2->cacheBefore();
  mma_builder.accumulatorTv(tv2c);

  // [M, N, K] -> [N, M, K]
  tv0cr->reorder({{-2, -3}, {-3, -2}});
  tv0cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());
  tv1cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv0cr);
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv1cr, true);
  tv2c->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());
  tv2->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());

  tv0cw->setMemoryType(MemoryType::Shared);
  tv1cw->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 16}, options);
  auto t1 = at::randn({16, 16}, options);

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams));
  auto cg_outputs = fe.runFusion({t0, t1});

  auto tref = t0.t().to(at::kFloat).matmul(t1.to(at::kFloat));

  testValidate(&fusion, cg_outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

// MMA unit test on Ampere
TEST_F(AmpereMmaTest, LargeNN) {
  Fusion fusion;
  FusionGuard fg(&fusion);

  // [K, M]
  auto tv0 = makeConcreteTensor({16, 16}, DataType::Half);
  // [N, K]
  auto tv1 = makeConcreteTensor({16, 16}, DataType::Half);
  fusion.addInput(tv0);
  fusion.addInput(tv1);

  // [K, M, N]
  auto tv0b = broadcast(tv0, {false, false, true});
  // [M, N, K]
  auto tv1b = broadcast(tv1, {true, false, false});

  // [M, N, K]
  auto tv0t = permute(tv0b, {1, 2, 0});
  auto tv2 = fusedMultiplySum(tv0t, tv1b, {2});

  fusion.addOutput(tv2);

  MatMulTileOptions gemm_tile;
  gemm_tile.cta_tile = GemmTile(16, 16, 16);
  gemm_tile.warp_tile = GemmTile(16, 16, 16);
  gemm_tile.instruction_tile = GemmTile(16, 16, 16);

  auto mma_builder =
      MmaBuilder(MmaOptions::MacroType::Ampere_16_16_16, gemm_tile)
          .layout(MmaOptions::MmaLayout::NN);

  auto mma_ops = ir_utils::getOpsOfType<MmaOp>(&fusion);
  NVF_CHECK(
      1 == mma_ops.size(),
      "Invalid number of MmaOp instances in fusion definition, expected 1, got ",
      mma_ops.size());
  mma_builder.configureMma(mma_ops.front());

  auto tv0cw = tv0b->cacheAfter();
  auto tv0cr = tv0t;
  tv0cr->definition()->as<LoadStoreOp>()->setOpType(
      LoadStoreOpType::LdMatrixTranspose);
  auto tv1cw = tv1b->cacheAfter();
  auto tv1cr = tv1cw->cacheAfter(LoadStoreOpType::LdMatrix);

  auto tv2c = tv2->cacheBefore();
  mma_builder.accumulatorTv(tv2c);

  // [M, N, K] -> [N, M, K]
  tv0cr->reorder({{-2, -3}, {-3, -2}});
  tv0cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::A).build());
  tv1cr->applyMmaSwizzle(mma_builder.operand(MmaOptions::Operand::B).build());
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv0cr);
  mma_utils::WarpMmaSwizzler::scheduleLdMatrix(tv1cr, true);
  tv2c->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());
  tv2->applyMmaSwizzle(
      mma_builder.operand(MmaOptions::Operand::Accumulator).build());

  tv0cw->setMemoryType(MemoryType::Shared);
  tv1cw->setMemoryType(MemoryType::Shared);

  auto options = at::TensorOptions().dtype(at::kHalf).device(at::kCUDA, 0);
  auto t0 = at::randn({16, 16}, options);
  auto t1 = at::randn({16, 16}, options);

  FusionExecutor fe;
  NVFUSER_TEST_CUDA_ARCH_COMPILE_CHECK(
      8,
      0,
      fe.compileFusion(&fusion, {t0, t1}, LaunchParams(), matmul_cparams));
  auto cg_outputs = fe.runFusion({t0, t1});

  auto tref = t0.t().to(at::kFloat).matmul(t1.t().to(at::kFloat));

  testValidate(&fusion, cg_outputs, {t0, t1}, {tref}, __LINE__, __FILE__);
}

} // namespace nvfuser
