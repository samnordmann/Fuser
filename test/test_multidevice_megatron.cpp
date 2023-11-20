// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <gtest/gtest.h>

// #include <fusion.h>
#include <ir/all_nodes.h>
#include <multidevice/communication.h>
#include <multidevice/communicator.h>
#include <multidevice/device_mesh.h>
#include <multidevice/pipeline.h>
#include <multidevice/pipeline_ir.h>
#include <multidevice/runtime.h>
#include <test/multidevice.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <ops/all_ops.h>


#include <iostream>

namespace nvfuser {

// M,K x K,N -> M,N
TensorView* Matmul(TensorView* tv0, TensorView* tv1, PipelineStageDescriptor& stage) {
  TensorView* tv2 = broadcast(tv0, {false, false, true});
  TensorView* tv3 = broadcast(tv1, {true, false, false});
  TensorView* tv4 = mul(tv2, tv3);
  TensorView* tv5 = sum(tv4, {1});
  stage.addVal({tv2, tv3, tv4, tv5});
  return tv5;
}

// TODO: batch parallelism
// Inputs: X[S,E], A[E,N], B[N,M]
// Compute: Y = GeLU(Matmul(X, A)), Z = Matmul(Y, B)
// matmul1: replicated x column-wise sharded -> columnwise sharded
// matmul2: column-wise sharded x row-wise sharded -> replicated
TEST_F(PipelineTest, MegatronMLP) {
  FusionGuard fg(fusion.get());
  PipelineStageDescriptor stage0;
  int num_devices = 4;
  std::vector<DeviceIdxType> device_mesh({0, 1, 2, 3});

  stage0.mesh = device_mesh;
  TensorView* tvx = makeContigTensor(2); // (N,K)
  TensorView* tva = makeContigTensor(2); // (K,M)
  TensorView* tvb = makeContigTensor(2); // (M,O)
  stage0.addVal({tvx, tva, tvb});

  // TensorView* tvy = Matmul(tvx, tva, stage0); //(N,M)
  // TensorView* tvz = Matmul(tvy, tvb, stage0); //(N,O)

  // Matmul 1: tvx x tva -> y (N,M)
  TensorView* tvx_b = broadcast(tvx, {false, false, true}); // (N,K,M)
  tvx_b->split(0, num_devices, false); // (D,N//D,K,M)
  TensorView* tva_b = broadcast(tva, {true, false, false}); // (N,K,M)
  tva_b->split(0, num_devices, false); // (D,N//D,K,M)
  TensorView* tvxa = mul(tvx_b, tva_b); // (N,K,M)
  tvxa->split(0, num_devices, false); // (D,N//D,K,M)
  TensorView* tvy = sum(tv_xa, {1}); // (N,M)
  tvy->split(0, num_devices, false); // (D,N//D,M)
  stage0.addVal({tvx_b, tvx_b, tvxa, tvy});

  // Matmul 1 sharding: Dimension N sharded
  tva->axis(0)->parallelize(ParallelType::DIDx); // column-wise
  tvx_b->axis(0)->parallelize(ParallelType::DIDx);
  tva_b->axis(0)->parallelize(ParallelType::DIDx);
  tvxa->axis(0)->parallelize(ParallelType::DIDx);
  tvy->axis(0)->parallelize(ParallelType::DIDx); // column-wise

  // Matmul 2: y x tvb -> z (N,O)
  TensorView* tvy_b = broadcast(tvy, {false, false, false, true}); // (D,N//D,M,O)
  tvy_b->split(2, num_devices, false); //(D,N//D,D,M//D,O)
  TensorView* tvb_b = broadcast(tvb, {true, false, false}); // (N,M,O)
  tv_b->split(0, num_devices, false); //(D,N//D,M,O)
  tv_b->split(2, num_devices, false); //(D,N//D,D,M//D,O)
  TensorView* tvyb = mul(tvy_b, tvb_b); // (D,N//D,D,M//D,O)
  TensorView* tvz = sum(tvyb, {2,3}); // (D,N//D,O)
  tvz->merge(0, 1) // (N,O)
  stage0.addVal({tvy_b, tvb_b, tvyb, tvz});

  // TODO: Need to shard multiple dimensions of a tensor along same axis
  // TODO: check splitting dimensions
  tvb->axis(0)->parallelize(ParallelType::DIDx); // row-wise
  // Shard two axes along DIDx? What would happen?
  tvy_b->axis(0)->parallelize(ParallelType::DIDx);
  tvy_b->axis(2)->parallelize(ParallelType::DIDx);
  tvb_b->axis(0)->parallelize(ParallelType::DIDx);
  tvb_b->axis(2)->parallelize(ParallelType::DIDx);
  tvyb->axis(0)->parallelize(ParallelType::DIDx);
  tvyb->axis(2)->parallelize(ParallelType::DIDx);

  // Matmul 2: y x tvb -> z (N,O)
  // TensorView* tvy_b = broadcast(tvy, {false, false, true}); // (N,M,O)
  // TensorView* tvb_b = broadcast(tvb, {true, false, false}); // (N,M,O)
  // TensorView* tvyb = mul(tvy_b, tvb_b); // (N,M,O)
  // TensorView* tvz = sum(tvyb, {1}); // (N,O)
  // stage0.addVal({tvy_b, tvb_b, tvyb, tvz});

  // Matmul 1 sharding: Dimension N and M sharded
  // tvb->axis(0)->parallelize(ParallelType::DIDx); // row-wise
  // tvy_b->axis(0)->parallelize(ParallelType::DIDx);
  // tvy_b->axis(1)->parallelize(ParallelType::DIDx);
  // tvb_b->axis(0)->parallelize(ParallelType::DIDx);
  // tvb_b->axis(1)->parallelize(ParallelType::DIDx);
  // tvyb->axis(0)->parallelize(ParallelType::DIDx);
  // tvyb->axis(1)->parallelize(ParallelType::DIDx);


  fusion->addInput(tvx);
  fusion->addInput(tva);
  fusion->addInput(tvb);
  fusion->addOutput(tvz);

  PipelineDescriptor descriptor {
      .stage_descriptors{std::move(stage0)}};
  pipeline = std::make_unique<Pipeline>(fusion.get(), std::move(descriptor));

  // TODO: 1. reorder tvb so that outermost axis is DIDx
  // 2. split tvs so that outermost dimension == num_devices
  int n = num_devices;
  int k = num_devices;
  int m = num_devices;
  int o = 16;
  auto x = at::randn({n, k}, tensor_options);
  auto a = at::randn({k, m}, tensor_options);
  auto b = at::randn({m,o}, tensor_options);
  inputs = {x, a. b};
  auto ref = at::matmul(at::matmul(x, a), b);
  validate();
}


TEST_F(PipelineTest, MegatronAttention) {
  FusionGuard fg(fusion.get());
  PipelineStageDescriptor stage0;
  TensorView* x = makeContigTensor(2);  // (S, D_model)
  TensorView* wq = makeContigTensor(3); // (H,D_model, D_k)
  TensorView* wk = makeContigTensor(3); // (H,D_model, D_k)
  TensorView* wv = makeContigTensor(3); // (H,D_model, D_v)
  TensorView* w = makeContigTensor(2);  // (H*D_v, D_model)
  fusion->addInput(x);
  fusion->addInput(wq);
  fusion->addInput(wk);
  fusion->addInput(wv);
  fusion->addInput(w);

  TensorView* q = Matmul(x, wq, stage0); // (H,S,D_k)
  TensorView* k = Matmul(x, wk, stage0); // (H,S,D_k)
  TensorView* v = Matmul(x, wv, stage0); // (H,S,D_v)

  TensorView* qk = Matmul(k, q, stage0); // (H,D_k,D_k)
  TensorView* qkv = Matmul(qk, v, stage0); // (H,S,D_v)
  qkv->reorder({2, 1})->merge({0,1}); //(S,H*D_v)
  TensorView* z = Matmul(qkv, w, stage0); // (S,D_model)
  fusion->addInput(z);

  // input and output tensors are not sharded 
  // shard multi-attention so that each head's weights
  // and intermediate tensors are on a different device
  wq->axis(2)->parallelize(ParallelType::DIDx);
  wv->axis(2)->parallelize(ParallelType::DIDx); 
  wk->axis(2)->parallelize(ParallelType::DIDx); 
  q->axis(2)->parallelize(ParallelType::DIDx); 
  v->axis(2)->parallelize(ParallelType::DIDx); 
  k->axis(2)->parallelize(ParallelType::DIDx); 
  qk->axis(2)->parallelize(ParallelType::DIDx); 
  qkv->axis(2)->parallelize(ParallelType::DIDx); 
  // matmul's weights after the self attention are sharded row wise
  w->axis(1)->parallelize(ParallelType::DIDx); 

  PipelineDescriptor descriptor {
      .stage_descriptors{std::move(stage0)}};
  pipeline = std::make_unique<Pipeline>(fusion.get(), std::move(descriptor));
  
  int H = 4;
  int S = 8;
  int D_model = 16;
  int D_k = 8;
  int D_v = 8;
  auto x_ = at::randn({S, D_model}, tensor_options);
  auto wk_ = at::randn({H, D_model, D_k}, tensor_options);
  auto wq_ = at::randn({H, D_model, D_k}, tensor_options);
  auto wv_ = at::randn({H, D_model, D_v}, tensor_options);
  auto w_ = at::randn({D_v*H, D_model}, tensor_options);
  // auto b = at::randn({n, m}, tensor_options);
  inputs = {x_, wk_, wq_, wv_, w_};
  auto q_ = at::matmul(x, wq_);
  auto k_ = at::matmul(x, wk_);
  auto v_ = at::matmul(x, wv_);
  auto qk_ = at::matmul(q_, k_);
  auto qkv_ = at::matmul(qk_, v_);
  auto ref = at::matmul(qkv_, w_)
  validate();
}

} // namespace nvfuser

#endif