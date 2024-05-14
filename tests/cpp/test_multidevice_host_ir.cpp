// clang-format off
/*
* SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
* All rights reserved.
* SPDX-License-Identifier: BSD-3-Clause
*/
// clang-format on
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
#include <tests/cpp/multidevice.h>
#include <torch/csrc/jit/codegen/cuda/interface.h>
#include <transform_replay.h>
#include <transform_rfactor.h>

#include <algorithm>
#include <iostream>

#include <host_ir/container.h>
#include <host_ir/executor.h>

namespace nvfuser {


namespace hir {

using MultiDeviceHostIrTestParams = std::tuple<bool,bool>;

class MultiDeviceHostIrTest:
public MultiDeviceTest,
public testing::WithParamInterface<MultiDeviceHostIrTestParams> {};

/*
    This file implements test that combine multidevice communications and host irs. See test_host_irs.cpp for an introduction on host irs and a summary of the different steps necessary to write a host program.

    The host program of the first test could be illustrated as follows:

    tv0_fusion: input, sharded accross devices on its first dimension

    tv1_fusion = Fusion0 (tv0_fusion), on each device

    tv2 = Allgather(tv1_fusion)

    tv2: output
*/

TEST_P(MultiDeviceHostIrTest, SingleFusionSingleComm) {
    auto [use_fusion_executor_cache, with_sharding_annotations] = GetParam();
    
    std::vector<int64_t> unsharded_input_sizes = {2, 8, 32};
    std::vector<int64_t> sharded_input_sizes = unsharded_input_sizes;
    sharded_input_sizes[0] = 1;

    // [Step 1] Define the Fusion we want to execute
    auto fusion = std::make_unique<Fusion>();
    FusionGuard fg(fusion.get());

    auto tv0_fusion = makeConcreteTensor(with_sharding_annotations ? unsharded_input_sizes : sharded_input_sizes);
    auto tv1_fusion = add(tv0_fusion, tv0_fusion);
    fusion->addInput(tv0_fusion);
    fusion->addOutput(tv1_fusion);

    DeviceMesh mesh({0,1});
    if (with_sharding_annotations) {
        for (auto tv: {tv0_fusion,tv1_fusion}){
            tv->setDeviceMesh(mesh);
            tv->axis(0)->parallelize(ParallelType::DIDx);
        }
    }

    // [Step 2)] Instantiate an HostIrContainer
    auto hic = std::make_unique<HostIrContainer>();
    FusionGuard::setCurFusion(hic.get());

    // [Step 3a)] Create a HostUnit Ir holding the fusions
    auto hu = IrBuilder::create<HostUnit>(static_cast<IrContainer*>(hic.get()), std::move(fusion));

    // [Step 3b)] Create a Communication Ir
    CommParams comm_params {
        .type = CommunicationType::Allgather,
        .root = 0,
        .is_root_in_mesh = true,
        .team = mesh.vector()
    };
    auto communication = IrBuilder::create<Communication>(static_cast<IrContainer*>(hic.get()), comm_params);

    // [Step 4)] Create TensorViews at the Host level
    IrCloner ir_cloner(hic.get());
    auto tv0 = ir_cloner.clone(hu->fusion_to_execute()->inputs().at(0)->as<TensorView>());
    auto tv1 = ir_cloner.clone(hu->fusion_to_execute()->outputs().at(0)->as<TensorView>());
    auto tv2 = set(tv1);
    tv2->axis(0)->parallelize(ParallelType::Serial);


      // [Step 5)a.] Create PostOnStream Irs representing executing the Fusion and the Communication
    std::vector<Val*> compute_inputs = {tv0};
    std::vector<Val*> compute_outputs = {tv1};
    auto post_compute = 
        IrBuilder::create<PostOnStream>(static_cast<IrContainer*>(hic.get()), hu, compute_inputs, compute_outputs);
    std::vector<Val*> communication_inputs = {tv1};
    std::vector<Val*> communication_outputs = {tv2};
    auto post_communication = IrBuilder::create<PostOnStream>(static_cast<IrContainer*>(hic.get()), communication, communication_inputs, communication_outputs);

    // [Step 6)] Define the Host program
    hic->pushBackTopLevelExprs(post_compute);
    hic->pushBackTopLevelExprs(post_communication);

    // [Step 7)] Define the Host program's global I/O
    hic->addInput(post_compute->inputs().back());
    hic->addOutput(post_communication->outputs().back());
    hic->print(debug());

    // [Step 8)] Execute the Host program
    HostIrExecutorParams params;
    params.use_fusion_executor_cache = use_fusion_executor_cache;
    if (with_sharding_annotations && use_fusion_executor_cache) {
        // sharding + autoscheduler is not supported at his point
        params.skip_auto_scheduling = true;
    }
    HostIrExecutor hie(std::move(hic), communicator, params);

    auto options = at::TensorOptions().device(communicator->device());
    at::Tensor unsharded_input = at::randn(unsharded_input_sizes, options);
    c10::IValue input = unsharded_input.slice(0, communicator->deviceId(), communicator->deviceId() + 1);
    at::Tensor output = at::empty(unsharded_input_sizes, options);
    auto ref_output = unsharded_input * 2;

    auto outputs = hie.runWithInput({{post_compute->inputs().back(), input},
        {post_communication->outputs().back(), output}});

    // validate the obtained results
    GTEST_EXPECT_TRUE(torch::allclose(ref_output, outputs.back()));
}

INSTANTIATE_TEST_SUITE_P(
    Manual,
    MultiDeviceHostIrTest,
    testing::Combine(testing::Bool(), testing::Bool()),
    [](const testing::TestParamInfo<MultiDeviceHostIrTestParams>& info) -> std::string {
        std::string s;
        s += std::get<0>(info.param) ? "useFusionExecutorCache"
                                  : "useFusionExecutor";
        s += "_";
        s += std::get<1>(info.param) ? "withShardingAnnotations"
                                  : "withoutShardingAnnotations";
      return s;
    });

} // namespace hir

} // namespace nvfuser
