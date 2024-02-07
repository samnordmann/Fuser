// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on

#include <ops/all_ops.h>
#include <kernel_ir.h>
#include <ir/builder.h>
#include <ir/cloner.h>
#include <ir/utils.h>
#include <host_ir.h>


namespace nvfuser {

namespace hir {

using kir::ForLoop;

std::unique_ptr<HostFusion> makeHostFusionFromFusion(Fusion* fusion) {
    auto host_fusion = std::make_unique<HostFusion>();
    FusionGuard fg(host_fusion.get());
    host_fusion->gpu_fusion = fusion;

    auto ir_container = static_cast<IrContainer*>(host_fusion.get());
    auto tvs = ir_utils::filterByType<TensorView>(fusion->vals());
    NVF_ERROR(std::all_of(tvs.begin(), tvs.end(),
                [](auto tv) {return tv->getLeafDomain().at(0)->isCPUDim();}
                ), "Need the outmost of all tvs to be of parallel type host");

    // Input
    NVF_ERROR(fusion->inputs().size()==1, "there must be exactly one input");
    IrCloner ir_cloner(host_fusion.get());
    // TensorView* input_tv = IrBuilder::create<TensorView>(ir_container, fusion->inputs().at(0)->as<TensorView>(), ir_cloner);
    TensorView* input_tv = ir_cloner.clone(fusion->inputs().at(0)->as<TensorView>());

    //For Loop
    // IterDomain* id = IrBuilder::create<IterDomain>(ir_container, fusion->inputs().at(0)->as<TensorView>()->getLeafDomain().at(0));
    IterDomain* id = input_tv->getLeafDomain().at(0);
    Val* index = IrBuilder::create<Val>(ir_container, 0, DataType::Index);
    Val* start = IrBuilder::create<Val>(ir_container, 0, DataType::Index);
    Val* stop = id->extent();
    Val* step = IrBuilder::create<Val>(ir_container, 1, DataType::Index);

    auto for_loop = IrBuilder::create<kir::ForLoop>(ir_container,
            id, index, start, stop, step, false, nullptr, false, DoubleBufferLoopStage::NotApplicable);

    Val* one = IrBuilder::create<Val>(ir_container, 1, DataType::Index);
    // struct Slice {
    //     Val* start = nullptr;
    //     Val* stop = nullptr;
    //     Val* step = nullptr;
    // };
    std::vector<Slice> ranges(input_tv->getLeafDomain().size());
    ranges.at(0).start = index;
    ranges.at(0).step = one;
    ranges.at(0).stop = add(index, one);

    TensorView* sliced_input = slice(input_tv, ranges);


    // Scope


    // new IR: post kernel, pointing to the original fusion



    std::cout << for_loop->toString() << std::endl;
    std::cout << sliced_input->toString() << std::endl;




    return host_fusion;
}

} // namespace hir

} // namespace nvfuser



    // auto host_fusion = std::make_unique<HostFusion>();
    // IrCloner cloner (static_cast<IrContainer*>(host_fusion.get()));
    // for (auto input : fusion->inputs()) {
    //     auto new_input = IrBuilder::clone<TensorView>(input->as<TensorView>(), &cloner);
    //     host_fusion->addInput(new_input);
    // }

    // for (auto output : fusion->outputs()) {
    //     auto new_output = IrBuilder::clone<TensorView>(output->as<TensorView>(), &cloner);
    //     host_fusion->addOutput(new_output);
    // }

    // auto ca_map = ComputeAtMap(host_fusion.get());
    // std::unordered_map<IterDomain*, IterDomain*> concrete_to_reference_map;
    // for (auto tv : ir_utils::filterByType<TensorView>(host_fusion->vals())) {
    //     for (auto id : tv->getLeafDomain()) {
    //         auto ca_id = ca_map.getConcreteMappedID(id, IdMappingMode::PERMISSIVE_RESIZE);
    //         concrete_to_reference_map[ca_id] = id;
    //     }        
    // }

