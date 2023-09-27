// clang-format off
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-present NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */
// clang-format on
#ifdef USE_DISTRIBUTED
#include <ir/utils.h>
#include <multidevice/allocator.h>
#include <fusion.h>
#include <executor.h>
#include <ir/cloner.h>

namespace nvfuser {

std::pair<std::unique_ptr<Fusion>, std::unordered_map<Val*, Val*>> copyFusionAndChangeOutputs(Fusion* fusion, std::unordered_set<Val*> outputs) {
    std::unique_ptr<Fusion> fusion_copy = std::make_unique<Fusion>();
    std::unordered_map<Val*, Val*> copy_to_original_map;
    auto original_to_copy_cloner = Fusion::copy(fusion, fusion_copy.get());

    auto original_inputs = fusion_copy->inputs();
    auto original_outputs = fusion_copy->outputs();

    // Remove original outputs
    std::for_each(
        original_outputs.begin(), original_outputs.end(), [&](auto& output) {
        fusion_copy->removeOutput(output);
        });

    // Add new outputs
    std::for_each(
        outputs.begin(),
        outputs.end(),
        [&](Val* const& output) {
        fusion_copy->addOutput(original_to_copy_cloner.clone(output));
        copy_to_original_map[original_to_copy_cloner.clone(output)] = output;
        });

    for (auto tv : ir_utils::filterByType<TensorView>(fusion_copy->vals())) {
      tv->setMemoryType(MemoryType::Global);
      for (auto i : c10::irange(tv->domain()->nDims())) {
        tv->axis(i)->parallelize(ParallelType::Serial);
      }
    }

    return std::make_pair<std::unique_ptr<Fusion>, std::unordered_map<Val*, Val*>>(std::move(fusion_copy), std::move(copy_to_original_map));
}


std::unordered_map<Val*, c10::IValue> allocatePipelineIntermediateBuffers(Pipeline* pipeline, DeviceIdxType dId, std::vector<c10::IValue> global_inputs_IValues) {
    std::unordered_set<Val*> vals_to_allocate;
    std::unordered_set<Val*> vals_to_not_allocate;
    const auto& exprs = pipeline->exprs();
    // std::cout << "RANK " << dId << " inside allocatePipelineIntermediateBuffers entering:" << std::endl;
    for (auto stage: ir_utils::filterByType<PipelineStage>(exprs)) {
        // std::cout << "RANK " << dId << " inside allocatePipelineIntermediateBuffers handling stage:" << stage << std::endl;
        if (stage->descriptor()->mesh.has(dId)) {
            // std::cout << "RANK " << dId << " inside allocatePipelineIntermediateBuffers mesh has it" << std::endl;
            for (auto input: stage->inputs()) {
                // std::cout << "RANK " << dId << " inside allocatePipelineIntermediateBuffers handling input" << input << std::endl;
                auto input_val = input->as<PipelineVal>()->getOriginalVal();
                vals_to_allocate.insert(input_val);
            }
            // for (auto output: stage->outputs()) {
            //     // std::cout << "RANK " << dId << " inside allocatePipelineIntermediateBuffers handling output" << output << std::endl;
            //     auto output_val = output->as<PipelineVal>()->getOriginalVal();
            //     vals_to_not_allocate.insert(output_val);
            // }
        }
    }
    for (auto global_input: pipeline->originalFusion()->inputs()) {
        // std::cout << "RANK " << dId << " inside allocatePipelineIntermediateBuffers handling  GLOBAL input" << global_input << std::endl;
        vals_to_not_allocate.insert(global_input);
    }

    std::stringstream ss;
    ss << "RANK " << dId << " has vals_to_allocate:{\n";
    for (auto val_to_allocate: vals_to_allocate) {
        ss << "   " << val_to_allocate << ", ";
    }
    ss << "\n}";
    ss << "RANK " << dId << " has vals_to_not_allocate:{\n";
    for (auto val_to_not_allocate: vals_to_not_allocate) {
        ss << "   " << val_to_not_allocate << ", ";
    }
    ss << "\n}";
    sleep(dId * 2);
    std::cout << ss.str() << std::endl;

    // std::cout << "RANK " << dId << " erase:" << std::endl;
    for (auto val_to_not_allocate: vals_to_not_allocate){
        vals_to_allocate.erase(val_to_not_allocate);
    }
    // vals_to_allocate.erase(vals_to_not_allocate.begin(), vals_to_not_allocate.end());
    // std::cout << "RANK " << dId << " inside allocatePipelineIntermediateBuffers entering copyFusionAndChangeOutputs:" << std::endl;
    auto [fusion_copy, copy_to_original_map] = copyFusionAndChangeOutputs(pipeline->originalFusion(), vals_to_allocate);
    FusionExecutor fe;
    // std::cout << "RANK " << dId << " inside allocatePipelineIntermediateBuffers compiling:" << std::endl;
    fe.compileFusion(fusion_copy.get(), global_inputs_IValues);
    // std::cout << "RANK " << dId << " inside allocatePipelineIntermediateBuffers allocOutputSpace:" << std::endl;
    auto buffers = fe.allocOutputSpace(global_inputs_IValues);

    std::unordered_map<Val*, c10::IValue> allocations;
    // std::cout << "RANK " << dId << " inside allocatePipelineIntermediateBuffers emplacing:" << std::endl;
    for (auto i: c10::irange(buffers.size())) {
        // std::cout << "fusion_copy->outputs().at(i) " << fusion_copy->outputs().at(i) << std::endl;
        // std::cout << "buffers.at(i) " << buffers.at(i) << std::endl;
        // std::cout << "copy_to_original_map[fusion_copy->outputs().at(i)] " << copy_to_original_map[fusion_copy->outputs().at(i)] << std::endl;
        // allocations.emplace(vals_to_allocate.at(i), buffers.at(i));
        allocations.emplace(copy_to_original_map[fusion_copy->outputs().at(i)], buffers.at(i));
    }

    return allocations;
}


} // namespace nvfuser

#endif
