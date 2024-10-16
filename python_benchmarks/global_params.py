import torch
from typing import Union, List, Tuple
from nvfuser import DataType
from .core import DEVICE_PROPERTIES
import numpy as np
import itertools
import os

# BENCHMARK_MODE = weekly/nightly.
BENCHMARK_MODE = os.getenv("BENCHMARK_MODE")
if not BENCHMARK_MODE:
    BENCHMARK_MODE = "nightly"

# Datatypes to benchmark
FLOAT_DTYPES = [torch.float32]
# Run only one of float16 / bfloat16.
if DEVICE_PROPERTIES["gpu_compute_capability_major"] >= 8:
    FLOAT_DTYPES.append(torch.bfloat16)
else:
    FLOAT_DTYPES.append(torch.float16)

# Datatypes that will be promoted to Datatype.Float in Fusion Definitions
PROMOTE_DTYPES = [DataType.BFloat16, DataType.Half]

# Model Parameters from LLMs (GPT2/3, PaLM, LLama)

# Embedding size: d_model, d_ff = 4 * d_model
D_MODEL_MIN = 768
D_MODEL_MAX = 18432

# (num_heads, n_embd) configurations seen in models.
LLM_CONFIGS = [
    (12, 768),  # GPT-2 (124M), GPT-3 (125M)
    (16, 1024),  # GPT-2 (350M), GPT-3 (350M)
    (20, 1280),  # GPT-2 (774M)
    (16, 1536),  # GPT-3 (760M)
    (25, 1600),  # GPT-2 (1558M)
    (24, 2048),  # GPT-3 (1.3B)
    (32, 2560),  # GPT-3 (2.7B)
    (16, 4096),  # PaLM (8B)
    (32, 4096),  # LLaMA (7B), GPT-3 (6.7B)
    (40, 5120),  # LLaMA (13B), GPT-3 (13B)
    (52, 6656),  # LLaMA (30B)
    (32, 8192),  # PaLM (63B)
    (64, 8192),  # LLaMA (65B)
    (96, 12288),  # GPT-3 (175B)
    (48, 18432),  # PaLM (540B)
]


# Utility function to generate input sizes for benchmarks
def generate_input_sizes(dims: Union[int, List] = 2) -> List[Tuple]:
    """
    The weekly vs nightly input ranges only differ for 2D inputs currently.
    Nightly input range:
        Batch size: [16->16384] Hidden size: [768, 4*18432] (step size = 256)
    Weekly input range:
        Batch size:
            [16]: Latency bound state
            [512, 1024]: Just filled the machine
            [16384]: Steady state (full machine)
        Hidden size: [768, 4*18432] (step size = 8)
    Note: The hidden size is restricted to 2 * 18432 for the batch size 16384 to avoid OOM.
    """
    inputs = []
    if isinstance(dims, int):
        dims = [dims]

    for dim in dims:
        if dim == 2:
            input_ranges = []

            step_size = 256
            # max_batch_range: set according to max size that fits in GPU memory
            batch_range = [2**i for i in range(4, 14)]  # {16, 8192}

            if BENCHMARK_MODE == "weekly":
                step_size = 8
                batch_range = [16, 512, 1024]

            # max_hidden_size = 4 * d_model_max (max hidden size in feedforward layers)
            # NOTE: Numpy arrays are not JSON serializable so convert them to enable storing benchmark data.
            hidden_range = np.arange(
                D_MODEL_MIN, 4 * D_MODEL_MAX + 1, step_size
            ).tolist()  # (768, 4*18432)
            input_ranges.append((batch_range, hidden_range))

            # Reduce max hidden size for largest batch size (16384) to avoid OOM in RMSNorm.
            hidden_range = np.arange(
                D_MODEL_MIN, 2 * D_MODEL_MAX + 1, step_size
            ).tolist()  # (768, 2*18432)
            input_ranges.append(([16384], hidden_range))
            for batch_range, hidden_range in input_ranges:
                inputs.extend(list(itertools.product(batch_range, hidden_range)))

        elif dim == 3:
            dim_range = [2**i for i in range(1, 10)]
            inputs.extend(list(itertools.product(dim_range, repeat=3)))
        elif dim == 4:
            # TODO: Add spatial_dim = 2.
            input_ranges = []

            batch_range = [2**i for i in range(1, 10)]  # {2, 512}
            channel_range = [2**i for i in range(1, 8)]  # {2, 128}
            spatial_range = [2**i for i in range(2, 7)]  # {4, 64}
            input_ranges.append((batch_range, channel_range, spatial_range))

            batch_range = [2**i for i in range(1, 7)]  # {2, 64}
            channel_range = [2**i for i in range(1, 6)]  # {2, 32}
            spatial_range = [128, 256]
            input_ranges.append((batch_range, channel_range, spatial_range))

            # Resnet/ResNext sizes
            batch_range = [2**i for i in range(5, 9)]  # {32, 256}
            channel_range = [2**i for i in range(6, 9)]  # {64, 256}
            spatial_range = [7 * 2**i for i in range(5)]  # {7, 112}
            input_ranges.append((batch_range, channel_range, spatial_range))

            for batch_range, channel_range, spatial_range in input_ranges:
                inputs.extend(
                    [
                        (n, c, hw, hw)
                        for (n, c, hw) in itertools.product(
                            batch_range, channel_range, spatial_range
                        )
                    ]
                )

            inputs.extend(
                [
                    (n, c, hw, hw)
                    for (n, (c, hw)) in itertools.product(
                        [128, 256],
                        [
                            (512, 7),
                            (512, 14),
                            (512, 28),
                            (1024, 7),
                            (1024, 14),
                            (2048, 7),
                        ],
                    )
                ]
            )

        else:
            raise NotImplementedError(
                f"Generating input sizes of dimension {dim} is not implemented"
            )
    return inputs


# Utility function to generate input sizes for attention benchmarks.
def generate_attn_inputs():
    batch_range = [16, 32]
    seq_lengths = [2**i for i in range(2, 10)]  # {4, 512}
    inputs = [
        (batch_size, seq_len, nh, n_embd)
        for (batch_size, seq_len, (nh, n_embd)) in itertools.product(
            batch_range, seq_lengths, LLM_CONFIGS
        )
    ]
    return inputs
