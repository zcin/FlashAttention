# Flash Attention

CUDA Kernel for Multi-Head Self Attention, using FlashAttention technique. Profiled with Nsight Compute.

Optimizations inspired by:
- [Flash Attention](https://arxiv.org/abs/2205.14135)
- [Flash Attention 2](https://arxiv.org/abs/2307.08691)
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
- [Optimizing Parallel Reduction in CUDA](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)

Techniques used:
- Loads K tiles from global memory in transposed form for efficient matrix multiplication, to avoid explicit transposing.
- Memory coalesced operations.

Profile results (run on NVIDIA A10G GPU on Lambda Labs):
| N | d_model | num_heads | # cycles | time(ms) | Compute Throughput |
| - | - | - | - | - | - |
| 8192 | 128 | 4 | 47998507 | 54.24 | 69.45 |

# Dependencies

`sudo apt install nsight-compute`

# To run and check the program.
1. `nvcc -o mhattention multi_head_attention.cu`
2. `python check_attention.py generate N d h`
3. `./mhattention N d h` or `sudo ncu --set full ./mhattention N d h` to profile
4. `python check_attention.py N d h`
