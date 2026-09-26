"""Level-2 quantization over 32-value groups (granularity 'mx')."""
import threading
import time

import torch

from .kmeans import get_signposts_kMEANS_fp6_parallel, quantize_channels_parallel_kmeans


def _quantize_level2(A, num_signposts=16, iters=3, chunk_size=65536, granularity='mx', group_size=32, block_algo="kmeans"):
    assert A.shape[-1] == 32, "Expected last dim = 32"
    orig_shape = A.shape

    if granularity in ['mx', 'group']:
        s = time.time()
        blocks = A.reshape(-1, group_size)
        N = blocks.shape[0]
        quantized_blocks = torch.empty_like(blocks)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = blocks[start:end]
            
            if block_algo == 'kmeans':
                centers = get_signposts_kMEANS_fp6_parallel(chunk, num_signposts, iters)
            else:
                raise ValueError(f"Unknown block_algo: {block_algo}. Must be 'kmeans' or 'kmedians'.")

            dist = (chunk.unsqueeze(2) - centers.unsqueeze(1)).abs()
            closest = dist.argmin(dim=2)
            quantized_blocks[start:end] = torch.gather(centers, 1, closest)
            del chunk, centers, dist, closest
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"Elapsed: {time.time() - s:.3f}s")
        return quantized_blocks.reshape(orig_shape)
    
    elif granularity == 'channel':
        torch.cuda.reset_peak_memory_stats()
        
        if block_algo == 'kmedians':
            raise ValueError(f"Per-Channel must use 'k-means'.")

        # --- 3D case (Fast K-Means) ---
        if A.ndim == 3:
            s = time.time()
            X, Y, D = A.shape

            # Pre-reshape: Makes all Y-channels contiguous rows. Shape: [Y, X*D]
            A_reshaped = A.permute(1, 0, 2).contiguous().reshape(Y, -1)
            
            # Run quantization on all Y channels in parallel
            quantized_channels = quantize_channels_parallel_kmeans(A_reshaped, num_signposts, iters)

            # Reshape back to original: [Y, X, D] -> [X, Y, D]
            quantized = quantized_channels.reshape(Y, X, D).permute(1, 0, 2).contiguous()
            return quantized
        
        elif A.ndim == 4:
                s = time.time()
                X, Y, Z, D = A.shape
                
                # Pre-reshape: Makes all Y-channels contiguous rows. Shape: [Y, X*Z*D]
                A_reshaped = A.permute(1, 0, 2, 3).contiguous().reshape(Y, -1)

                # Use the fast, parallel, per-channel k-means
                quantized_channels = quantize_channels_parallel_kmeans(A_reshaped, num_signposts, iters)
                
                # quantized: [Y, X, Z, D] -> [X, Y, Z, D]
                quantized = quantized_channels.reshape(Y, X, Z, D).permute(1, 0, 2, 3).contiguous()
                
                peak_mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                return quantized
