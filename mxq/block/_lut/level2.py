"""Level-2 quantization over 32-value groups (granularity 'mx')."""
import threading
import time

import torch

from .kmeans import get_signposts_kMEANS_fp6_parallel

_layer_counter = 0
_layer_counter_lock = threading.Lock()


def _get_next_layer_number():
    """Thread-safe global counter for layer processing"""
    global _layer_counter
    with _layer_counter_lock:
        _layer_counter += 1
        num_gpus = torch.cuda.device_count()
        # Calculate actual layer number (accounting for GPU replicas)
        actual_layer = (_layer_counter + num_gpus - 1) // num_gpus
        return actual_layer


def _quantize_level2(A, num_signposts=16, iters=3, chunk_size=65536, granularity='mx', group_size=32, block_algo="kmeans"):
    assert A.shape[-1] == 32, "Expected last dim = 32"
    orig_shape = A.shape
    
    # Helper to get GPU ID for logging
    def _get_gpu_id(device):
        if device.type == 'cuda':
            return f"GPU {device.index}" if device.index is not None else "GPU 0"
        return "CPU"
    
    gpu_id = _get_gpu_id(A.device)
    layer_num = _get_next_layer_number()

    if granularity in ['mx', 'group']:
        s = time.time()
        # print(f"[{gpu_id}] [Layer {layer_num}] Running '{granularity}' (blockwise) quantization with {block_algo}...")
        blocks = A.reshape(-1, group_size)
        N = blocks.shape[0]
        quantized_blocks = torch.empty_like(blocks)
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = blocks[start:end]
            
            # --- K-Means/K-Medians Switch ---
            if block_algo == 'kmeans':
                # print(f"[{gpu_id}] [Layer {layer_num}] running kmeans")
                centers = get_signposts_kMEANS_fp6_parallel(chunk, num_signposts, iters)
                # log_signposts(centers)
            else:
                raise ValueError(f"Unknown block_algo: {block_algo}. Must be 'kmeans' or 'kmedians'.")
            # --- END: K-Means/K-Medians Switch ---

            dist = (chunk.unsqueeze(2) - centers.unsqueeze(1)).abs()
            closest = dist.argmin(dim=2)
            quantized_blocks[start:end] = torch.gather(centers, 1, closest)
            del chunk, centers, dist, closest
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        print(f"Elapsed: {time.time() - s:.3f}s")
        return quantized_blocks.reshape(orig_shape)
