"""Choosing the 16 signposts (table entries) per row: k-means, then snap to the codebook."""
import torch

from .codebook import E3M2_CODEBOOK

def get_signposts_kMEANS_fp6_chunked(channels_batch, num_signposts, iters, chunk_size, random_init=False):
    """Chunked version."""
    Y, N = channels_batch.shape
    all_centers = []
    
    num_chunks = (Y + chunk_size - 1) // chunk_size
    
    for i, chunk_start in enumerate(range(0, Y, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, Y)
        chunk = channels_batch[chunk_start:chunk_end]
        
        chunk_centers = get_signposts_kMEANS_fp6_parallel_improved_fast(chunk, num_signposts, iters, random_init)
        all_centers.append(chunk_centers.cpu())
        
        del chunk, chunk_centers
    
    result = torch.cat(all_centers, dim=0).to(channels_batch.device)
    del all_centers
    return result


def get_signposts_kMEANS_fp6_parallel_improved_fast(channels_batch, num_signposts=16, iters=0, random_init=False):
    """Optimized K-MEANS with smart memory management."""
    Y, N = channels_batch.shape
    device, dtype = channels_batch.device, channels_batch.dtype
    fp6_codebook = torch.tensor(E3M2_CODEBOOK, device=device, dtype=dtype)

    # Initialization
    if random_init:
        rand_indices = torch.randint(0, N, (Y, num_signposts), device=device)
        centers = torch.gather(channels_batch, 1, rand_indices)
    else:
        sorted_vals, _ = channels_batch.sort(dim=1)
        q_idx = torch.linspace(0, N - 1, num_signposts, device=device).long()
        centers = sorted_vals[:, q_idx]

    # K-MEANS ITERATIONS
    for iter_num in range(iters):

        # Determine batch size based on available memory (MORE CONSERVATIVE)
        if torch.cuda.is_available():
            free_mem, _ = torch.cuda.mem_get_info()
            # Use only 30% of free memory (very conservative) to account for fragmentation and overhead
            usable_mem = int(free_mem * 0.6)
            # Memory per mini-batch: N * num_signposts * 4 bytes per element * mini_batch_size
            # Add 50% overhead for intermediate operations and memory fragmentation
            mem_per_row = N * num_signposts * 4 * 1.5
            optimal_batch = max(128, int(usable_mem / mem_per_row))
            mini_batch_size = min(optimal_batch, Y)
        else:
            mini_batch_size = min(4096, Y)
        
        # Assignment phase
        clusters = torch.empty(Y, N, dtype=torch.long, device=device)
        
        for mb_start in range(0, Y, mini_batch_size):
            mb_end = min(mb_start + mini_batch_size, Y)
            mb_data = channels_batch[mb_start:mb_end]
            mb_centers = centers[mb_start:mb_end]
            
            try:
                dist = (mb_data.unsqueeze(2) - mb_centers.unsqueeze(1)).abs()
                clusters[mb_start:mb_end] = dist.argmin(dim=2)
                del dist
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # Emergency fallback: process row by row
                    torch.cuda.empty_cache()
                    for row_idx in range(mb_data.shape[0]):
                        row = mb_data[row_idx:row_idx+1]
                        row_centers = mb_centers[row_idx:row_idx+1]
                        dist_row = (row.unsqueeze(2) - row_centers.unsqueeze(1)).abs()
                        clusters[mb_start + row_idx] = dist_row.argmin(dim=2)
                        del row, row_centers, dist_row
                    del mb_data, mb_centers
                    
                    # After the row-by-row loop
                    assert clusters[mb_start:mb_end].min() >= 0, "Negative cluster index!"
                    assert clusters[mb_start:mb_end].max() < num_signposts, f"Cluster index {clusters[mb_start:mb_end].max()} >= {num_signposts}!"
                    
                else:
                    raise

        # Update phase
        if torch.cuda.is_available():
            free_mem_now, _ = torch.cuda.mem_get_info()
            one_hot_mem = Y * N * num_signposts * 4
            use_fast_path = one_hot_mem < (free_mem_now * 0.4)  # More conservative
        else:
            use_fast_path = True
        
        if use_fast_path:
            try:
                one_hot = torch.nn.functional.one_hot(clusters, num_classes=num_signposts).to(dtype)
                counts = one_hot.sum(dim=1).clamp_min(1.0)
                sums = (channels_batch.unsqueeze(2) * one_hot).sum(dim=1)
                new_centers = sums / counts
                del one_hot, sums, counts
            except RuntimeError:
                # if iter_num == 0:
                #     print("Fast path OOM, using scatter")
                sums = torch.zeros(Y, num_signposts, device=device, dtype=dtype)
                counts = torch.zeros(Y, num_signposts, device=device, dtype=dtype)
                sums.scatter_add_(dim=1, index=clusters, src=channels_batch)
                counts.scatter_add_(dim=1, index=clusters, src=torch.ones_like(channels_batch, dtype=dtype))
                counts = counts.clamp_min(1.0)
                new_centers = sums / counts
                del sums, counts
        else:
            # if iter_num == 0:
            #     print("Using memory-safe scatter for updates")
            sums = torch.zeros(Y, num_signposts, device=device, dtype=dtype)
            counts = torch.zeros(Y, num_signposts, device=device, dtype=dtype)
            sums.scatter_add_(dim=1, index=clusters, src=channels_batch)
            counts.scatter_add_(dim=1, index=clusters, src=torch.ones_like(channels_batch, dtype=dtype))
            counts = counts.clamp_min(1.0)
            new_centers = sums / counts
            del sums, counts
        
        centers = new_centers
        del clusters

    # Final snap to FP6
    centers = fp6_codebook[torch.argmin(
        (centers.unsqueeze(-1) - fp6_codebook.view(1, 1, -1)).abs(), dim=-1
    )]
    
    centers, _ = centers.sort(dim=1)
    
    # # DEBUG: Check centers are valid
    # if torch.isnan(centers).any() or torch.isinf(centers).any():
    #     print(f"[ERROR] Invalid centers: NaN={torch.isnan(centers).any()}, Inf={torch.isinf(centers).any()}")
    # print(f"[DEBUG K-means] Centers range: [{centers.min().item():.6f}, {centers.max().item():.6f}], unique={torch.unique(centers.float()).numel()}")
    
    return centers


def get_signposts_kMEANS_fp6_parallel(channels_batch, num_signposts=16, iters=0, chunk_size=8192, random_init=False):
    """Entry point with adaptive chunking."""
    Y, N = channels_batch.shape
    
    if torch.cuda.is_available():
        # Only clear cache once at entry point if memory is tight
        free_mem, total_mem = torch.cuda.mem_get_info()
        if free_mem < total_mem * 0.3:  # Only if <30% free
            torch.cuda.empty_cache()
            free_mem, _ = torch.cuda.mem_get_info()
            
        est_mem_needed = Y * N * num_signposts * 8
        
        if est_mem_needed > (free_mem * 0.5):  # More conservative threshold
            optimal_chunk = int((free_mem * 0.3) / (N * num_signposts * 8))  # Use 30% not 50%
            chunk_size = max(256, min(optimal_chunk, Y))  # Lower minimum chunk size
            # print(f"[K-means] Chunking {Y} channels into chunks of {chunk_size}")
            return get_signposts_kMEANS_fp6_chunked(channels_batch, num_signposts, iters, chunk_size, random_init)
    
    return get_signposts_kMEANS_fp6_parallel_improved_fast(channels_batch, num_signposts, iters, random_init)
