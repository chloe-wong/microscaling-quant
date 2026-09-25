"""Mapping every value to its nearest signpost; per-channel (row) quantization."""
import torch

from .kmeans import get_signposts_kMEANS_fp6_parallel

def quantize_channels_parallel_kmeans(channels_batch, num_signposts, iters, chunk_size=8192, random_init=False):
    """Optimized quantization with CHANNEL BATCHING for large inputs."""
    Y, N = channels_batch.shape
    
    needs_channel_batching = False
    batch_size = None
    
    dist_mem_bytes = Y * N * num_signposts * 4
    dist_mem_gb = dist_mem_bytes / (1024**3)
    
    if dist_mem_gb > 6.0 and Y > 512:
        if torch.cuda.is_available():
            free_mem, _ = torch.cuda.mem_get_info()
            target_mem_per_batch = 2 * 1024**3
            mem_per_channel = N * num_signposts * 4
            batch_size = max(256, min(2048, int(target_mem_per_batch / mem_per_channel)))
            needs_channel_batching = batch_size < Y
   
    if needs_channel_batching:
        all_quantized = []
        
        for ch_start in range(0, Y, batch_size):
            ch_end = min(ch_start + batch_size, Y)
            # print(f"  Channel batch [{ch_start}:{ch_end}]")
            batch_data = channels_batch[ch_start:ch_end]
            
            # Get centers for this batch
            batch_centers = get_signposts_kMEANS_fp6_parallel(batch_data, num_signposts, iters, chunk_size, random_init)
            
            # Quantize this batch
            batch_quantized = _quantize_with_centers(batch_data, batch_centers, num_signposts)
            
            all_quantized.append(batch_quantized)
            
            del batch_data, batch_centers, batch_quantized
            # Only clear cache between channel batches (not inside loops)
            if torch.cuda.is_available() and (ch_end - ch_start) > 512:
                torch.cuda.empty_cache()
        
        quantized_output = torch.cat(all_quantized, dim=0)
        del all_quantized
        return quantized_output
    
    # Standard path: no channel batching needed
    centers = get_signposts_kMEANS_fp6_parallel(channels_batch, num_signposts, iters, chunk_size, random_init)
    quantized_output = _quantize_with_centers(channels_batch, centers, num_signposts)
    
    return quantized_output


def _quantize_with_centers(channels_batch, centers, num_signposts):
    """Helper to quantize given pre-computed centers."""
    Y, N = channels_batch.shape
    dist_mem_bytes = Y * N * num_signposts * 4
    
    if torch.cuda.is_available():
        free_mem, _ = torch.cuda.mem_get_info()
        safe_mem = int(free_mem * 0.4)  # More conservative (down from 0.5)
        needs_chunking = dist_mem_bytes > safe_mem
        
        if needs_chunking:
            dist_mem_per_row = N * num_signposts * 4
            total_mem_per_row = int(dist_mem_per_row * 1.5)  # More overhead (up from 1.3)
            optimal_quant_chunk = max(64, safe_mem // total_mem_per_row)  # Lower minimum
            quant_chunk_size = min(optimal_quant_chunk, Y)
        else:
            quant_chunk_size = Y
    else:
        needs_chunking = dist_mem_bytes > (1 * 1024**3)
        quant_chunk_size = min(2048, Y) if needs_chunking else Y
    
    if needs_chunking:
        quantized_chunks = []
        chunk_start = 0
        
        while chunk_start < Y:
            chunk_end = min(chunk_start + quant_chunk_size, Y)
            
            # Only check/adjust memory if this is a large chunk
            if torch.cuda.is_available() and (chunk_end - chunk_start) > 256:
                free_now, _ = torch.cuda.mem_get_info()
                chunk_mem_needed = (chunk_end - chunk_start) * N * num_signposts * 4
                
                if chunk_mem_needed > free_now * 0.5:  # More conservative
                    max_rows = max(32, int((free_now * 0.3) / (N * num_signposts * 4 * 1.5)))  # Lower limits
                    new_chunk_end = min(chunk_start + max_rows, Y)
                    if new_chunk_end > chunk_start:
                        chunk_end = new_chunk_end
            
            chunk = channels_batch[chunk_start:chunk_end]
            chunk_centers = centers[chunk_start:chunk_end]
            
            try:
                dist = (chunk.unsqueeze(2) - chunk_centers.unsqueeze(1)).abs()
                closest = dist.argmin(dim=2)
                q_chunk = torch.gather(chunk_centers, 1, closest)
                quantized_chunks.append(q_chunk)
                del dist, closest
            except RuntimeError as e:
                if "out of memory" in str(e):
                    # print(f"[Emergency] OOM, row-by-row fallback")
                    torch.cuda.empty_cache()  # Only clear on actual OOM
                    q_chunk_list = []
                    for row_idx in range(chunk.shape[0]):
                        row = chunk[row_idx:row_idx+1]
                        row_centers = chunk_centers[row_idx:row_idx+1]
                        dist_row = (row.unsqueeze(2) - row_centers.unsqueeze(1)).abs()
                        closest_row = dist_row.argmin(dim=2)
                        q_row = torch.gather(row_centers, 1, closest_row)
                        q_chunk_list.append(q_row)
                        del row, row_centers, dist_row, closest_row
                        
                    q_chunk = torch.cat(q_chunk_list, dim=0)
                    quantized_chunks.append(q_chunk)
                    del q_chunk_list
                else:
                    raise
            
            del chunk, chunk_centers
            chunk_start = chunk_end
        
        quantized_output = torch.cat(quantized_chunks, dim=0)
        del quantized_chunks
    else:
        dist = (channels_batch.unsqueeze(2) - centers.unsqueeze(1)).abs()
        closest = dist.argmin(dim=2)
        quantized_output = torch.gather(centers, 1, closest)
        del dist, closest
    
    return quantized_output
