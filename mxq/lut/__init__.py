"""mxq.lut — a look-up table per group of codes: k-means entries, snapped to the FP6 E3M2 codebook.

    I, T = fit(groups, num_signposts=16, iters=3)      groups: (G, n), one table per row
    codes = T.gather(1, I)

    I   (G, n) int64, each value's index into its row's table
    T   (G, num_signposts) the table entries, sorted

The rows carry no layout: the caller decides what a group is (mxq.block.lut: a block, or a channel).

    kmeans      choosing the entries per row, then snapping them to the codebook
    assign      fit: batching over rows to fit in memory, and each value's nearest entry
    codebook    the grid the entries snap to
"""
from .assign import fit

__all__ = ["fit"]
