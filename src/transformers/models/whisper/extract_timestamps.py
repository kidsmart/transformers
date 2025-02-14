import numpy as np
from typing import List, Tuple, Optional, Union


def median_filter(inputs: np.ndarray, filter_width: int) -> np.ndarray:
    """
    Applies a median filter along the last dimension of the input.
    
    Args:
        inputs: Input array to filter
        filter_width: Width of the median filter (must be odd)
    Returns:
        Filtered array
    """
    if filter_width <= 0 or filter_width % 2 != 1:
        raise ValueError("`filter_width` should be an odd number")
    
    pad_width = filter_width // 2
    if inputs.shape[-1] <= pad_width:
        return inputs
    
    # Pad the edges
    padded = np.pad(inputs, ((0, 0), (0, 0), (pad_width, pad_width)), mode='reflect')
    
    # Create a sliding window view
    windows = np.lib.stride_tricks.sliding_window_view(padded, filter_width, axis=-1)
    return np.median(windows, axis=-1)


def dynamic_time_warping(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Dynamic Time Warping between input audio and output tokens.
    
    Args:
        matrix: Similarity matrix between audio and tokens
    Returns:
        Tuple of (text_indices, time_indices)
    """
    output_length, input_length = matrix.shape
    cost = np.full((output_length + 1, input_length + 1), np.inf, dtype=np.float32)
    trace = np.full((output_length + 1, input_length + 1), -1, dtype=np.float32)
    
    cost[0, 0] = 0
    for j in range(1, input_length + 1):
        for i in range(1, output_length + 1):
            c0 = cost[i - 1, j - 1]
            c1 = cost[i - 1, j]
            c2 = cost[i, j - 1]
            
            if c0 < c1 and c0 < c2:
                c, t = c0, 0
            elif c1 < c0 and c1 < c2:
                c, t = c1, 1
            else:
                c, t = c2, 2
                
            cost[i, j] = matrix[i - 1, j - 1] + c
            trace[i, j] = t
    
    # Backtrace
    i = trace.shape[0] - 1
    j = trace.shape[1] - 1
    trace[0, :] = 2
    trace[:, 0] = 1
    
    text_indices = []
    time_indices = []
    
    while i > 0 or j > 0:
        text_indices.append(i - 1)
        time_indices.append(j - 1)
        if trace[i, j] == 0:
            i -= 1
            j -= 1
        elif trace[i, j] == 1:
            i -= 1
        elif trace[i, j] == 2:
            j -= 1
        else:
            raise RuntimeError(f"Internal error in dynamic time warping. Unexpected trace[{i}, {j}]")
    
    return np.array(text_indices[::-1]), np.array(time_indices[::-1])


def extract_token_timestamps(
    cross_attentions: List[np.ndarray],
    sequences: np.ndarray,
    alignment_heads: List[Tuple[int, int]],
    median_filter_width: int = 7,
    time_precision: float = 0.02,
    num_frames: Optional[Union[int, np.ndarray]] = None
) -> np.ndarray:
    """
    Calculate token timestamps from cross attention data.
    
    Args:
        cross_attentions: List of cross attention arrays from each decoder layer
        sequences: Array of token sequences
        alignment_heads: List of (layer_idx, head_idx) tuples for timestamp calculation
        median_filter_width: Width of median filter for smoothing
        time_precision: Time precision in seconds
        num_frames: Optional number of frames to consider
    
    Returns:
        Array of timestamps for each token
    """
    # Stack and select specific cross-attention layers and heads
    weights = np.stack([cross_attentions[l][:, h] for l, h in alignment_heads])
    weights = np.transpose(weights, (1, 0, 2, 3))
    
    batch_size = sequences.shape[0]
    input_length = cross_attentions[0].shape[2]
    timestamps = np.zeros((batch_size, input_length + 1), dtype=np.float32)
    
    # Handle num_frames if specified
    if num_frames is not None:
        if isinstance(num_frames, (int, np.integer)):
            weights = weights[..., :num_frames // 2]
        elif isinstance(num_frames, np.ndarray) and np.unique(num_frames).size == 1:
            weights = weights[..., :num_frames[0] // 2]
        else:
            num_frames = np.asarray(num_frames)
    
    if num_frames is None or isinstance(num_frames, (int, np.integer)):
        # Normalize and smoothen weights
        std = np.std(weights, axis=-2, keepdims=True)
        mean = np.mean(weights, axis=-2, keepdims=True)
        weights = (weights - mean) / (std + 1e-6)  # Add epsilon to prevent division by zero
        weights = median_filter(weights, median_filter_width)
        weights = np.mean(weights, axis=1)  # Average attention heads
    
    # Process each batch element
    for batch_idx in range(batch_size):
        if num_frames is not None and isinstance(num_frames, np.ndarray):
            matrix = weights[batch_idx, ..., :num_frames[batch_idx] // 2]
            std = np.std(matrix, axis=-2, keepdims=True)
            mean = np.mean(matrix, axis=-2, keepdims=True)
            matrix = (matrix - mean) / (std + 1e-6)
            matrix = median_filter(matrix, median_filter_width)
            matrix = np.mean(matrix, axis=0)
        else:
            matrix = weights[batch_idx]
        
        # Compute timestamps using DTW
        text_indices, time_indices = dynamic_time_warping(-matrix)
        jumps = np.pad(np.diff(text_indices), (1, 0), constant_values=1).astype(bool)
        jump_times = time_indices[jumps] * time_precision
        timestamps[batch_idx, 1:] = jump_times
    
    return timestamps


def calculate_timestamps_from_crossattention(
    cross_attentions: List[np.ndarray],
    sequences: np.ndarray,
    alignment_heads: List[Tuple[int, int]],
    median_filter_width: int = 7,
    time_precision: float = 0.02,
    num_frames: Optional[int] = None,
) -> np.ndarray:
    """
    Wrapper function to calculate timestamps from cross attention data.
    
    Args:
        cross_attentions: List of cross attention arrays
        sequences: Token sequence array
        alignment_heads: List of (layer, head) tuples for alignment
        median_filter_width: Width of median filter
        time_precision: Time precision in seconds
        num_frames: Optional number of frames to process
    
    Returns:
        Array of timestamps for each token
    """
    return extract_token_timestamps(
        cross_attentions=cross_attentions,
        sequences=sequences,
        alignment_heads=alignment_heads,
        median_filter_width=median_filter_width,
        time_precision=time_precision,
        num_frames=num_frames
    )