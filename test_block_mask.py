import torch
import numpy as np
from scipy.sparse import bsr_matrix
from typing import Tuple
import time
from flex_flashinfer import _convert_to_bsr_optimized


def test_bsr_conversion():
    """Test the BSR conversion with various patterns."""
    
    def create_test_mask(size: int, pattern: str = 'causal') -> torch.Tensor:
        if pattern == 'causal':
            return torch.tril(torch.ones(size, size, dtype=torch.bool))
        elif pattern == 'checkerboard':
            mask = torch.zeros(size, size, dtype=torch.bool)
            mask[::2, ::2] = True
            mask[1::2, 1::2] = True
            return mask
        elif pattern == 'random':
            return torch.rand(size, size) > 0.5
        raise ValueError(f"Unknown pattern: {pattern}")
    
    def print_comparison(name: str, torch_val: torch.Tensor, scipy_val: np.ndarray):
        print(f"\n{name}:")
        print(f"Torch: {torch_val.cpu().numpy()}")
        print(f"Scipy: {scipy_val}")
        
    test_cases = [
        (6, 2, 'causal'),      # Small case for debugging
        (8, 2, 'checkerboard'),
        (128, 32, 'causal'),
        (256, 64, 'random'),
    ]
    
    print("\nTest Results:")
    print(f"{'Size':>8} {'Block':>8} {'Pattern':>12} {'Time(ms)':>10} {'Status':>10}")
    print("-" * 50)
    
    for size, block_size, pattern in test_cases:
        mask = create_test_mask(size, pattern)
        
        # Get scipy result
        mask_np = mask.cpu().numpy()
        bsr = bsr_matrix(mask_np, blocksize=(block_size, block_size))
        
        # Time and get our result
        torch.cuda.synchronize()
        start = time.perf_counter()
        indptr, indices, block_data = _convert_to_bsr_optimized(mask, block_size, block_size)
        torch.cuda.synchronize()
        opt_time = (time.perf_counter() - start) * 1000
        
        # Verify results
        try:
            np.testing.assert_array_equal(indptr.cpu().numpy(), bsr.indptr)
            np.testing.assert_array_equal(indices.cpu().numpy(), bsr.indices)
            np.testing.assert_array_equal(block_data.cpu().numpy(), bsr.data)
            status = "✓"
        except AssertionError as e:
            status = "✗"
            print(f"\nError in {pattern} {size}x{size} with block size {block_size}:")
            print_comparison("indptr", indptr, bsr.indptr)
            print_comparison("indices", indices, bsr.indices)
            print(f"block_data shapes - Torch: {block_data.shape}, Scipy: {bsr.data.shape}")
            
        print(f"{size:>8} {block_size:>8} {pattern:>12} {opt_time:>10.2f} {status:>10}")

if __name__ == "__main__":
    test_bsr_conversion()