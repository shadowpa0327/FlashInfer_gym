import torch
from typing import Callable, Optional, Tuple, Union
import flashinfer


def _vmap_for_mask(fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
    """Applies vmap over query and key indices.
    
    Args:
        fn: Function that takes (q_idx, kv_idx) and returns a boolean mask
        
    Returns:
        Vectorized function that can handle batched inputs
    """
    # Vmap over key/value indices first
    fn = torch.vmap(fn, in_dims=(None, 0))
    # Then vmap over query indices
    fn = torch.vmap(fn, in_dims=(0, None))
    return fn

def _create_dense_mask(
    mask_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    q_len: int,
    kv_len: int,
    device: torch.device
) -> torch.Tensor:
    """Creates a dense attention mask using the provided mask function.
    
    Args:
        mask_fn: Function that takes (q_idx, kv_idx) and returns boolean mask
        q_len: Query sequence length
        kv_len: Key/value sequence length
        device: Device to create tensors on
        
    Returns:
        Dense boolean mask tensor of shape [q_len, kv_len]
    """
    # Generate indices for mask computation
    q_idx = torch.arange(q_len, device=device)
    kv_idx = torch.arange(kv_len, device=device)
    
    # Vectorize the mask function
    vmapped_mask_fn = _vmap_for_mask(mask_fn)
    
    # Compute dense mask
    mask = vmapped_mask_fn(q_idx, kv_idx)  # [q_len, kv_len]
    return mask

def _convert_to_bsr(
    mask: torch.Tensor,
    block_size_row: int,
    block_size_col: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert a dense mask to BSR format using optimized tensor operations.
    Adapted from FlexAttention's approach but simplified for 2D masks."""
    M, N = mask.shape
    assert M % block_size_row == 0, "M must be divisible by block_size_row"
    assert N % block_size_col == 0, "N must be divisible by block_size_col"
    
    # Reshape to blocks [M//R, R, N//C, C]
    mask_blocked = mask.reshape(
        M // block_size_row, block_size_row, 
        N // block_size_col, block_size_col
    )
    
    # Permute to get block-wise view [M//R, N//C, R, C]
    mask_blocked = mask_blocked.permute(0, 2, 1, 3)
    
    # Get per-block sparsity pattern [M//R, N//C]
    block_mask = mask_blocked.any(dim=(-2, -1))
    
    # Convert to BSR format
    MB = M // block_size_row
    
    # Get number of non-zeros per row and create indptr
    nnz_per_row = block_mask.sum(dim=-1)
    indptr = torch.zeros(MB + 1, dtype=torch.int32, device=mask.device)
    indptr[1:] = torch.cumsum(nnz_per_row, dim=0)
    
    # Get total number of non-zero blocks
    total_nnz = int(indptr[-1].item())
    
    if total_nnz == 0:
        return (
            indptr,
            torch.empty(0, dtype=torch.int32, device=mask.device),
            torch.empty(0, block_size_row, block_size_col, dtype=mask.dtype, device=mask.device)
        )
        
    # Get row and column indices of non-zero blocks
    nz_rows, nz_cols = torch.nonzero(block_mask, as_tuple=True)
    
    # Sort by row to ensure correct ordering
    sort_idx = torch.argsort(nz_rows)
    indices = nz_cols[sort_idx].to(torch.int32)
    
    # Extract block data using the sorted indices
    block_data = mask_blocked[nz_rows[sort_idx], nz_cols[sort_idx]]

    return indptr, indices, block_data

class FlexFlashInfer:
    """A flexible wrapper for FlashInfer that supports mask function definition similar to FlexAttention."""
    
    def __init__(
        self,
        workspace_buffer: Optional[torch.Tensor] = None,
        device: Union[str, torch.device] = "cuda"
    ):
        """Initialize FlexFlashInfer.
        
        Args:
            workspace_buffer: Optional workspace buffer for FlashInfer
            device: Device to place tensors on
        """
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device
        
        if workspace_buffer is None:
            # Allocate 128MB workspace by default
            workspace_buffer = torch.empty(
                128 * 1024 * 1024,
                dtype=torch.uint8,
                device=device
            )
        self.flash_wrapper = None  # Will be initialized in prepare()
        self.workspace_buffer = workspace_buffer
        
    def prepare(
        self,
        mask_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        num_heads: int,
        q_len: int,
        kv_len: int,
        block_size_row: int = 64,
        block_size_col: int = 64,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """Prepare the wrapper by converting mask function to BSR format.
        
        Args:
            mask_fn: Function that takes (q_idx, kv_idx) and returns boolean mask
            num_heads: Number of attention heads
            q_len: Query sequence length
            kv_len: Key/value sequence length
            block_size_row: Row block size for BSR format
            block_size_col: Column block size for BSR format
            num_kv_heads: Number of key/value heads (for GQA)
            head_dim: Head dimension (needed for FlashInfer initialization)
            dtype: Data type for computation
        """
        if num_kv_heads is None:
            num_kv_heads = num_heads
            
        if head_dim is None:
            raise ValueError("head_dim must be specified")
        
        # Cache sequence lengths
        self.cached_q_len = q_len
        self.cached_kv_len = kv_len

        # Create dense mask using vmapped function
        mask = _create_dense_mask(mask_fn, q_len, kv_len, self.device)
        # Convert to BSR format
        indptr, indices, block_mask = _convert_to_bsr(
            mask, 
            block_size_row,
            block_size_col
        )

        # Initialize FlashInfer wrapper
        self.flash_wrapper = flashinfer.BlockSparseAttentionWrapper(
            self.workspace_buffer
        )
        
        # Plan the computation
        self.flash_wrapper.plan(
            indptr=indptr,
            indices=indices,
            M=q_len,
            N=kv_len,
            R=block_size_row,
            C=block_size_col,
            num_qo_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            mask=block_mask,
            q_data_type=dtype
        )
        
    def forward(
        self,
        query: torch.Tensor,  # [seq_len_q, num_qo_heads, head_dim]
        key: torch.Tensor,    # [seq_len_kv, num_kv_heads, head_dim]
        value: torch.Tensor,  # [seq_len_kv, num_kv_heads, head_dim]
        return_lse: bool = False
    ):
        """Compute attention using FlashInfer.
        
        Args:
            query: Query tensor of shape [seq_len_q, num_qo_heads, head_dim]
            key: Key tensor of shape [seq_len_kv, num_kv_heads, head_dim]
            value: Value tensor of shape [seq_len_kv, num_kv_heads, head_dim]
            return_lse: Whether to return log sum exp
            
        Returns:
            output: Attention output tensor of shape [seq_len_q, num_qo_heads, head_dim]
            lse (optional): Log sum exp values of shape [seq_len_q, num_qo_heads]
            
        Raises:
            AssertionError: If input sequence lengths don't match cached values
            RuntimeError: If prepare() hasn't been called
        """
        if self.flash_wrapper is None:
            raise RuntimeError("Call prepare() before forward()")
        
        # Validate sequence lengths against cached values
        if query.size(0) != self.cached_q_len:
            raise AssertionError(
                f"Query sequence length mismatc with the one specificied in preparation: got {query.size(0)}, "
                f"expected {self.cached_q_len}"
            )

        if key.size(0) != self.cached_kv_len or value.size(0) != self.cached_kv_len:
            raise AssertionError(
                f"Key/Value sequence length mismatc with the one specificied in preparation: got {key.size(0)}, "
                f"expected {self.cached_kv_len}"
            )        
            
        # Run FlashInfer attention
        output = self.flash_wrapper.run(query, key, value, return_lse=return_lse)
        
        if return_lse:
            output, lse = output
            return output, lse
        
        return output

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)
    