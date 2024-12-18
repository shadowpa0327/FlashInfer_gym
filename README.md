# FlashInfer_gym

This repo provides a FlexAttention-style interface for [FlashInfer](https://github.com/flashinfer-ai/flashinfer), enabling easy definition of custom attention patterns with high-performance sparse attention computation.

## Todo
- [ ] Add performance comparison with FlexAttention
- [ ] Unit test
  - [x] Correceness test
  - [x] Mask Creation Test
  - [ ] Refactor unit test
- [ ] Support `score_mod`

## Features

- Easy-to-use interface similar to PyTorch's [FlexAttention](https://pytorch.org/blog/flexattention/)
- Support for various attention patterns (causal, sliding window, attamba)
- High-performance sparse attention computation powered by FlashInfer
- Simple mask function definition interface

## Installation

```bash
# Create conda environment
conda create -n flashinfer_gym python=3.10 
conda activate flashinfer_gym

# Install other dependencies
pip install -r requirements.txt

# Add the FlashInfer wheel index
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

## Quick Start

Here's a minimal example using causal attention:

```python
import torch
from flex_flash_infer import FlexFlashInfer, get_mask_fn

# Initialize the wrapper
flex_flash = FlexFlashInfer(device="cuda")

# Prepare with causal mask
flex_flash.prepare(
    mask_fn=get_mask_fn("causal"),
    seq_len_q=1024,        # Query sequence length
    seq_len_kv=1024,       # Key/Value sequence length
    num_qo_heads=8,        # Number of query heads
    num_kv_heads=8,        # Number of key/value heads
    head_dim=64,          # Head dimension
    block_size_row=64,    # Block size for sparse computation
    block_size_col=64,
    dtype=torch.float16
)

# Create input tensors
q = torch.randn((1024, 8, 64), dtype=torch.float16, device="cuda")
k = torch.randn((1024, 8, 64), dtype=torch.float16, device="cuda")
v = torch.randn((1024, 8, 64), dtype=torch.float16, device="cuda")

# Run attention
output = flex_flash(q, k, v)
```

## Supported Mask Types

You can easily create different attention patterns using `get_mask_fn`:

```python
# Causal mask
mask_fn = get_mask_fn("causal")

# Sliding window with causality
mask_fn = get_mask_fn("sliding_window", window_size=256)

# Attamba mask (chunk-based attention)
mask_fn = get_mask_fn("attamba", window_size=256, chunk_size=128)
```

Or define your own custom mask function:

```python
def custom_mask_fn(q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
    # Define your custom attention pattern
    return (q_idx >= kv_idx) & (q_idx - kv_idx <= 256)

flex_flash.prepare(mask_fn=custom_mask_fn, ...)
```

## Testing

To verify the correceness, we provide the following scripts for comparing the FlashInfer with Pytorch SDPA with customized mask:

```bash
python test_flex_flash.py --q_len 1024 --kv_len 1024 --num_qo_heads 8 --mask_type causal
```

## References

- [FlashInfer GitHub](https://github.com/flashinfer-ai/flashinfer)
- [FlexAttention Blog Post](https://pytorch.org/blog/flexattention/)
