import torch
import torch.nn.functional as F
import argparse
from typing import Optional
from flex_flashinfer import FlexFlashInfer, _create_dense_mask
from masks import get_mask_fn

def parse_args():
    parser = argparse.ArgumentParser(description='Test FlexFlashInfer Implementation')
    parser.add_argument('--q_len', type=int, default=1024, help='Query sequence length')
    parser.add_argument('--kv_len', type=int, default=1024, help='Key/Value sequence length')
    parser.add_argument('--num_qo_heads', type=int, default=32, help='Number of query heads')
    parser.add_argument('--num_kv_heads', type=int, default=32, help='Number of key/value heads')
    parser.add_argument('--head_dim', type=int, default=64, help='Head dimension')
    parser.add_argument('--block_size_row', type=int, default=1, help='Block size for rows')
    parser.add_argument('--block_size_col', type=int, default=64, help='Block size for columns')
    parser.add_argument('--mask_type', type=str, default='causal', choices=['causal', 'sliding_window', 'attamba'], help='Type of attention mask')
    parser.add_argument('--window_size', type=int, default=8, help='Window size for windowed attention')
    parser.add_argument('--chunk_size', type=int, default=8, help='Chunk size for attamba mask')
    return parser.parse_args()

def display_args(args):
    """Display arguments in a nicely formatted way."""
    def _format_section(title, items):
        width = 60
        print("\n" + "=" * width)
        print(f"{title:^{width}}")
        print("-" * width)
        for k, v in items:
            print(f"{k:>25}: {v:<30}")
    
    # Group arguments by category
    sequence_args = [
        ("Query Length", args.q_len),
        ("KV Length", args.kv_len),
    ]
    
    model_args = [
        ("Query Heads", args.num_qo_heads),
        ("KV Heads", args.num_kv_heads),
        ("Head Dimension", args.head_dim),
    ]
    
    block_args = [
        ("Block Size (Row)", args.block_size_row),
        ("Block Size (Col)", args.block_size_col),
    ]
    
    mask_args = [
        ("Mask Type", args.mask_type),
        ("Window Size", args.window_size if hasattr(args, 'window_size') else 'N/A'),
        ("Chunk Size", args.chunk_size if hasattr(args, 'chunk_size') else 'N/A'),
    ]
    
    print("\n" + "🚀 FlexFlashInfer Configuration 🚀".center(60))
    _format_section("Sequence Configuration", sequence_args)
    _format_section("Model Configuration", model_args)
    _format_section("Block Sparsity Configuration", block_args)
    _format_section("Mask Configuration", mask_args)
    print("=" * 60 + "\n")

def run_comparison(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dense_mask: torch.Tensor,
    flex_flash_wrapper: FlexFlashInfer,
    atol: float = 5e-3,
    rtol: float = 1e-3
) -> bool:
    """Run and compare SDPA and FlexFlashInfer implementations"""
    
    # Prepare inputs for SDPA (needs BHLD format)
    q_sdpa = q.transpose(0, 1).unsqueeze(0)  # [1, H, L, D]
    k_sdpa = k.transpose(0, 1).unsqueeze(0)  # [1, H, L, D]
    v_sdpa = v.transpose(0, 1).unsqueeze(0)  # [1, H, L, D]
    
    # Run SDPA
    with torch.no_grad():
        out_sdpa = F.scaled_dot_product_attention(
            q_sdpa, k_sdpa, v_sdpa,
            attn_mask=dense_mask
        )
    
    # Run FlexFlashInfer
    with torch.no_grad():
        out_flex = flex_flash_wrapper(q, k, v)
        # Convert to SDPA format for comparison
        out_flex = out_flex.transpose(0, 1).unsqueeze(0)
    
    # Compare results
    is_close = torch.allclose(out_sdpa, out_flex, atol=atol, rtol=rtol)
    
    if not is_close:
        max_diff = torch.max(torch.abs(out_sdpa - out_flex))
        print(f"Max difference: {max_diff:.6f}")
        
    return is_close


def main():
    args = parse_args()
    display_args(args)
    
    # Initialize tensors
    q = torch.randn((args.q_len, args.num_qo_heads, args.head_dim), 
                    dtype=torch.float16, device="cuda")
    k = torch.randn((args.kv_len, args.num_kv_heads, args.head_dim),
                    dtype=torch.float16, device="cuda")
    v = torch.randn((args.kv_len, args.num_kv_heads, args.head_dim),
                    dtype=torch.float16, device="cuda")
    
    # Create mask function
    mask_fn = get_mask_fn(args.mask_type, args.window_size)

    # Create dense mask for SDPA
    dense_mask = _create_dense_mask(
        mask_fn,
        args.q_len,
        args.kv_len,
        device=torch.device("cuda")
    )
    
    # Initialize FlexFlashInfer
    flex_flash = FlexFlashInfer(device="cuda")
    
    # Prepare FlexFlashInfer
    flex_flash.prepare(
        mask_fn=mask_fn,
        q_len=args.q_len,
        kv_len=args.kv_len,
        num_heads=args.num_qo_heads,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        block_size_row=args.block_size_row,
        block_size_col=args.block_size_col,
    )
    
    # Run comparison
    print("\nTesting Correctness...")
    is_correct = run_comparison(q, k, v, dense_mask, flex_flash)
    print(f"Test result: {'✅ PASSED' if is_correct else '❌ FAILED'}")

if __name__ == "__main__":
    main()