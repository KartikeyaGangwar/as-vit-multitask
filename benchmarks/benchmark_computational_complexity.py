"""
================================================================================
Comprehensive Computational Complexity & Efficiency Profiler (IEEE TPAMI Standard)
Profiles:
  1. Total Parameters (M)
  2. Active Parameters per Token (M)
  3. Floating Point Operations (GFLOPs / MACs)
  4. Peak GPU VRAM Memory (MB / GB) during Forward + Backward Pass
  5. Inference Latency (ms / image) with CUDA Event Synchronization
  6. Inference Throughput (Frames Per Second / FPS)

Evaluated across all 7 benchmark architectures:
  - Single-Task ViT Baselines (Cumulative 3 models)
  - Standard Multi-Task ViT (MT-ViT)
  - PCGrad (NeurIPS 2020)
  - CAGrad (NeurIPS 2021)
  - Static MoE-ViT (E=4, Top-2)
  - Static MoE-ViT (E=8, Top-2)
  - Proposed AS-ViT (Ours, Autonomous AMR Discovery)

Hardware Platform:
  - Dedicated GPU: NVIDIA GeForce GTX 1650 (4.0 GB)
  - PyTorch 2.6.0+cu124, CUDA 12.4
================================================================================
"""
import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.as_vit import ASViT
from benchmarks.models_baseline import SingleTaskViT, MonolithicMTViT, StaticMoEViT


def count_parameters(model: nn.Module) -> float:
    """Returns total parameter count in Millions (M)."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def compute_flops(model: nn.Module, input_tensor: torch.Tensor) -> float:
    """
    Computes approximate forward GFLOPs for ViT architectures.
    Standard analytical formulation for PatchEmbed + TransformerBlocks + TaskHeads.
    """
    B, C, H, W = input_tensor.shape
    P = 16
    M = (H // P) * (W // P) # Sequence length = 196
    
    total_flops = 0.0
    # Patch embedding: B * M * (3 * P * P * D)
    D = getattr(model, "embed_dim", 192)
    total_flops += 2.0 * B * M * (3 * P * P * D)
    
    # Check model type
    if isinstance(model, SingleTaskViT):
        depth = model.depth
        for _ in range(depth):
            # Attention: QKV proj (3 * 2 * B * M * D^2), QK^T (2 * B * H_heads * M * M * d_k), Softmax, Attn*V, Out proj
            total_flops += 2.0 * B * M * (3 * D * D) + 2.0 * B * M * M * D + 2.0 * B * M * (D * D)
            # FFN: 2 * B * M * (2 * D * 4D)
            total_flops += 2.0 * B * M * (2 * D * int(4 * D))
        # Head:
        total_flops += 2.0 * B * H * W * D * 13
        
    elif isinstance(model, MonolithicMTViT):
        depth = model.depth
        for _ in range(depth):
            total_flops += 2.0 * B * M * (3 * D * D) + 2.0 * B * M * M * D + 2.0 * B * M * (D * D)
            total_flops += 2.0 * B * M * (2 * D * int(4 * D))
        # 3 Heads:
        total_flops += 2.0 * B * H * W * D * (13 + 1 + 3)
        
    elif isinstance(model, StaticMoEViT):
        depth = model.depth
        top_k = 2
        for _ in range(depth):
            total_flops += 2.0 * B * M * (3 * D * D) + 2.0 * B * M * M * D + 2.0 * B * M * (D * D)
            # Router:
            total_flops += 2.0 * B * M * D * model.num_experts
            # Active top-k experts:
            total_flops += top_k * 2.0 * B * M * (2 * D * int(4 * D))
        total_flops += 2.0 * B * H * W * D * (13 + 1 + 3)
        
    elif isinstance(model, ASViT):
        depth = len(model.blocks)
        for blk in model.blocks:
            total_flops += 2.0 * B * M * (3 * D * D) + 2.0 * B * M * M * D + 2.0 * B * M * (D * D)
            # PoU Router (distance to N centroids):
            total_flops += 2.0 * B * M * D * blk.num_subspaces
            # Subspaces:
            total_flops += blk.num_subspaces * 2.0 * B * M * (2 * D * int(4 * D))
        total_flops += 2.0 * B * H * W * D * (13 + 1 + 3)

    return total_flops / 1e9 # GFLOPs


def profile_model_efficiency(
    name: str,
    create_fn,
    device: torch.device,
    batch_size: int = 1,
    num_warmup: int = 20,
    num_eval: int = 100,
) -> dict:
    print(f"[*] Profiling computational efficiency for: {name}...")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    
    # 1. Instantiate model
    model = create_fn().to(device)
    model.eval()
    
    # 2. Count parameters
    if name == "Single-Task ViT Baselines":
        # Cumulative for all 3 tasks
        m_seg = SingleTaskViT(task="segmentation", embed_dim=192, depth=4).to(device)
        m_dep = SingleTaskViT(task="depth", embed_dim=192, depth=4).to(device)
        m_nor = SingleTaskViT(task="surface_normals", embed_dim=192, depth=4).to(device)
        total_params = count_parameters(m_seg) + count_parameters(m_dep) + count_parameters(m_nor)
        active_params = total_params
        flops = (compute_flops(m_seg, torch.zeros(1, 3, 224, 224).to(device)) +
                 compute_flops(m_dep, torch.zeros(1, 3, 224, 224).to(device)) +
                 compute_flops(m_nor, torch.zeros(1, 3, 224, 224).to(device)))
        del m_seg, m_dep, m_nor
    elif "Static MoE-ViT" in name:
        total_params = count_parameters(model)
        # Active params: Backbone + Router + Top-2 Experts + Heads
        # In MoE with top-2, active is Backbone + 2 experts + Heads
        num_exp = model.num_experts
        expert_params = sum(p.numel() for blk in model.blocks for exp in blk.experts for p in exp.parameters()) / (num_exp * 1e6)
        active_params = total_params - (num_exp - 2) * expert_params
        flops = compute_flops(model, torch.zeros(1, 3, 224, 224).to(device))
    elif "AS-ViT" in name:
        total_params = count_parameters(model)
        active_params = total_params # All active subspaces modulate continuously via PoU
        flops = compute_flops(model, torch.zeros(1, 3, 224, 224).to(device))
    else:
        total_params = count_parameters(model)
        active_params = total_params
        flops = compute_flops(model, torch.zeros(1, 3, 224, 224).to(device))

    # 3. Peak VRAM during training pass (forward + backward)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x_train = torch.randn(4, 3, 224, 224, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    
    out = model(x_train)
    if isinstance(out, tuple):
        out = out[0]
    if isinstance(out, dict):
        loss = sum(v.mean() for v in out.values() if isinstance(v, torch.Tensor))
    else:
        loss = out.mean()
        
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    del x_train, out, loss, optimizer
    torch.cuda.empty_cache()

    # 4. Latency & Throughput (batch_size = 1)
    model.eval()
    x_test = torch.randn(batch_size, 3, 224, 224, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(x_test)
    torch.cuda.synchronize(device)

    # Timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    timings = []
    with torch.no_grad():
        for _ in range(num_eval):
            start_event.record()
            _ = model(x_test)
            end_event.record()
            torch.cuda.synchronize(device)
            timings.append(start_event.elapsed_time(end_event))
            
    latency_ms = float(np.median(timings))
    throughput_fps = float(1000.0 / latency_ms * batch_size)

    del model, x_test
    torch.cuda.empty_cache()

    result = {
        "method": name,
        "total_params_m": round(total_params, 2),
        "active_params_m": round(active_params, 2),
        "gflops": round(flops, 2),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "peak_vram_gb": round(peak_vram_mb / 1024.0, 2),
        "latency_ms": round(latency_ms, 2),
        "fps": round(throughput_fps, 1),
    }
    print(f"    Total Params: {result['total_params_m']}M | Active: {result['active_params_m']}M | "
          f"GFLOPs: {result['gflops']} | Peak VRAM: {result['peak_vram_mb']}MB | "
          f"Latency: {result['latency_ms']}ms | FPS: {result['fps']}")
    return result


def run_full_complexity_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting Hardware Profiling on Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    benchmarks = [
        ("Single-Task ViT Baselines", lambda: SingleTaskViT(task="segmentation", embed_dim=192, depth=4)),
        ("Standard Multi-Task ViT (MT-ViT)", lambda: MonolithicMTViT(embed_dim=192, depth=4)),
        ("PCGrad (NeurIPS 2020)", lambda: MonolithicMTViT(embed_dim=192, depth=4)),
        ("CAGrad (NeurIPS 2021)", lambda: MonolithicMTViT(embed_dim=192, depth=4)),
        ("Static MoE-ViT (E=4)", lambda: StaticMoEViT(embed_dim=192, depth=4, num_experts=4, top_k=2)),
        ("Static MoE-ViT (E=8)", lambda: StaticMoEViT(embed_dim=192, depth=4, num_experts=8, top_k=2)),
        ("Proposed AS-ViT (Ours)", lambda: ASViT(embed_dim=192, depth=4, num_heads=4, initial_subspaces_per_block=[1, 2, 3, 2])),
    ]
    
    results = []
    for name, fn in benchmarks:
        res = profile_model_efficiency(name, fn, device)
        results.append(res)
        
    os.makedirs("results/data", exist_ok=True)
    out_json = "results/data/computational_complexity_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n[+] Computational Complexity Benchmark completed! Saved to: {out_json}")
    return results


if __name__ == "__main__":
    run_full_complexity_benchmark()
