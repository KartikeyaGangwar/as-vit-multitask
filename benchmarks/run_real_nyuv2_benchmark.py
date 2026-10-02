"""
================================================================================
Real-World NYUv2 Master Benchmark Suite (IEEE TPAMI Standard).
Trains and evaluates all 7 multi-task architectures on official 795 train / 654 test RGB-D frames.

Author: Kartikeya Gangwar (Department of Mathematics, University of Delhi)
Contact: kartikeyagangwar@proton.me
================================================================================
"""
import argparse
import copy
import gc
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.as_vit import ASViT
from benchmarks.models_baseline import SingleTaskViT, MonolithicMTViT, StaticMoEViT
from engine.loss_functions import MultiTaskLossModule
from engine.optimizers.pcgrad import PCGrad
from engine.optimizers.cagrad import CAGrad
from engine.gram_clash_profiler import VectorizedGramClashProfiler
from engine.subspace_amr_manager import LatentSubspaceAMRManager
from data.real_nyuv2_pipeline import RealNYUv2Dataset
from eval_multitask import evaluate_model, MultiTaskEvaluator


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def cleanup_gpu():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def get_gpu_mem_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0.0


class RealNYUv2BenchmarkRunner:
    """
    Executes systematic benchmark comparisons on Real-World NYUv2 dataset.
    """
    def __init__(
        self,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 4,
        img_size: int = 224,
        batch_size: int = 16,
        epochs: int = 10,
        lr: float = 3e-4,
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.img_size = img_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        set_seed(seed)
        cleanup_gpu()

        print("=" * 85, flush=True)
        print(f"[*] INITIALIZING REAL-WORLD NYUv2 BENCHMARK ON DEVICE: {self.device}", flush=True)
        print(f"[*] GPU Memory Allocated: {get_gpu_mem_mb():.2f} MB", flush=True)
        print("=" * 85, flush=True)

        # Real NYUv2 Datasets
        print("[*] Loading Real-World NYUv2 795 Train / 654 Test split...", flush=True)
        self.train_dataset = RealNYUv2Dataset(split="train", img_size=(img_size, img_size), download=False)
        self.val_dataset = RealNYUv2Dataset(split="val", img_size=(img_size, img_size), download=False)
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=0)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        print(f"[+] Loaded {len(self.train_dataset)} Real Train frames & {len(self.val_dataset)} Real Test frames!", flush=True)

        self.loss_module = MultiTaskLossModule().to(self.device)
        self.results: Dict[str, Dict[str, float]] = {}

    def run_single_task_baselines(self) -> Dict[str, float]:
        print("\n" + "=" * 80, flush=True)
        print("  [STAGE 1/7] REAL-WORLD SINGLE-TASK ViT BASELINES", flush=True)
        print("=" * 80, flush=True)
        st_metrics = {}

        tasks = ["segmentation", "depth", "surface_normals"]
        for task in tasks:
            cleanup_gpu()
            print(f"\n  --> Training Single-Task ViT ({task}) on Real NYUv2 [GPU Mem: {get_gpu_mem_mb():.2f} MB]...", flush=True)
            model = SingleTaskViT(
                task=task,
                img_size=self.img_size,
                embed_dim=self.embed_dim,
                depth=self.depth,
                num_heads=self.num_heads,
            ).to(self.device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

            for epoch in range(1, self.epochs + 1):
                model.train()
                epoch_loss = 0.0
                for batch in self.train_loader:
                    images = batch["image"].to(self.device)
                    target = batch[task].to(self.device)
                    optimizer.zero_grad()
                    pred = model(images)[task]
                    if task == "segmentation":
                        loss = self.loss_module.seg_loss(pred, target)
                    elif task == "depth":
                        loss = self.loss_module.depth_loss(pred, target)
                    else:
                        loss = self.loss_module.normal_loss(pred, target)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                scheduler.step()
                print(f"      [Epoch {epoch:2d}/{self.epochs:2d}] Loss: {epoch_loss/len(self.train_loader):.4f} | GPU Mem: {get_gpu_mem_mb():.2f} MB", flush=True)

            metrics = evaluate_model(model, self.val_loader, self.device)
            if task == "segmentation":
                st_metrics["seg_miou"] = metrics["seg_miou"]
                print(f"      [Real Test Result] mIoU: {metrics['seg_miou']:.2f}%", flush=True)
            elif task == "depth":
                st_metrics["depth_abs_rel"] = metrics["depth_abs_rel"]
                st_metrics["depth_rmse"] = metrics["depth_rmse"]
                print(f"      [Real Test Result] AbsRel: {metrics['depth_abs_rel']:.4f}, RMSE: {metrics['depth_rmse']:.4f}", flush=True)
            else:
                st_metrics["normals_mean_angle"] = metrics["normals_mean_angle"]
                st_metrics["normals_median_angle"] = metrics["normals_median_angle"]
                print(f"      [Real Test Result] Mean Angle: {metrics['normals_mean_angle']:.2f} deg, Median: {metrics['normals_median_angle']:.2f} deg", flush=True)

            del model, optimizer, scheduler
            cleanup_gpu()

        st_metrics["delta_m"] = 0.00
        self.results["Single-Task ViT Baselines"] = st_metrics
        return st_metrics

    def _train_real_mt_model(
        self,
        name: str,
        model: nn.Module,
        st_baselines: Dict[str, float],
        optimizer_type: str = "adamw",
    ) -> Dict[str, float]:
        cleanup_gpu()
        print(f"\n  --> Training {name} on Real NYUv2 [GPU Mem: {get_gpu_mem_mb():.2f} MB]...", flush=True)
        model = model.to(self.device)
        base_opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)

        if optimizer_type == "pcgrad":
            opt = PCGrad(base_opt)
        elif optimizer_type == "cagrad":
            opt = CAGrad(base_opt, c=0.5)
        else:
            opt = base_opt

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=self.epochs)

        for epoch in range(1, self.epochs + 1):
            model.train()
            epoch_loss = 0.0
            for batch in self.train_loader:
                images = batch["image"].to(self.device)
                targets = {k: v.to(self.device) for k, v in batch.items() if k != "image"}

                if optimizer_type in ["pcgrad", "cagrad"]:
                    opt.zero_grad()
                    preds = model(images)
                    ind_losses = self.loss_module.compute_individual_losses(preds, targets)
                    loss_list = [ind_losses["segmentation"], ind_losses["depth"], ind_losses["surface_normals"]]
                    if optimizer_type == "pcgrad":
                        opt.pc_backward(loss_list)
                    else:
                        opt.cagrad_backward(loss_list)
                    opt.step()
                    total_loss = sum(l.item() for l in loss_list)
                else:
                    opt.zero_grad()
                    preds = model(images)
                    loss, _ = self.loss_module(preds, targets)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()
                    total_loss = loss.item()

                epoch_loss += total_loss

            scheduler.step()
            val_metrics = evaluate_model(model, self.val_loader, self.device, st_baselines=st_baselines)
            print(
                f"      [Epoch {epoch:2d}/{self.epochs:2d}] Loss: {epoch_loss/len(self.train_loader):.4f} | "
                f"mIoU: {val_metrics['seg_miou']:.2f}% | Depth AbsRel: {val_metrics['depth_abs_rel']:.4f} | "
                f"Normals: {val_metrics['normals_mean_angle']:.2f} deg | Delta M: {val_metrics['delta_m']:+.2f}% | GPU Mem: {get_gpu_mem_mb():.2f} MB",
                flush=True
            )

        final_metrics = evaluate_model(model, self.val_loader, self.device, st_baselines=st_baselines)
        print(f"  [+] {name} Real Test Result: mIoU={final_metrics['seg_miou']:.2f}%, Depth AbsRel={final_metrics['depth_abs_rel']:.4f}, Normals Mean={final_metrics['normals_mean_angle']:.2f} deg | Delta M={final_metrics['delta_m']:+.2f}%", flush=True)
        self.results[name] = final_metrics

        del model, base_opt, opt, scheduler
        cleanup_gpu()
        return final_metrics

    def run_all(self) -> Tuple[pd.DataFrame, Dict]:
        start_time = time.time()

        # 1. Single-Task
        st_baselines = self.run_single_task_baselines()

        # 2. Monolithic MT-ViT
        print("\n" + "=" * 80, flush=True)
        print("  [STAGE 2/7] REAL MONOLITHIC MULTI-TASK ViT", flush=True)
        print("=" * 80, flush=True)
        mt_vit = MonolithicMTViT(img_size=self.img_size, embed_dim=self.embed_dim, depth=self.depth, num_heads=self.num_heads)
        self._train_real_mt_model("Standard Multi-Task ViT (MT-ViT)", mt_vit, st_baselines, "adamw")

        # 3. PCGrad
        print("\n" + "=" * 80, flush=True)
        print("  [STAGE 3/7] REAL PCGrad (NeurIPS 2020)", flush=True)
        print("=" * 80, flush=True)
        pcgrad_vit = MonolithicMTViT(img_size=self.img_size, embed_dim=self.embed_dim, depth=self.depth, num_heads=self.num_heads)
        self._train_real_mt_model("PCGrad (NeurIPS 2020)", pcgrad_vit, st_baselines, "pcgrad")

        # 4. CAGrad
        print("\n" + "=" * 80, flush=True)
        print("  [STAGE 4/7] REAL CAGrad (NeurIPS 2021)", flush=True)
        print("=" * 80, flush=True)
        cagrad_vit = MonolithicMTViT(img_size=self.img_size, embed_dim=self.embed_dim, depth=self.depth, num_heads=self.num_heads)
        self._train_real_mt_model("CAGrad (NeurIPS 2021)", cagrad_vit, st_baselines, "cagrad")

        # 5. Static MoE-ViT (E=4)
        print("\n" + "=" * 80, flush=True)
        print("  [STAGE 5/7] REAL STATIC MoE-ViT (E=4)", flush=True)
        print("=" * 80, flush=True)
        moe4_vit = StaticMoEViT(num_experts=4, top_k=2, img_size=self.img_size, embed_dim=self.embed_dim, depth=self.depth, num_heads=self.num_heads)
        self._train_real_mt_model("Static MoE-ViT (E=4)", moe4_vit, st_baselines, "adamw")

        # 6. Static MoE-ViT (E=8)
        print("\n" + "=" * 80, flush=True)
        print("  [STAGE 6/7] REAL STATIC MoE-ViT (E=8)", flush=True)
        print("=" * 80, flush=True)
        moe8_vit = StaticMoEViT(num_experts=8, top_k=2, img_size=self.img_size, embed_dim=self.embed_dim, depth=self.depth, num_heads=self.num_heads)
        self._train_real_mt_model("Static MoE-ViT (E=8)", moe8_vit, st_baselines, "adamw")

        # 7. Proposed AS-ViT (Real-World Two-Stage AMR)
        print("\n" + "=" * 80, flush=True)
        print("  [STAGE 7/7] REAL-WORLD PROPOSED AS-ViT (TWO-STAGE DISCOVER-AND-DEPLOY)", flush=True)
        print("=" * 80, flush=True)
        cleanup_gpu()
        from train_as_vit import TwoStageASViTTrainer
        as_trainer = TwoStageASViTTrainer(
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            img_size=self.img_size,
            batch_size=self.batch_size,
            lr=self.lr,
            device=self.device,
            seed=self.seed,
        )
        disc_epochs = max(2, self.epochs // 2)
        prod_epochs = self.epochs
        disc_info = as_trainer.run_stage1_discovery(self.train_loader, self.val_loader, disc_epochs=disc_epochs, profile_freq=1)
        as_model, as_metrics = as_trainer.run_stage2_production(disc_info, self.train_loader, self.val_loader, prod_epochs=prod_epochs)

        evaluator = MultiTaskEvaluator(single_task_baselines=st_baselines)
        as_metrics["delta_m"] = evaluator.compute_multitask_gain(as_metrics)
        self.results["Proposed AS-ViT (Ours)"] = as_metrics

        del as_trainer, as_model
        cleanup_gpu()

        elapsed = time.time() - start_time
        print(f"\n[+] Real-World NYUv2 Benchmark Suite Completed in {elapsed:.2f}s!", flush=True)

        rows = []
        for method, m in self.results.items():
            rows.append({
                "Method": method,
                "Segmentation (mIoU %)": f"{m.get('seg_miou', 0.0):.2f}",
                "Depth (Abs Rel)": f"{m.get('depth_abs_rel', 0.0):.4f}",
                "Depth (RMSE)": f"{m.get('depth_rmse', 0.0):.4f}",
                "Normals Mean Angle (deg)": f"{m.get('normals_mean_angle', 0.0):.2f}",
                "Normals Median Angle (deg)": f"{m.get('normals_median_angle', 0.0):.2f}",
                "Multi-Task Gain (Delta M %)": f"{m.get('delta_m', 0.0):+.2f}%",
            })
        df = pd.DataFrame(rows)

        os.makedirs("results/data", exist_ok=True)
        df.to_csv("results/data/real_table_nyuv2_results.csv", index=False)
        with open("results/data/real_nyuv2_benchmark_results.json", "w") as f:
            json.dump(self.results, f, indent=2)

        return df, self.results


def main():
    parser = argparse.ArgumentParser(description="Run Real-World NYUv2 Benchmarks")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    runner = RealNYUv2BenchmarkRunner(
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
    df, _ = runner.run_all()
    print("\n" + "=" * 85, flush=True)
    print("             REAL-WORLD NYUv2 BENCHMARK EVALUATION MATRIX (795 / 654)            ", flush=True)
    print("=" * 85, flush=True)
    print(df.to_string(index=False), flush=True)
    print("=" * 85, flush=True)


if __name__ == "__main__":
    main()
