# Cloud TPU v7x (Ironwood) — Model Validation & Serving Guides

This directory contains end-to-end architecture blueprints, Google Kubernetes Engine (GKE) provisioning commands, TorchTPU serving configurations, and systematic validation runbooks for frontier open-weight models on **Google Cloud TPU v7x (Ironwood)**.

## Validated & Day-0 Model Guides

| Model | Architecture & Context | Recommended TPU Stack | Reference Topology | Guide |
| :--- | :--- | :--- | :--- | :--- |
| **[Kimi K3.1 Pro](Kimi-K3.1-Pro/README.md)** (`moonshotai/Kimi-K3.1-Pro`) | 2.8T Total / 104B Active MoE (69 KDA + 24 Gated-MLA, MoonViT-V2 Multimodal, MXFP4) · **1M Context** | **TorchTPU** + **`SGLang-TorchTPU`** (`SGLANG_PLATFORM=tpu_oot`, Pallas KDA/MLA kernels, `raiden` HiCache) + **`llm-d`** (`LeaderWorkerSet` on GKE Standard) + **Managed Lustre** | **1 Replica**: `2x2x4` (4 hosts, 16 chips, 32 TensorCores, `TP=32, EP=8`)<br>**Day-0 1P1D Pool**: `2 × 2x2x4` (8 hosts, 32 chips, 64 TensorCores) | **[Validation & Deployment Runbook](Kimi-K3.1-Pro/README.md)** |

## TPU v7x (Ironwood) Hardware Quick Reference

Each physical **TPU v7x** package is a dual-TensorCore chip exposing **2 logical PyTorch (`torch_tpu`) devices** while Kubernetes (`google.com/tpu`) schedules by **physical chip count**:

| Machine Type | Slice Topology | Physical Chips / Host (`google.com/tpu`) | Logical `torch_tpu` Devices / Host | HBM3e Capacity | HBM3e Bandwidth |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tpu7x-standard-1t` | `1x1x1` (Single-Chip) | `1` | `2` (96 GiB / core) | 192 GiB / chip | 7,380 GB/s / chip |
| `tpu7x-standard-4t` | `2x2x1` (Single-Host) | `4` | `8` | 768 GiB / host | 29.5 TB/s / host |
| `4 × tpu7x-standard-4t` | `2x2x4` (Multi-Host Slice) | `16` (`4` × 4 pods) | `32` | 3,072 GiB (3 TiB) / slice | 118.1 TB/s / slice |
