# Jina Embeddings v2 on Google Cloud TPU v5e (GKE)

This directory contains GKE manifests, automated cluster provisioning scripts, and benchmark tooling for serving and testing **Jina Embeddings v2 Small** (`jinaai/jina-embeddings-v2-small-en`) on **Google Cloud TPU v5e** using **vLLM-TPU** on Google Kubernetes Engine (GKE).

---

## Directory Structure

```
.
├── README.md                           # This guide
└── gke/
    ├── cluster_setup.sh                # End-to-end VPC, GKE cluster & TPU v5e node pool provisioning
    └── jina_v5e_deployment.yaml        # GKE Deployment, ConfigMap, Service & Secret manifests
```

---

## Quick Start on GKE

### 1. Provision GKE Cluster and TPU v5e Node Pool
```bash
cd gke/
chmod +x cluster_setup.sh
./cluster_setup.sh
```

### 2. Verify Pod Health
```bash
kubectl get pods -l app=jina-embeddings-v2 -w
```
Once `Running`, test the health check:
```bash
kubectl exec -it deployment/jina-embeddings-v2-tpu -- curl http://127.0.0.1:8000/health
```

### 3. Run PANW k6 Benchmark
See [`../../benchmarks/panw_k6/`](../../benchmarks/panw_k6/) for the full k6 test suite and result analyzer.

For detailed performance numbers and comparison against NVIDIA L4 GPU baselines, see [PANW k6 GKE Benchmark Report](../../docs/models/panw_k6_gke_benchmark_report.md).
