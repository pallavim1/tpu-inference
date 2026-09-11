#!/usr/bin/env bash
# Step 3: Setup Benchmark Runner Pod
set -euo pipefail

echo "=========================================================================="
echo " 3. Setting Up CPU Benchmark Runner Pod"
echo "=========================================================================="

if [ -f "deploy/cpu_benchmark_runner.yaml" ]; then
    kubectl apply -f deploy/cpu_benchmark_runner.yaml
elif [ -f "panw-benchmark/cpu_benchmark_runner.yaml" ]; then
    kubectl apply -f panw-benchmark/cpu_benchmark_runner.yaml
elif [ -f "cpu_benchmark_runner.yaml" ]; then
    kubectl apply -f cpu_benchmark_runner.yaml
fi

kubectl wait --for=condition=ready pod/cpu-benchmark-runner --timeout=120s

kubectl exec cpu-benchmark-runner -- bash -c "
  apt-get update && apt-get install -y gnupg curl ca-certificates git python3-pip
  gpg -k 2>/dev/null || true
  gpg --no-default-keyring --keyring /usr/share/keyrings/k6-archive-keyring.gpg --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys C5AD17C747E3415A3642D7AAB9732172C82409E5
  echo 'deb [signed-by=/usr/share/keyrings/k6-archive-keyring.gpg] https://dl.k6.io/deb stable main' | tee /etc/apt/sources.list.d/k6.list
  apt-get update && apt-get install -y k6
  pip install --upgrade pip
  pip install openpyxl pandas
  mkdir -p /workspace/results /workspace/cpu_to_tpu_results
"

echo "✅ CPU Benchmark Runner Pod initialized with k6 and dependencies!"
