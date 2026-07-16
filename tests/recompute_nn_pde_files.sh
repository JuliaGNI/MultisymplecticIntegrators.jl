#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

cd "$(dirname "$0")/.."

for file in nnint_fully_0907_sinegordon/*.jld2; do
  [[ -e "$file" ]] || continue
  echo "Processing $file"
  julia --project=. recompute_nn_pde_hamiltonian.jl --apply "$file"
done
