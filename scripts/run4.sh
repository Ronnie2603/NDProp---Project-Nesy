#!/usr/bin/env bash
set -euo pipefail

# Training run configured for the decision-based semantics controller.
export CUDA_LAUNCH_BLOCKING=1
difficulty=${2:-medium}
cuda=${1:-0}

python src/main.py \
  --device cuda:$cuda \
  --epochs 1000 \
  --model-iters 10 \
  --logical-dimension 32 \
  --tnorm godel \
  --tnorm-wfs godel \
  --batch-size 64 \
  --min-stable-models 1 \
  --train-dir /home/moriyama/ASP/benchmarks/random_klp/3-lp_1sm/train_$difficulty \
  --eval-dir /home/moriyama/ASP/benchmarks/random_klp/3-lp_1sm/val_$difficulty\
  --atom-hidden 32 \
  --lr 0.001 \

# Additional options you might toggle:
# --convergence-tol 1e-4           # adjust early-stop tolerance for tau convergence
# --disable-convergence            # run full model-iters regardless of stability
# --lr 1e-3                        # change learning rate
# --atoms-range 3 8                # sample smaller programs
# --rules-range 6 18               # adjust rule count range
# --train-instances 128            # limit number of training instances per epoch
