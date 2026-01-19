

CUDA_LAUNCH_BLOCKING=1 python src/main.py \
  --device cuda:1 \
  --epochs 2000 \
  --model-iters 20 \
  --atom-hidden 64 \
  --batch-size 32 \
  --tnorm godel \
  --support-agg godel \
  --rnn-type gru \
  --loss-type bce \
  --train-instances 128\
  --gating-mode atom \
  --atoms-range 5 10\
  --rules-range 10 25 \
  --atom-gates 2 \
  --min-stable-models 1\
  --logical-dimension 1\
  --train-dir /home/moriyama/ASP/benchmarks/random_data/lean_splits/train_medium\
  --eval-dir /home/moriyama/ASP/benchmarks/random_data/lean_splits/val_medium\
  --use-wfs-bounds\
  # --lr 1e-1 \
  # --train-dir /home/moriyama/ASP/benchmarks/GraphColouring/generated_graphs\
  # --eval-dir /home/moriyama/ASP/benchmarks/GraphColouring/generated_graphs_eval\
  # --use-gnn-backbone \
  # --train-handcrafted-dir /home/moriyama/ASP/GNN-ASP/analysis_cases\
  # --eval-handcrafted-dir /home/moriyama/ASP/GNN-ASP/analysis_cases\
  # --lr 1e-2 \
  # --gnn-init \
  # --gnn-init-iters 10 \
  # --allow-facts \
  # --positional-init \
  # --allow-constraints \

