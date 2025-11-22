# FNNOperator Training Guide

Quick reference for training FNNOperator on heat equation data.

**Note:** All commands should be run from the project root directory. Paths are automatically resolved relative to the project root.

---

## 1. Generate Data (32×32)

```bash
python neuraloperator/heat_eq_data/data_gen/generate_heat_data.py --dimension 2 --nx 32 --ny 32 --nt 64 --dataset-size 256 --output-dir fnnoperator/data/32x32
```

**Other sizes:**
- 16×16: `--nx 16 --ny 16 --output-dir fnnoperator/data/16x16`
- 64×64: `--nx 64 --ny 64 --output-dir fnnoperator/data/64x64`

---

## 2. Train Model

```bash
python fnnoperator/train_heat_operator.py --data-dir fnnoperator/data/32x32 --dimension 2
```

**Predict all time steps:**
```bash
python fnnoperator/train_heat_operator.py --data-dir fnnoperator/data/32x32 --dimension 2 --predict-all-time-steps
```

**Custom training:**
```bash
python fnnoperator/train_heat_operator.py --data-dir fnnoperator/data/32x32 --dimension 2 --n-epochs 200 --batch-size 64 --learning-rate 5e-4 --width 512 --run-name "experiment_v1"
```

---

## 3. Run Inference

```bash
python fnnoperator/inference_heat_operator.py --checkpoint fnnoperator/outputs/heat_eq_2d_20241215_143022/checkpoints/final_model.pt --data-dir fnnoperator/data/32x32
```

**Options:**
- `--n-samples 10`: Number of samples to visualize (default: 10)
- `--n-time-steps 6`: Number of time steps to show in temporal evolution (default: 6)
- `--output-dir`: Custom output directory (default: checkpoint_dir/../inference)

---

## Output Location

```
fnnoperator/outputs/{run_name}/
├── checkpoints/final_model.pt
├── logs/training_history.json
├── predictions/test_predictions.npy
└── inference/
    ├── visualizations/ (heatmap comparisons)
    ├── predictions.npy
    └── metrics.json
```

---

## Model Sizes

| Grid | Points | Parameters |
|------|--------|------------|
| 16×16 | 256 | ~0.20M |
| 32×32 | 1,024 | ~0.59M |
| 64×64 | 4,096 | ~2.16M |
| 64×128 | 8,192 | ~4.26M |
