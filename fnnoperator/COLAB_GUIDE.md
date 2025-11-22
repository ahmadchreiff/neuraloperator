# Running FNNOperator on Google Colab

## Quick Start

1. **Open the notebook**: Upload `COLAB_SETUP.ipynb` to Google Colab
2. **Enable GPU**: Runtime → Change runtime type → GPU (T4 or better)
3. **Follow the cells**: Run each cell in order

## What You Need to Upload

### Required Python Files:
- `FNN.py`
- `FNNOperator.py`
- `FNNTrainer.py`
- `train_heat_operator.py`
- `inference_heat_operator.py` (optional, for testing)

### Data Files (or generate in Colab):
- `train.npz`
- `test.npz`
- `validate.npz`

## Steps Overview

1. **Install dependencies** - PyTorch with CUDA, numpy, matplotlib
2. **Check GPU** - Verify GPU is available
3. **Create structure** - Set up directories
4. **Upload code** - Upload your Python files
5. **Upload data** - Upload your .npz files OR generate data
6. **Run training** - Execute training with your parameters
7. **Download results** - Download trained model

## Example Training Commands

### Small model (32x32 grid):
```bash
python fnnoperator/train_heat_operator.py \
    --data-dir fnnoperator/data/32x32 \
    --dimension 2 \
    --predict-all-time-steps \
    --batch-size 8 \
    --gradient-accumulation-steps 4 \
    --use-amp \
    --n-epochs 50
```

### Large model (128x128 grid):
```bash
python fnnoperator/train_heat_operator.py \
    --data-dir fnnoperator/data/128x128 \
    --dimension 2 \
    --predict-all-time-steps \
    --batch-size 1 \
    --gradient-accumulation-steps 8 \
    --use-amp \
    --n-epochs 50
```

## Colab Advantages

- ✅ **Free GPU** (T4, sometimes V100 or A100)
- ✅ **No local setup** needed
- ✅ **Easy sharing** of notebooks
- ✅ **Automatic saving** to Google Drive (optional)

## Colab Limitations

- ⚠️ **Session timeout** - ~12 hours max (save checkpoints!)
- ⚠️ **Disk space** - ~80GB available (manage your data size)
- ⚠️ **RAM** - ~12-25GB (watch for OOM errors)

## Tips

1. **Save to Drive**: Mount Google Drive to persist files
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Use smaller data first**: Test with 32x32 before 128x128

3. **Monitor GPU memory**: 
   ```python
   !nvidia-smi
   ```

4. **Save checkpoints frequently**: Models are large, download them!

5. **Use mixed precision**: Always use `--use-amp` for memory savings

## Troubleshooting

### "CUDA out of memory"
- Reduce `--batch-size` (try 1 or 2)
- Increase `--gradient-accumulation-steps`
- Use `--use-amp`
- Reduce grid size or time steps

### "Module not found"
- Make sure you uploaded all .py files
- Check that files are in `fnnoperator/` directory
- Restart runtime after uploading files

### "File not found"
- Check paths are relative to `/content/`
- Verify data files are in correct subdirectory
- Use absolute paths: `/content/fnnoperator/data/...`

