# Install

```bash
conda create -n ail2abel python=3.8
conda activate ail2abel
# install torch1.8.1 and the corresponding cuda version
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install "fairseq==0.12.2" "torchmetrics==1.1.1" "numpy==1.23.3"
```

# Prepare dataset

```bashe
python3 preprocess_data.py dataset/ailabel_x64_O2_270w_100pp dataset/data-bin/ailabel_x64_O2_270w_100pp --num_worker 40
```

# Train

```bash
bash train.sh dataset/data-bin/ailabel_x64_O2_270w_100pp
```

# Validate

```bash
bash validate.sh checkpoints/ailabel_x64_O2_270w_100pp/checkpoint_best.pt dataset/data-bin/ailabel_x64_O2_270w_100pp
```
