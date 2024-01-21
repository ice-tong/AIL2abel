# Introduction

This is the code for predicting the variable names on the AIL IR. The code is based on [fairseq](https://github.com/facebookresearch/fairseq), and the [trex](https://github.com/CUMLSec/trex) also helps us a lot.

# Install

```bash
conda create -n ail2abel python=3.8
conda activate ail2abel
# install torch1.8.1 and the corresponding cuda version
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

pip install "fairseq==0.12.2" "torchmetrics==1.1.1" "numpy==1.23.3"
```

# Prepare datasets

Download the dataset from [ailabel_x64_O2_270w_100pp.tar.gz](https://drive.google.com/file/d/1MUrmRRvykx-jSCQLAqrvOAPXn3Nvproe/view?usp=sharing) and extract it to `datasets/ailabel_x64_O2_270w_100pp`.

```bash
mkdir datasets
# download the dataset to `datasets/ailabel_x64_O2_270w_100pp.tar.gz` and extract it
tar -zxvf datasets/ailabel_x64_O2_270w_100pp.tar.gz -C datasets/
```

Preprocess the dataset to the format that fairseq can use by running the following command:
```bash
# the first parameter is the path of the dataset, the second parameter is the path of the output, 
# and the third parameter is the number of processes used for preprocessing
bash preprocess_data.sh datasets/ailabel_x64_O2_270w_100pp datasets/data-bin/ailabel_x64_O2_270w_100pp 40
```

We also provide a small dataset for debugging which can be download from [ailabel_x64_O2_270w_10pp.tar.gz](https://drive.google.com/file/d/1CI-Tg3Tv0P_OejRX3oqPtlCxt26nFhoE/view?usp=sharing).


# Train

```bash
bash train.sh datasets/data-bin/ailabel_x64_O2_270w_100pp
```

# Validate

```bash
bash validate.sh checkpoints/ailabel_x64_O2_270w_100pp/checkpoint_best.pt datasets/data-bin/ailabel_x64_O2_270w_100pp
```

We provide the trained checkpoints for the dataset `ailabel_x64_O2_270w_100pp` mentioned above. You can download it from [ailabel_x64_O2_270w_100pp_checkpoint_best.pt](https://drive.google.com/file/d/1UUa1l_EOvajgh9hzzt-UdnO0SedZnNds/view?usp=sharing).
