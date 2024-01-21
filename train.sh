#!/usr/bin/env bash
set -xe


BIN_DATA_DIR=${1:-"datasets/data-bin/ailabel_x64_O2_270w_100pp"}
EXP_NAME=${2:-`basename $BIN_DATA_DIR`}

CHECKPOINT_DIR=checkpoints
LOG_DIR=logs

EXP_CHECKPOINT_DIR=$CHECKPOINT_DIR/$EXP_NAME
LOG_FILE=$LOG_DIR/$EXP_NAME/`date +%Y%m%d_%H_%M_%S`-train.log

mkdir -p $EXP_CHECKPOINT_DIR
mkdir -p $LOG_DIR/$EXP_NAME


TOTAL_UPDATES=500000  # Total number of training steps
WARMUP_UPDATES=10000  # Warmup the learning rate over this many updates
PEAK_LR=1e-4          # Peak learning rate, adjust as needed, official suggested: 1e-4
TOKENS_PER_SAMPLE=512 # Max sequence length
MAX_POSITIONS=512     # Num. positional embeddings (usually same as above)
BATCH_SIZE=28      # Number of sequences per batch (batch size)
UPDATE_FREQ=8         # Increase the batch size 32x
ENCODER_EMB_DIM=768
ENCODER_LAYERS=6
ENCODER_ATTENTION_HEADS=8

CUDA_VISIBLE_DEVICES=0 fairseq-train \
  $BIN_DATA_DIR \
  --user-dir ./ail2abel \
  --task ail2abel --criterion ail2abel --reset-optimizer --arch ail2abel \
  --sample-break-mode eos --tokens-per-sample $TOKENS_PER_SAMPLE \
  --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
  --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES \
  --total-num-update $TOTAL_UPDATES \
  --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
  --batch-size $BATCH_SIZE --update-freq $UPDATE_FREQ \
  --max-update $TOTAL_UPDATES --log-format json --log-interval 10 \
  --no-epoch-checkpoints --save-dir $EXP_CHECKPOINT_DIR/ \
  --encoder-layers $ENCODER_LAYERS --encoder-embed-dim $ENCODER_EMB_DIM \
  --encoder-attention-heads $ENCODER_ATTENTION_HEADS \
  --random-token-prob 0.2 --mask-prob 0.2 \
  --memory-efficient-fp16 --batch-size-valid 20 \
  --skip-invalid-size-inputs-valid-test \
  --ddp-backend no_c10d \
  --restore-file $EXP_CHECKPOINT_DIR/checkpoint_last.pt \
  --log-file $LOG_FILE
