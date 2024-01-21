set -xe

CHECKPOINT=${1:-"checkpoints/ailabel_x64_O2_270w_100pp/checkpoint_best.pt"}
BIN_DATA_DIR=${2:-"/root/autodl-tmp/data-bin/ailabel_x64_O2_270w_100pp"}
BATCH_SIZE=${3:-"20"}

fairseq-validate $BIN_DATA_DIR --path $CHECKPOINT \
    --user-dir ./ail2abel/ --task ail2abel \
    --batch-size $BATCH_SIZE --skip-invalid-size-inputs-valid-test
