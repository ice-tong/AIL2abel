#!/bin/bash

src_dir=${1:-"datasets/ailabel_x64_O2_270w_10pp"}
dst_dir=${2:-"datasets/data-bin/ailabel_x64_O2_270w_10pp"}
num_worker=${3:-"20"}

mkdir -p ${dst_dir}

for field in "ail_token" "ail_token_label" "stmt_idxs" "op_idxs"; do
    fairseq-preprocess --only-source \
        --srcdict ${src_dir}/vocabs/${field}/dict.txt \
        --trainpref ${src_dir}/train.${field} \
        --validpref ${src_dir}/valid.${field} \
        --destdir ${dst_dir}/${field} \
        --workers ${num_worker}
done
