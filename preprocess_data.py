import argparse
import subprocess

AIL_TOKEN_FIELD = 'ail_token'
AIL_TOKEN_LABEL_FIELD = 'ail_token_label'
INST_POS_FIELD = 'stmt_idxs'  # instruction positional embedding
OP_POS_FIELD = 'op_idxs'  # opcode/operand positional embedding

ALL_FIELDS = [AIL_TOKEN_FIELD, AIL_TOKEN_LABEL_FIELD, INST_POS_FIELD, OP_POS_FIELD]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('src_dir', type=str, default=None)
    parser.add_argument('dst_dir', type=str, default=None)
    parser.add_argument('--num_worker', type=str, default="60")
    return parser.parse_args()


def main():
    args = parse_args()
    src_dir = args.src_dir
    dst_dir = args.dst_dir
    num_worker = args.num_worker

    for field in ALL_FIELDS:
        retcode = subprocess.call(
            ['fairseq-preprocess', '--only-source', 
            '--srcdict', f'{src_dir}/vocabs/{field}/dict.txt', 
            '--trainpref', f'{src_dir}/train.{field}',
            '--validpref', f'{src_dir}/valid.{field}', 
            '--destdir', f'{dst_dir}/{field}',
            '--workers', num_worker]
        )
        if retcode != 0:
            raise Exception(f'Error in {field}')



if __name__ == '__main__':
    main()
