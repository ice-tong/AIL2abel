# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os

import numpy as np

from fairseq import utils
from fairseq.data import (
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import LegacyFairseqTask, register_task

# from config import AIL_TOKEN_FIELD, AIL_TOKEN_LABEL_FIELD, ALL_FIELDS
AIL_TOKEN_FIELD = 'ail_token'
AIL_TOKEN_LABEL_FIELD = 'ail_token_label'
INST_POS_FIELD = 'stmt_idxs'  # instruction positional embedding
OP_POS_FIELD = 'op_idxs'  # opcode/operand positional embedding

TRAIN_FIELDS = [AIL_TOKEN_FIELD, INST_POS_FIELD, OP_POS_FIELD]
ALL_FIELDS = [AIL_TOKEN_FIELD, AIL_TOKEN_LABEL_FIELD, INST_POS_FIELD, OP_POS_FIELD]


logger = logging.getLogger(__name__)


@register_task("ail2abel")
class AIL2abelTask(LegacyFairseqTask):
    """Adapted from masked_lm - Task for pretraining trex models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )
        parser.add_argument(
            "--sample-break-mode",
            default="complete",
            choices=["none", "complete", "complete_doc", "eos"],
            help='If omitted or "none", fills each sample with tokens-per-sample '
                 'tokens. If set to "complete", splits samples only at the end '
                 "of sentence, but may include multiple sentences per sample. "
                 '"complete_doc" is similar but respects doc boundaries. '
                 'If set to "eos", includes only one sentence per sample.',
        )
        parser.add_argument(
            "--tokens-per-sample",
            default=512,
            type=int,
            help="max number of total tokens over all segments "
                 "per sample for BERT dataset",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.1,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.1,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--freq-weighted-replacement",
            default=False,
            action="store_true",
            help="sample random replacement words based on word frequencies",
        )
        parser.add_argument(
            "--mask-multiple-length",
            default=1,
            type=int,
            help="repeat the mask indices multiple times",
        )
        parser.add_argument(
            "--mask-stdev", default=0.0, type=float, help="stdev of the mask length"
        )
        parser.add_argument(
            "--shorten-method",
            default="none",
            choices=["none", "truncate", "random_crop"],
            help="if not none, shorten sequences that exceed --tokens-per-sample",
        )
        parser.add_argument(
            "--shorten-data-split-list",
            default="",
            help="comma-separated list of dataset splits to apply shortening to, "
                 'e.g., "train,valid" (default: all dataset splits)',
        )

    def __init__(self, args, dictionary_dict):
        super().__init__(args)
        self.dictionary_dict = dictionary_dict
        self.seed = args.seed

        # add mask token
        self.mask_idx_dict = {}
        self.mask_idx_dict[AIL_TOKEN_FIELD] = dictionary_dict[AIL_TOKEN_FIELD].add_symbol("<mask>")

    @classmethod
    def setup_task(cls, args, **kwargs):
        # paths = utils.split_paths(args.data)

        paths = os.listdir(args.data)
        assert len(paths) > 0
        if len(paths) != len(ALL_FIELDS):
            logger.error(f'ERROR: invalid paths: {paths}, expected: {ALL_FIELDS}')
            raise ValueError()

        dictionary_dict = {}
        for field in ALL_FIELDS:
            dictionary_dict[field] = Dictionary.load(os.path.join(args.data, field, "dict.txt"))
            logger.info(f"{field} dictionary: {len(dictionary_dict[field])} types")
        return cls(args, dictionary_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0

        src_tokens = {}
        tgt_tokens = {}
        for field in ALL_FIELDS:
            split_path = os.path.join(self.args.data, field, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary[field],
                self.args.dataset_impl,
                combine=combine,
            )
            if dataset is None:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, split_path)
                )

            dataset = maybe_shorten_dataset(
                dataset,
                split,
                self.args.shorten_data_split_list,
                self.args.shorten_method,
                self.args.tokens_per_sample,
                self.args.seed,
            )

            # create continuous blocks of tokens
            dataset = TokenBlockDataset(
                dataset,
                dataset.sizes,
                self.args.tokens_per_sample - 1,  # one less for <s>
                pad=self.source_dictionary[field].pad(),
                eos=self.source_dictionary[field].eos(),
                break_mode=self.args.sample_break_mode,
            )
            logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

            # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
            dataset = PrependTokenDataset(dataset, self.source_dictionary[field].bos())

            if field == AIL_TOKEN_LABEL_FIELD:
                tgt_tokens[field] = RightPadDataset(
                    dataset,
                    pad_idx=self.source_dictionary[field].pad()
                )
            src_tokens[field] = RightPadDataset(
                    dataset,
                    pad_idx=self.source_dictionary[field].pad()
                )

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(dataset))

        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    "id": IdDataset(),
                    "net_input": {
                        "src_tokens": src_tokens,
                        "src_lengths": NumelDataset(dataset, reduce=False),
                    },
                    "target": {
                        "tgt_tokens": tgt_tokens,
                    },
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(dataset, reduce=True),
                },
                sizes=[dataset.sizes],
            ),
            sort_order=[
                shuffle,
                dataset.sizes,
            ],
        )

    @property
    def source_dictionary(self):
        return self.dictionary_dict

    @property
    def target_dictionary(self):
        return self.dictionary_dict
    
    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        model.register_maskvar_head(getattr(args, 'classification_head_name', 'maskvar'))

        return model
