from collections import Counter
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.text import CharErrorRate

from fairseq import metrics, utils
from fairseq.data import Dictionary
from fairseq.criterions import FairseqCriterion, register_criterion

# from config import AIL_TOKEN_FIELD, AIL_TOKEN_LABEL_FIELD
AIL_TOKEN_FIELD = 'ail_token'
AIL_TOKEN_LABEL_FIELD = 'ail_token_label'
INST_POS_FIELD = 'stmt_idxs'  # instruction positional embedding
OP_POS_FIELD = 'op_idxs'  # opcode/operand positional embedding

TRAIN_FIELDS = [AIL_TOKEN_FIELD, INST_POS_FIELD, OP_POS_FIELD]
ALL_FIELDS = [AIL_TOKEN_FIELD, AIL_TOKEN_LABEL_FIELD, INST_POS_FIELD, OP_POS_FIELD]


class FocalLoss(nn.CrossEntropyLoss):
    """Focal loss for classification tasks on imbalanced datasets"""

    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='sum'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss) if self.reduction == 'mean' else \
            torch.sum(loss) if self.reduction == 'sum' else loss


class _FairseqCriterion(FairseqCriterion):
    def __init__(self, task):
        super(FairseqCriterion, self).__init__()
        self.task = task

        if hasattr(task, "target_dictionary"):
            if type(task.target_dictionary) is Dictionary:
                tgt_dict = task.target_dictionary
                self.padding_idx = tgt_dict.pad() if tgt_dict is not None else -100
            elif type(task.target_dictionary) is dict:
                self.padding_idx_dict = {}
                for field in TRAIN_FIELDS:
                    self.padding_idx_dict[field] = task.target_dictionary[field].pad()


@register_criterion("ail2abel")
class AIL2abelCriterion(_FairseqCriterion):
    """Implementation for the loss used in masked language model (MLM) training."""

    def __init__(self, task, tpu=False):
        super().__init__(task)
        self.focal_loss = FocalLoss(
            gamma=2, ignore_index=self.padding_idx_dict[AIL_TOKEN_FIELD], reduction="mean")

        self.class_tp = [0, ] * len(self.task.source_dictionary[AIL_TOKEN_LABEL_FIELD].indices)
        self.class_fp = [0, ] * len(self.task.source_dictionary[AIL_TOKEN_LABEL_FIELD].indices)
        # self.class_tn = [0, ] * len(self.task.source_dictionary[AIL_TOKEN_LABEL_FIELD].indices)
        self.class_fn = [0, ] * len(self.task.source_dictionary[AIL_TOKEN_LABEL_FIELD].indices)
        print("vocab len:", len(self.class_fn))
        
        self.gt_and_preds = []

        self.CER = CharErrorRate()

    def _post_process(self, pred_labels, target_labels):
        # post process: vote the predicted name if same var
        same_var_idxs = {}
        for idx, target in enumerate(target_labels):
            var_name_tuple = tuple(target.tolist())
            if var_name_tuple in same_var_idxs:
                same_var_idxs[var_name_tuple].append(idx)
            else:
                same_var_idxs[var_name_tuple] = [idx]

        for var_name_tuple, idxs in same_var_idxs.items():
            if len(idxs) <= 1:
                continue
            vote_counter = Counter()
            for idx in idxs:
                vote_counter[tuple(pred_labels[idx].tolist())] += 1
            for idx in idxs:
                pred_labels[idx] = vote_counter.most_common(1)[0][0]
        # post process end
        return pred_labels, target_labels
    
    def compute_precision_recall_accuracy_classwise(self, pred_logist, targets):
        pred_labels = pred_logist.argmax(1)
        # reshape
        pred_labels = pred_labels.reshape(-1, 4).detach().cpu().numpy()
        targets = targets.reshape(-1, 4).detach().cpu().numpy()
        
        # post process: vote the predicted name if same var
        pred_labels, _ = self._post_process(pred_labels, targets)
        for pred, target in zip(pred_labels.reshape(-1), targets.reshape(-1)):
            if pred == target:
                self.class_tp[target] += 1
            else:
                self.class_fp[pred] += 1
                self.class_fn[target] += 1

    def compute_precision_recall_accuracy(self, pred_logist, targets):
        vpad_idx = self.task.source_dictionary[AIL_TOKEN_LABEL_FIELD].indices["<vpad>"]
        tmpstk_var_idx = self.task.source_dictionary[AIL_TOKEN_LABEL_FIELD].indices["<TmpStackVar>"]
        global_var_idx = self.task.source_dictionary[AIL_TOKEN_LABEL_FIELD].indices["<GlobalVar>"]
        return_idx = self.task.source_dictionary[AIL_TOKEN_LABEL_FIELD].indices["return"]
        address_idx = self.task.source_dictionary[AIL_TOKEN_LABEL_FIELD].indices["address"]
        
        # stack_idx = self.task.source_dictionary[AIL_TOKEN_LABEL_FIELD].indices["stack"]
        # check_idx = self.task.source_dictionary[AIL_TOKEN_LABEL_FIELD].indices["check"]
        # guard_idx = self.task.source_dictionary[AIL_TOKEN_LABEL_FIELD].indices["guard"]

        ignore_label_idxs = {vpad_idx, tmpstk_var_idx, global_var_idx, return_idx, address_idx}
        
        pred_labels = pred_logist.argmax(1)
        # reshape
        pred_labels = pred_labels.reshape(-1, 4).detach().cpu().numpy()
        targets = targets.reshape(-1, 4).detach().cpu().numpy()

        # post process: vote the predicted name if same var
        pred_labels, _ = self._post_process(pred_labels, targets)
            
        precision_list = []
        recall_list = []
        acc_list = []
        pred_strs = []
        tgt_strs = []
        for pred, target in zip(pred_labels, targets):
            if len(set(target).difference(ignore_label_idxs)) == 0:
                continue
            intersect = np.intersect1d(target, pred[pred != vpad_idx])
            tp = int(len(intersect))
            fp = int(len(pred[pred != vpad_idx])) - tp
            fn = int(len(target[target!=vpad_idx])) - tp
            if (tp + fp) > 0:
                precision_list.append(tp / (tp + fp))
            if (tp + fn) > 0:
                recall_list.append(tp / (tp + fn))
            union = np.union1d(target[target!=vpad_idx], pred[pred != vpad_idx])
            if int(len(union)) > 0:
                acc_list.append(int(len(intersect)) / int(len(union)))
            
            pred_str = self.task.source_dictionary[AIL_TOKEN_LABEL_FIELD].string(pred[pred != vpad_idx])
            target_str = self.task.source_dictionary[AIL_TOKEN_LABEL_FIELD].string(target[target != vpad_idx])
            pred_strs.append(pred_str.replace(' ', ''))
            tgt_strs.append(target_str.replace(' ', ''))
            self.gt_and_preds.append((target_str.replace(' ', ''), pred_str.replace(' ', '')))

        cer = self.CER(pred_strs, tgt_strs).item()
        return np.mean(precision_list), np.mean(recall_list), np.mean(acc_list), cer

    def forward(self, model, sample, reduce=True):
        mask_idx = self.task.source_dictionary[AIL_TOKEN_FIELD].indices["<mask>"]
        vpad_idx = self.task.source_dictionary[AIL_TOKEN_LABEL_FIELD].indices["<vpad>"]
        masked_code = sample["net_input"]["src_tokens"]["ail_token"].eq(mask_idx)

        if not masked_code.any() and not model.training:
            # Skip this sample during evaluation if no variable found.
            raise Exception('no variable found, drop sample')
        elif not masked_code.any():
            masked_code = None
            
        outputs, _ = model(**sample["net_input"], masked_code=masked_code, classification_head_name="maskvar")
        targets = sample["target"]["tgt_tokens"][AIL_TOKEN_LABEL_FIELD][masked_code]
        loss = 1000 * self.focal_loss(outputs.view(-1, outputs.size(-1)).float(), targets.view(-1))
        
        # the predicted index
        predicts = torch.argmax(outputs, dim=-1)
        
        # simple compute the accuracy
        remove_vpad = ~((targets == predicts) & targets.eq(vpad_idx))
        accuracy = torch.mean((predicts == targets).float()[remove_vpad])
        
        if model.training:
            # We do not compute the evaluation metric
            precision, recall, multi_accuracy, cer = 0.0, 0.0, 0.0, 0.0
        else:
            precision, recall, multi_accuracy, cer = self.compute_precision_recall_accuracy(outputs, targets)
            self.compute_precision_recall_accuracy_classwise(outputs, targets)
            # precision, recall, multi_accuracy = 0.0, 0.0, 0.0
            # if np.isnan(precision) or np.isnan(recall) or np.isnan(multi_accuracy):
            #     raise DropSample('drop sample')
            
            

        # count the sample size
        sample_size = len(outputs)

        predict_vars = outputs[remove_vpad, :]
        target_vars = targets[remove_vpad]

        if random.random() < 0.00005 and sample_size > 16:
            # only randomly log some prediction in case screen flushing
            idx = random.randint(0, sample_size-16)
            idx -= idx % 5

            targets_code_idx = target_vars.view(-1)[idx:idx+16]
            pred_x = predict_vars.view(-1, predict_vars.size(-1))[idx:idx+16]
            if pred_x.shape[0] != 0:
                pred_code_idx = torch.argmax(pred_x, dim=-1)
                tmp_var_idx = self.task.source_dictionary[AIL_TOKEN_LABEL_FIELD].indices["<TmpStackVar>"]
                remove_tmp_var = ~((targets_code_idx == pred_code_idx) & targets_code_idx.eq(tmp_var_idx))
                print(f'tgt code:', self.task.source_dictionary[AIL_TOKEN_LABEL_FIELD].string(targets_code_idx[remove_tmp_var]))
                print(f'pred code:', self.task.source_dictionary[AIL_TOKEN_LABEL_FIELD].string(pred_code_idx[remove_tmp_var]))

        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["nsentences"],
            "sample_size": sample_size,
            "accuracy": accuracy.data,
            "multi_accuracy": multi_accuracy,
            "precision": precision,
            "recall": recall,
            "cer": cer,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        acc_list = [log.get("accuracy") for log in logging_outputs if 'accuracy' in log]
        multi_accuracy_list = [log.get("multi_accuracy") for log in logging_outputs if 'multi_accuracy' in log]
        precision_list = [log.get("precision") for log in logging_outputs if 'precision' in log]
        recall_list = [log.get("recall") for log in logging_outputs if 'recall' in log]
        cer_list = [log.get("cer") for log in logging_outputs if 'cer' in log]

        precision_list = [p for p in precision_list if not np.isnan(p)]
        if len(precision_list) > 0:
            precision = sum(precision_list) / len(precision_list)
        else:
            precision = 0
        recall_list = [r for r in recall_list if not np.isnan(r)]
        if len(recall_list) > 0:
            recall = sum(recall_list) / len(recall_list)
        else:
            recall = 0
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0
        
        multi_accuracy_list = [acc for acc in multi_accuracy_list if not np.isnan(acc)]
        if len(multi_accuracy_list) > 0:
            multi_acc = sum(multi_accuracy_list) / len(multi_accuracy_list)
        else:
            multi_acc = 0.0

        metrics.log_scalar("loss", loss_sum / sample_size, sample_size, round=3)
        metrics.log_derived("code_ppl", lambda meters: utils.get_perplexity(meters["loss"].avg))
        metrics.log_scalar("accuracy", sum(acc_list) / len(acc_list), 1, round=4)
        metrics.log_scalar("multi_accuracy", multi_acc, 1, round=4)
        metrics.log_scalar("precision", precision, 1, round=4)
        metrics.log_scalar("recall", recall, 1, round=4)
        metrics.log_scalar("f1", f1, 1, round=4)
        cer_list = [c for c in cer_list if not np.isnan(c)]
        cer = 100.0 if len(cer_list) == 0 else sum(cer_list)/len(cer_list)
        metrics.log_scalar("cer", cer, 1, round=4)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
