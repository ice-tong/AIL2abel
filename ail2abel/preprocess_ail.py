import pickle
from typing import Dict
from .ail_tokenizer import AILTokener, TokenerContext, VAR_LABELS_COUNTER, AILOpType, Config


def create_samples_from_ail(ail_pickle_fpath: str) -> List[Dict[str, List[str]]]:
    with open(ail_pickle_fpath, 'rb') as f:
        ail_stmts = pickle.load(f)

    if len(ail_stmts) < 1:
        return []

    ctx = TokenerContext()
    ail_tokeners: Dict[str, "AILTokener"] = {}
    for stmt_addr, stmt in ail_stmts.items():
        ail_tokeners[stmt_addr] = AILTokener.from_ail_stmt(stmt, ctx=ctx)

    samples = []
    sample = {}
    max_tokens_len = 1024

    for idx, (stmt_addr, ail_tokener) in enumerate(ail_tokeners.items()):
        tokens = ail_tokener.tokens
        var_masked_tokens = ail_tokener.var_masked_tokens
        stmt_idxs = [str(idx) for _ in range(len(tokens))]
        op_idxs = [str(i) for i in range(len(tokens))]
        
        if "ail_token" not in sample:
            sample["ail_token"] = var_masked_tokens
            sample["ail_token_label"] = tokens
            sample["stmt_idxs"] = stmt_idxs
            sample["op_idxs"] = op_idxs
        else:
            sample["ail_token"] += var_masked_tokens
            sample["ail_token_label"] += tokens
            sample["stmt_idxs"] += stmt_idxs
            sample["op_idxs"] += op_idxs
        
        if len(sample["ail_token"]) >= max_tokens_len:
            samples.append(sample)
            sample = {}
    
    if sample not in samples:
        samples.append(sample)
    
    # We should drop the sample that we don't need
    valid_samples = []
    for sample in samples:
        if "ail_token" not in sample:
            continue
        if sample["ail_token"].count(AILOpType.VMASK.value) == 0:
            continue
        valid_samples.append(sample)
    return valid_samples
