import os
import argparse
import pickle
import logging
from collections import OrderedDict
from multiprocessing import Process
import json
from typing import Dict, List

from func_timeout import func_set_timeout, FunctionTimedOut

import angr
import ailment
from ailment.tagged_object import TaggedObject
from ailment.expression import StackBaseOffset, BasePointerOffset, Register, UnaryOp, Const, Tmp
from angr.analyses.decompiler import AILSimplifier, BlockSimplifier
from angr.analyses.decompiler.optimization_passes import _all_optimization_passes
from angr.analyses.decompiler.optimization_passes.stack_canary_simplifier import StackCanarySimplifier
from angr.analyses import register_analysis

from ail_tokenizer import AILTokener, TokenerContext, VAR_LABELS_COUNTER, AILOpType, Config
from ail_tokenizer.name2label import split_var_name


class AARCH64StackCanarySimplifier(StackCanarySimplifier):
    
    ARCHES = ["AARCH64"]
    
    def _is_stack_chk_guard_laod_expr(self, load_expr) -> bool:
        if not isinstance(load_expr, ailment.Expr.Load):
            return False
        if not isinstance(load_expr.addr, ailment.Expr.Const):
            return False
        # Check if the address is the glaobal vraibale that named `__stack_chk_guard`
        symbol = self.project.loader.find_symbol(load_expr.addr.value)
        if symbol is not None and symbol.name == "__stack_chk_guard":
            return True
        return False
    
    def _find_canary_init_stmt(self):
        first_block = self._get_block(self._func.addr)
        if first_block is None:
            return None

        for idx, stmt in enumerate(first_block.statements):
            if (
                isinstance(stmt, ailment.Stmt.Store)
                and isinstance(stmt.addr, ailment.Expr.StackBaseOffset)
                and self._is_stack_chk_guard_laod_expr(stmt.data)
            ):
                return first_block, idx

        return None
    
    def _find_canary_comparison_stmt(self, block, canary_value_stack_offset):
        for idx, stmt in enumerate(block.statements):
            if not isinstance(stmt, ailment.Stmt.ConditionalJump):
                continue
            
            condition = stmt.condition
            if not isinstance(condition, ailment.Expr.BinaryOp) or condition.op != "CmpNE":
                raise NotImplementedError(f"Unhanlded canary comparison statement: {stmt}")
            
            expr0, expr1 = condition.operands
            if not isinstance(expr0, ailment.Expr.Load) or not isinstance(expr1, ailment.Expr.Load):
                raise NotImplementedError(f"Unhanlded canary comparison statement: {stmt}")
            
            if not (
                (
                    self._is_stack_canary_load_expr(expr0, self.project.arch.bits, canary_value_stack_offset)
                    and self._is_stack_chk_guard_laod_expr(expr1)
                )
                or
                (
                    self._is_stack_chk_guard_laod_expr(expr0) and
                    self._is_stack_canary_load_expr(expr1, self.project.arch.bits, canary_value_stack_offset)
                )
            ):
                raise NotImplementedError(f"Unhanlded canary comparison statement: {stmt}")

            # Fanlly, we find the canary comparison statement by the above checks.
            return idx


def get_optimization_passes(proj: "angr.Project"):
    # NOTE(yancong): Add all optimization passes for aarch64.
    if proj.arch.name != "AARCH64":
        return None
    passes = []
    for pass_, is_default in _all_optimization_passes:
        if is_default:
            passes.append(pass_)
    passes.append(AARCH64StackCanarySimplifier)
    return passes


class DONOTAILSimplifier(AILSimplifier):

    def _simplify(self, *args, **kwargs):
        # DO NOT call super().__init__(), we don't want to simplify the AIL
        pass


class DONOTBlockSimplifier(BlockSimplifier):

    def _analyze(self, *args, **kwargs):
        # DO NOT call super().__init__(), we don't want to simplify the AIL
        block = self.block

        # We only want to replace tmp variable with the stack base offset, ohterwise the debug 
        # variable mapping will be wrong.
        propagator = self._compute_propagation(block)
        prop_state = list(propagator.model.states.values())[0]
        replacements = prop_state._replacements
        if replacements:
            for codeloc, repls in replacements.items():
                item_to_remove = []
                for old, new in repls.items():
                    if not isinstance(new, BasePointerOffset) and isinstance(old, Tmp):
                        item_to_remove.append(old)
                for item in item_to_remove:
                    repls.pop(item)

            _, block = self._replace_and_build(
                block, replacements, replace_registers=True,
                replacement_recorder=self._replacement_recorder)

        self.result_block = block


logging.root.setLevel(logging.FATAL)


def _hack_str(self):
    if type(self.value) is bytes:
        return self.value.decode("utf-8", errors="ignore")
    if type(self.value) is str:
        return self.value
    try:
        return "%#x<%d>" % (self.value, self.bits)
    except TypeError:
        return "%f<%d>" % (self.value, self.bits)

Const.__str__ = _hack_str


def extract_memory_data(cfg, val):
    if type(val) is int and val in cfg.functions:
        func = cfg.functions[val]
        return func.name
    
    if val not in cfg.memory_data:
        return val

    memory_order = 'little' if cfg.project.arch.memory_endness == 'Iend_LE' else 'big'

    mem_data = cfg.memory_data[val]
    try:
        if mem_data.sort == "string" or mem_data.sort == "unicode":
            val = cfg.project.loader.memory.load(val, mem_data.size)
        elif mem_data.sort == "integer":
            val = int.from_bytes(cfg.project.loader.memory.load(val, mem_data.size), memory_order)
        elif mem_data.sort == "code reference":
            val = "code reference"
        elif mem_data.sort == "unknown" and mem_data.size == 4:
            val = int.from_bytes(cfg.project.loader.memory.load(val, mem_data.size), memory_order)
        elif mem_data.sort == "unknown" and mem_data.size != 4:
            # unknown data
            # FIXME(yancong): should we truncate the data with limited size?
            val = int.from_bytes(cfg.project.loader.memory.load(val, mem_data.size), memory_order)
        elif mem_data.sort == "pointer-array":
            val = int.from_bytes(cfg.project.loader.memory.load(val, mem_data.size), memory_order)
        else:
            val = int.from_bytes(cfg.project.loader.memory.load(val, mem_data.size), memory_order)
    except Exception as e:
        print("extract memory data failed:", e)
        # raise e
    return val


def extract_const(stmt_or_expr, consts=None):
    if consts is None:
        consts = []
    
    if isinstance(stmt_or_expr, Const):
        consts.append(stmt_or_expr)
        return consts

    if isinstance(stmt_or_expr, (list, tuple)):
        for item in stmt_or_expr:
            extract_const(item, consts)
        return consts

    if not isinstance(stmt_or_expr, TaggedObject):
        return consts
    
    slots = stmt_or_expr.__slots__
    if isinstance(stmt_or_expr, StackBaseOffset):
        # The slots of StackBaseOffset is None, use the slots of BasePointerOffset.
        slots = BasePointerOffset.__slots__
    if isinstance(stmt_or_expr, Register):
        slots = list(Register.__slots__) + ['variable', 'variable_offset']
    
    if isinstance(stmt_or_expr, UnaryOp):
        slots = list(stmt_or_expr.__slots__) + ['operand', 'variable', 'variable_offset']

    for attr in slots:
        extract_const(getattr(stmt_or_expr, attr), consts)

    return consts


def extract_var(stmt_or_expr, variables=None):
    if variables is None:
        variables = []
    
    if isinstance(stmt_or_expr, (list, tuple)):
        for item in stmt_or_expr:
            extract_var(item, variables)
        return variables

    if not isinstance(stmt_or_expr, TaggedObject):
        return variables
    
    slots = stmt_or_expr.__slots__
    if isinstance(stmt_or_expr, StackBaseOffset):
        # The slots of StackBaseOffset is None, use the slots of BasePointerOffset.
        slots = BasePointerOffset.__slots__
    if isinstance(stmt_or_expr, Register):
        slots = list(Register.__slots__) + ['variable', 'variable_offset']
    
    if isinstance(stmt_or_expr, UnaryOp):
        slots = list(stmt_or_expr.__slots__) + ['operand', 'variable', 'variable_offset']

    for attr in slots:
        if attr == 'variable':
            var = getattr(stmt_or_expr, attr)
            if var is not None:
                variables.append(var)
        else:
            extract_var(getattr(stmt_or_expr, attr), variables)

    return variables


def tokenize_ail_stmts(ail_stmts: Dict[str, "TaggedObject"]) -> List[Dict[str, List[str]]]:
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

        vid2var = {}
        for ail_var in extract_var(ail_stmts[stmt_addr]):
            vid = ctx.normalize_var_ident(
                ail_var.ident, is_func_arg=ail_var.is_function_argument)
            vid2var[vid] = {"var_name": ail_var.name, "var_labels": split_var_name(ail_var.name)}

        if "ail_token" not in sample:
            sample["ail_token"] = var_masked_tokens
            sample["ail_token_label"] = tokens
            sample["stmt_idxs"] = stmt_idxs
            sample["op_idxs"] = op_idxs
            sample["ail_variables"] = vid2var
        else:
            sample["ail_token"] += var_masked_tokens
            sample["ail_token_label"] += tokens
            sample["stmt_idxs"] += stmt_idxs
            sample["op_idxs"] += op_idxs
            sample["ail_variables"].update(vid2var)
            
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

    for sample in valid_samples:
        for key, value in sample.items():
            if isinstance(value, list):
                sample[key] = ' '.join(value)
    return valid_samples


@func_set_timeout(10*2)
def clinic_it(proj, func, without_simplification):
    if without_simplification:
        register_analysis(DONOTAILSimplifier, "AILSimplifier")
        register_analysis(DONOTBlockSimplifier, "AILBlockSimplifier")
    else:
        register_analysis(AILSimplifier, "AILSimplifier")
        register_analysis(BlockSimplifier, "AILBlockSimplifier")
    optimization_passes = get_optimization_passes(proj)
    return proj.analyses.Clinic(func, remove_dead_memdefs=False,
                                optimization_passes=optimization_passes)


@func_set_timeout(30*2)
def process_function(function, proj, cfg, debug, without_simplification):
    try:
        clinic = clinic_it(proj, function, without_simplification)
    except FunctionTimedOut:
        return
    except BaseException as e:
        if debug:
            raise e
        return

    if clinic.graph is None:
        return

    var_mgr = clinic.variable_kb.variables[function.addr]

    stmts = OrderedDict()

    for ail_block in sorted(clinic.graph, key=lambda x: x.addr):
        for stmt in ail_block.statements:
            consts = extract_const(stmt)
            for const in consts:
                const.value = extract_memory_data(cfg, const.value)
            vars = extract_var(stmt)
            for var in vars:
                unified_var = var_mgr.unified_variable(var)
                if unified_var is not None:
                    var.ident = unified_var.ident

            stmts[f'{ail_block.addr}@{stmt.idx}'] = stmt
    return stmts


def process_elf(elf_fpath, pickle_dir, without_simplification, debug):
    """
    Process an ELF file and return a dictionary of functions and their
    corresponding basic blocks.
    """
    print(f'processing elf: {elf_fpath}')
    try:
        # Load the binary
        proj = angr.Project(elf_fpath, load_options={"auto_load_libs": False, "load_debug_info": True})
        proj.kb.dvars.load_from_dwarf()
        cfg = proj.analyses.CFG(show_progressbar=debug, data_references=True, normalize=True)
    except Exception as e:
        if debug:
            raise e
        else:
            print(e)
        return

    os.makedirs(pickle_dir, exist_ok=True)
    
    if without_simplification:
        save_dir = os.path.dirname(pickle_dir)
        wo_pickle_dir = os.path.join(save_dir+'_without_simp', os.path.basename(elf_fpath))
        os.makedirs(wo_pickle_dir, exist_ok=True)

    for function in list(cfg.kb.functions.values()):

        if function.is_simprocedure or function.is_plt or function.alignment:
            continue

        try:
            stmts = process_function(function, proj, cfg, debug, without_simplification=without_simplification)
        except Exception as e:
            # if debug:
            #     raise e
            # else:
            #     print(e)
            continue
        
        if stmts is None:
            continue

        samples = tokenize_ail_stmts(stmts)
        if len(samples) != 0:
            sample_json_fpath = os.path.join(pickle_dir, f'{function.name}.json')
            with open(sample_json_fpath, 'w') as f:
                json.dump(tokenize_ail_stmts(stmts), f, indent=4)

        if debug:
            pickle_fpath = os.path.join(pickle_dir, f'{function.name}.pkl')
            with open(pickle_fpath, 'wb') as f:
                pickle.dump(stmts, f)
            with open(pickle_fpath.replace('.pkl', '.txt'), 'w', encoding='utf-8') as f:
                for stmt in stmts.values():
                    f.write(str(stmt) + '\n')


def run(elf_fnames, elf_dir, save_dir, args, debug=False, proc_num=4):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for elf_fname in elf_fnames:
        # elf_fname = 'todolist.o' # CFI parse error
        # elf_fname = 'auth'  # AttributeError: 'MultiValuedMemory' object has no attribute '_pages'
        # elf_fname = 'smtpctl'
        elf_fpath = os.path.join(elf_dir, elf_fname)
        pickle_dir = os.path.join(save_dir, elf_fname)

        if os.path.exists(pickle_dir) and not debug:
            print(f'skip since local save path exists: {pickle_dir}')
            continue

        if os.path.getsize(elf_fpath) > 1024 * 1024 * 100:
            print(f'skip since bianry size > 100M: {elf_fpath}')
            continue

        try:
            process = Process(target=process_elf, args=(
                elf_fpath, pickle_dir, args.without_simplification, debug))
            process.start()
            process.join(timeout=60*20*2)
        except Exception as e:
            if debug:
                raise e
            else:
                print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_dir", default="./elf_files")
    parser.add_argument("out_dir", default="./pickle_files")
    parser.add_argument("-x", "--debug", default=False, action='store_true')
    parser.add_argument("-wos", "--without_simplification", default=False, action='store_true')
    args = parser.parse_args()

    root_binary_dir = os.path.normpath(args.in_dir)
    root_save_dir = os.path.normpath(args.out_dir)
    if not os.path.exists(root_save_dir):
        os.makedirs(root_save_dir)

    binary_fnames = os.listdir(root_binary_dir)

    run(elf_fnames=binary_fnames, elf_dir=root_binary_dir,
        save_dir=root_save_dir, args=args, debug=args.debug)
