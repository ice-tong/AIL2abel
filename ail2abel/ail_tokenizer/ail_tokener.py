import math
import struct
from typing import Any, List, Optional, TYPE_CHECKING, Union

from ailment import Stmt, Expr
from itanium_demangler import parse as demangle
from angr.sim_variable import SimMemoryVariable, SimStackVariable

from .name2label import split_var_name
from .ail_op_type import AILOpType, UNARY_OP_MAPPING, BINARY_OP_MAPPING
from .tokener_context import VAR_LABELS_COUNTER
from .cfg import Config

if TYPE_CHECKING:
    from ailment.statement import Statement
    from ailment.expression import Expression
    from angr.sim_variable import SimVariable

    from .tokener_context import TokenerContext


def split_func_name(func_name):
    results = []
    cursor = 0
    try:
        demangle_func_name = demangle(func_name)
        if  demangle_func_name is not None:
            func_name = str(demangle_func_name)
    except:
        pass
    for idx, char in enumerate(func_name):
        # if char in {"_", ".", '/', '-', '@', ':', '<', '>'}:
        if not char.isalpha():
            results.append(func_name[cursor:idx])
            cursor = idx+1
    results.append(func_name[cursor:])
    results = [i for i in results if i]
    if len(results) > 1:
        return results[:10]
    
    if len(results) == 0:
        print(func_name)
        return [func_name]

    # maybe camel case varaiable name
    maybe_camel_vname = results[0]
    results = []
    cursor = 0
    for idx, char in enumerate(maybe_camel_vname):
        if char.isupper():
            results.append(maybe_camel_vname[cursor:idx])
            cursor = idx
    results.append(maybe_camel_vname[cursor:])
    results = [i for i in results if i]

    return results[:10]


class AILTokenerBase:

    def __init__(self, raw_stuff: Any, ctx: "TokenerContext", variable: "SimVariable" = None):
        self.raw_stuff = raw_stuff
        self.ctx = ctx
        self.variable = variable

        # Lazy initialization of the tokens.
        self._tokens = None
        self._var_masked_tokens = None

    def var_tokens(self, variable: "SimVariable", var_mask: bool = False) -> List[str]:
        # Add the normalized variable ident to the tokens.
        normalized_var_ident = self.ctx.normalize_var_ident(
            variable.ident, is_func_arg=variable.is_function_argument)
        tokens = [normalized_var_ident]
        if var_mask:
            tokens += [AILOpType.VMASK.value] * Config.NUM_VAR_TOKENS
        else:
            if type(self.variable) is SimStackVariable and self.variable.name in [
                    "s_%0x" % abs(self.variable.offset), "s_-%0x" % abs(self.variable.offset)]:
                var_labels = ["<TmpStackVar>"]
            elif type(self.variable) is SimMemoryVariable and self.variable.name is not None \
                    and self.variable.name.startswith('g_'):
                var_labels = ["<GlobalVar>"]
            else:
                var_labels = split_var_name(variable.name)
            
            for var_label in var_labels[:Config.NUM_VAR_TOKENS]:
                VAR_LABELS_COUNTER[var_label] += 1

            if len(var_labels) < Config.NUM_VAR_TOKENS:
                var_labels += ["<vpad>"] * (Config.NUM_VAR_TOKENS - len(var_labels))
            else:
                var_labels = var_labels[:Config.NUM_VAR_TOKENS]
            tokens += var_labels
        return tokens

    def get_tokens(self) -> List[str]:
        tokens, var_masked_tokens = [], []
        if self.variable is not None:
            tokens += self.var_tokens(self.variable, var_mask=False)
            var_masked_tokens += self.var_tokens(self.variable, var_mask=True)
        return tokens, var_masked_tokens

    @property
    def tokens(self) -> List[str]:
        if self._tokens is None:
            self._tokens, self._var_masked_tokens = self.get_tokens()
        return self._tokens

    @property
    def var_masked_tokens(self) -> List[str]:
        if self._var_masked_tokens is None:
            self._tokens, self._var_masked_tokens = self.get_tokens()
        return self._var_masked_tokens

    def _var_hash(self) -> int:
        if self.variable is not None:
            return hash(self.variable.name)
        else:
            return 0


class ConstAILTokener(AILTokenerBase):

    def __init__(self, const_value, raw_stuff, ctx, variable: "SimVariable" = None):
        super().__init__(raw_stuff=raw_stuff, ctx=ctx, variable=variable)
        if type(const_value) is str:
            self._is_original_str = True
        else:
            self._is_original_str = False

        if type(const_value) is bytes:
            const_value = const_value.decode("utf-8", "replace")

        self.const_value = const_value
        
    def _is_string(self, num: int) -> Optional[str]:
        if num < 0:
            return None

        # convert to bytes
        byte_string = num.to_bytes(math.ceil((num.bit_length() + 7) / 8), Config.ENDIANNESS)

        if not byte_string:
            return None

        # check if the last byte is 0
        if byte_string[-1] != 0:
            return None

        # check if all bytes are printable
        for byte in byte_string[:-1]:
            if not (32 <= byte <= 126):
                return None

        return byte_string[:-1].decode('ascii')

    def _get_int_tokesn(self, value: int) -> List[str]:
        if value <= 0xff and value >= -0xff:
            return [hex(value)]
        elif value > 0x400000 and value < 0x400000 + 4*1024*1024*1024 and value in self.ctx.code_loc_map:
            # NOTE(yancong): We assume the integer in this range is a code location.
            return [self.ctx.get_code_loc(value)]
        elif value > 0x400000 and value < 0x400000 + 4*1024*1024*1024 and value in self.ctx.mem_loc_map:
            # NOTE(yancong): We assume the integer in this range is a memory location.
            return [self.ctx.get_mem_loc(value)]
        elif self._is_string(value) is not None:
            return self._get_str_tokens(self._is_string(value))
        else:
            num_hex_str = value.to_bytes(math.ceil((value.bit_length() + 7) / 8), Config.ENDIANNESS, signed=True).hex()
            num_hex_str = num_hex_str.lstrip('0')
            if len(num_hex_str) %2 != 0:
                num_hex_str = num_hex_str.rjust(len(num_hex_str)+1, '0')
            num_tokens = [num_hex_str[i:i+2] for i in range(0, len(num_hex_str), 2)]
            num_tokens = num_tokens[:10]
            return [f"<hex:{len(num_tokens)}bit>"] + num_tokens

    def _get_float_tokens(self, value: float) -> List[str]:
        num_hex_str = struct.pack('d', value).hex()
        num_tokens = [num_hex_str[i:i+2] for i in range(0, len(num_hex_str), 2)]
        return ["<fp_hex>"] + num_tokens

    def _get_str_tokens(self, value: str) -> List[str]:
        return self.ctx.tokenize_string(value)

    def get_tokens(self) -> List[str]:
        tokens, var_masked_tokens = super().get_tokens()
        value = self.const_value
        if type(value) is str:
            const_tokens = self._get_str_tokens(value)
        elif type(value) is int:
            const_tokens = self._get_int_tokesn(value)
        elif type(value) is float:
            const_tokens = self._get_float_tokens(value)
        elif value is None:
            const_tokens = [AILOpType.NONE.value]
        else:
            raise TypeError(f"Unsupported type `{type(value)}`!")
        tokens += const_tokens
        var_masked_tokens += const_tokens
        return tokens, var_masked_tokens
    
    def __hash__(self) -> int:
        hash_list = [self._var_hash()]
        value = self.const_value
        if type(value) is str:
            hash_list.append(hash(value))
        elif type(value) is int and value > 0x400000 and value < 0x400000 + 4*1024*1024*1024 and value in self.ctx.code_loc_map:
            # NOTE(yancong): We assume the integer in this range is a code location.
            hash_list.append(self.ctx.get_code_loc(value))
        elif type(value) is int and value > 0x400000 and value < 0x400000 + 4*1024*1024*1024 and value in self.ctx.mem_loc_map:
            # NOTE(yancong): We assume the integer in this range is a memory location.
            hash_list.append(self.ctx.get_mem_loc(value))
        elif type(value) is int:
            hash_list.append(value)
        elif type(value) is float:
            hash_list.append(value)
        elif value is None:
            hash_list.append(AILOpType.NONE.value)
        return hash(tuple(hash_list))


class TmpAILTokener(AILTokenerBase):
    
    def __init__(self, tmp_idx, raw_stuff, ctx, variable: "SimVariable" = None):
        super().__init__(raw_stuff=raw_stuff, ctx=ctx, variable=variable)
        self.tmp_idx = tmp_idx
    
    def get_tokens(self) -> List[str]:
        tokens, var_masked_tokens = super().get_tokens()
        # NOTE: We do not add the tmp index to the tokens.
        tmp_tokens = [AILOpType.TMP.value]
        tokens += tmp_tokens
        var_masked_tokens += tmp_tokens
        return  tokens, var_masked_tokens
    
    def __hash__(self) -> int:
        return hash((self._var_hash(), AILOpType.TMP.value))


class RegAILTokener(AILTokenerBase):

    def __init__(self, reg_offset, raw_stuff, ctx, variable: "SimVariable" = None):
        super().__init__(raw_stuff=raw_stuff, ctx=ctx, variable=variable)
        self.reg_offset = reg_offset

    def get_tokens(self) -> List[str]:
        tokens, var_masked_tokens = super().get_tokens()
        # NOTE(yancong): We do not add the reg offset to the tokens.
        reg_tokens = [AILOpType.REG.value]
        tokens += reg_tokens
        var_masked_tokens += reg_tokens
        return tokens, var_masked_tokens
    
    def __hash__(self) -> int:
        return hash((self._var_hash(), AILOpType.REG.value, self.reg_offset))


class AILTokener(AILTokenerBase):
    """The AIL tokener of a statement or an expression.

    Args:
        op_type (str): The operation type of the statement or expression.
        operands (List["AILTokener"]): The AIL tokens of the operands.
        variable: The variable attached to the statement or expression.
    """

    def __init__(self,
                 op_type: "AILOpType",
                 operands: List["AILTokenerBase"],
                 raw_stuff: Any,
                 ctx: "TokenerContext",
                 variable: "SimVariable" = None):
        super().__init__(raw_stuff=raw_stuff, ctx=ctx, variable=variable)
        # assert isinstance(op_type, AILOpType)
        self.op_type = op_type
        self.operands = operands

    @classmethod
    def from_ail_stmt(cls, stmt: "Statement", ctx: "TokenerContext") -> "AILTokener":
        """Create an AILTokener object from an AIL statement.

        Args:
            stmt (Statement): The AIL statement.
            ctx (TokenerContext): The tokener context, for normalizing the code location and
                variable ident.

        Returns:
            AILTokener: The AILTokener object.
        """
        if type(stmt) is Stmt.Assignment:
            dst = cls.from_ail_expr(stmt.dst, ctx=ctx)
            src = cls.from_ail_expr(stmt.src, ctx=ctx)
            return cls(AILOpType.ASSIGN, [dst, src], raw_stuff=stmt, ctx=ctx)
        elif type(stmt) is Stmt.Store:
            addr = cls.from_ail_expr(stmt.addr, ctx=ctx)
            data = cls.from_ail_expr(stmt.data, ctx=ctx)
            if type(addr) is ConstAILTokener:
                # NOTE: We normalize the address-like constant to the code location.
                ctx.add_mem_loc(addr.const_value, check_range=False)
            return cls(AILOpType.STORE, [addr, data], raw_stuff=stmt, ctx=ctx, variable=stmt.variable)
        elif type(stmt) is Stmt.Jump:
            target = cls.from_ail_expr(stmt.target, ctx=ctx)
            if type(target) is ConstAILTokener:
                # NOTE: We normalize the address-like constant to the code location.
                ctx.add_code_loc(target.const_value, check_range=False)
            return cls(AILOpType.JUMP, [target], raw_stuff=stmt, ctx=ctx)
        elif type(stmt) is Stmt.ConditionalJump:
            condition = cls.from_ail_expr(stmt.condition, ctx=ctx)
            true_target = cls.from_ail_expr(stmt.true_target, ctx=ctx)
            false_target = cls.from_ail_expr(stmt.false_target, ctx=ctx)
            if type(true_target) is ConstAILTokener:
                # NOTE: We normalize the address-like constant to the code location.
                ctx.add_code_loc(true_target.const_value, check_range=False)
            if type(false_target) is ConstAILTokener:
                # NOTE: We normalize the address-like constant to the code location.
                ctx.add_code_loc(false_target.const_value, check_range=False)
            # NOTE(yancong): the target could be None
            return cls(AILOpType.CJUMP, [condition, true_target, false_target], raw_stuff=stmt, ctx=ctx)
        elif type(stmt) is Stmt.Call:
            if type(stmt.target) is Expr.Const and type(stmt.target.value) is str:
                # try demangle the function name if it is mangled
                try:
                    demangle_func_name = demangle(stmt.target.value)
                    if demangle_func_name is not None:
                        stmt.target.value = str(demangle_func_name)
                except:
                    pass
            elif type(stmt.target) is Expr.Const and type(stmt.target.value) is int:
                ctx.add_code_loc(stmt.target.value, check_range=False)
            target = cls.from_ail_expr(stmt.target, ctx=ctx)
            args_list = stmt.args if stmt.args is not None else []
            args = [cls.from_ail_expr(arg, ctx=ctx) for arg in args_list]
            # TODO(yancong): Handle the return value and calling convention.
            return cls(AILOpType.CALL, [target] + args, raw_stuff=stmt, ctx=ctx)
        elif type(stmt) is Stmt.Return:
            ret_exprs = [cls.from_ail_expr(ret_expr, ctx=ctx) for ret_expr in stmt.ret_exprs]
            # TODO(yancong): Handle the target.
            return cls(AILOpType.RETURN, ret_exprs, raw_stuff=stmt, ctx=ctx)
        elif type(stmt) is Stmt.Label:
            # NOTE: We normalize the address-like label to the code location.
            ctx.add_code_loc(stmt.ins_addr, check_range=False)
            addr = cls.from_ail_expr(stmt.ins_addr, ctx=ctx)
            return cls(AILOpType.LABEL, [addr], raw_stuff=stmt, ctx=ctx)
        elif type(stmt) is Stmt.DirtyStatement:
            return cls(AILOpType.DIRTYSTMT, [], raw_stuff=stmt, ctx=ctx)
        else:
            raise TypeError(f"Unsupported stmt type `{type(stmt)}`!")

    @classmethod
    def from_ail_expr(cls, expr: "Expression", ctx: "TokenerContext") -> "AILTokenerBase":
        """Create an AILTokener object from an AIL expression.

        Args:
            expr (Expression): The AIL expression.
            ctx (TokenerContext): The tokener context, for normalizing the code location and
                variable ident.

        Returns:
            AILTokener: The AILTokener object.
        """
        if type(expr) in (int, str) or expr is None:
            if type(expr) is str and expr not in ["__CFADD__"]:
                print("Unhandeld str: ", expr)
            # ??? Should we build LiteralAILTokener for int / str?
            return ConstAILTokener(expr, raw_stuff=expr, ctx=ctx)
        elif type(expr) is Stmt.Call:
            # NOTE(yancong): The call expression is Statement and Expression at the same time.
            if type(expr.target) is Expr.Const and type(expr.target.value) is str:
                # try demangle the function name if it is mangled
                try:
                    demangle_func_name = demangle(expr.target.value)
                    if demangle_func_name is not None:
                        expr.target.value = str(demangle_func_name)
                except:
                    pass
            elif type(expr.target) is Expr.Const and type(expr.target.value) is int:
                ctx.add_code_loc(expr.target.value, check_range=False)
            target = cls.from_ail_expr(expr.target, ctx=ctx)
            args_list = expr.args if expr.args is not None else []
            args = [cls.from_ail_expr(arg, ctx=ctx) for arg in args_list]
            # TODO(yancong): Handle the return value and calling convention.
            return cls(AILOpType.CALL, [target] + args, raw_stuff=expr, ctx=ctx)
        elif type(expr) is Expr.Const:
            return ConstAILTokener(expr.value, raw_stuff=expr, ctx=ctx, variable=expr.variable)
        elif type(expr) is Expr.Tmp:
            return TmpAILTokener(expr.tmp_idx, raw_stuff=expr, ctx=ctx, variable=expr.variable)
        elif type(expr) is Expr.Register:
            return RegAILTokener(expr.reg_offset, raw_stuff=expr, ctx=ctx, variable=expr.variable)
        elif type(expr) in (Expr.UnaryOp, Expr.Reinterpret):
            # assert expr.op in UNARY_OP_MAPPING, f"Unsupported unary op `{expr.op}`!"
            if expr.op not in UNARY_OP_MAPPING:
                print(f"Unsupported unary op `{expr.op}`!")
                return cls(AILOpType.UNKOPTYPE, [], raw_stuff=expr, ctx=ctx, variable=expr.variable)
            operand = cls.from_ail_expr(expr.operand, ctx=ctx)
            return cls(UNARY_OP_MAPPING[expr.op], [operand], raw_stuff=expr, ctx=ctx, variable=expr.variable)
        elif type(expr) is Expr.BinaryOp:
            # assert expr.op in BINARY_OP_MAPPING, f"Unsupported binary op `{expr.op}`!"
            if expr.op not in BINARY_OP_MAPPING:
                print(f"Unsupported binary op `{expr.op}`!")
                return cls(AILOpType.UNKOPTYPE, [], raw_stuff=expr, ctx=ctx, variable=expr.variable)
            op_type = BINARY_OP_MAPPING[expr.op]
            operands = [cls.from_ail_expr(operand, ctx=ctx) for operand in expr.operands]
            return cls(op_type, operands, raw_stuff=expr, ctx=ctx, variable=expr.variable)
        elif type(expr) is Expr.TernaryOp:
            raise NotImplementedError(expr.op)
        elif type(expr) is Expr.Convert:
            operand = cls.from_ail_expr(expr.operand, ctx=ctx)
            from_bits = cls.from_ail_expr(expr.from_bits, ctx=ctx)
            bits = cls.from_ail_expr(expr.bits, ctx=ctx)
            return cls(AILOpType.CONVERT, [from_bits, bits, operand], raw_stuff=expr, ctx=ctx)
        elif type(expr) is Expr.Load:
            addr = cls.from_ail_expr(expr.addr, ctx=ctx)
            if type(addr) is ConstAILTokener:
                # NOTE: We normalize the address-like constant to the code location.
                ctx.add_mem_loc(addr.const_value, check_range=False)
            size = cls.from_ail_expr(expr.size, ctx=ctx)
            return cls(AILOpType.LOAD, [addr, size], raw_stuff=expr, ctx=ctx, variable=expr.variable)
        elif type(expr) is Expr.ITE:
            condition = cls.from_ail_expr(expr.cond, ctx=ctx)
            iftrue = cls.from_ail_expr(expr.iftrue, ctx=ctx)
            iffalse = cls.from_ail_expr(expr.iffalse, ctx=ctx)
            if type(iftrue) is ConstAILTokener:
                # NOTE: We normalize the address-like constant to the code location.
                ctx.add_code_loc(iftrue.const_value, check_range=False)
            if type(iffalse) is ConstAILTokener:
                # NOTE: We normalize the address-like constant to the code location.
                ctx.add_code_loc(iffalse.const_value, check_range=False)
            return cls(AILOpType.ITE, [condition, iftrue, iffalse], raw_stuff=expr, ctx=ctx, variable=expr.variable)
        elif type(expr) is Expr.DirtyExpression:
            return cls(AILOpType.DIRTYEXPR, [], raw_stuff=expr, ctx=ctx)
        elif type(expr) is Expr.VEXCCallExpression:
            raise NotImplementedError(expr)
        elif type(expr) is Expr.MultiStatementExpression:
            raise NotImplementedError(expr)
        elif type(expr) is Expr.BasePointerOffset:
            raise NotImplementedError(expr)
        elif type(expr) is Expr.StackBaseOffset:
            offset = cls.from_ail_expr(expr.offset, ctx=ctx)
            return cls(AILOpType.STKOFFSET, [offset], raw_stuff=expr, ctx=ctx, variable=expr.variable)
        else:
            raise TypeError(f"Unsupported expr type `{type(expr)}`!")

    def get_tokens(self) -> List[str]:
        tokens, var_masked_tokens = super().get_tokens()
        tokens.append(self.op_type.value)
        var_masked_tokens.append(self.op_type.value)
        for operand in self.operands:
            tokens += operand.tokens
            var_masked_tokens += operand.var_masked_tokens
        return tokens, var_masked_tokens

    def __hash__(self) -> int:
        hash_list = [self._var_hash(), self.op_type.value]
        for operand in self.operands:
            hash_list.append(hash(operand))
        return hash(tuple(hash_list))
