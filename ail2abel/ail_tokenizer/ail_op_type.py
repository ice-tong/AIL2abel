from enum import Enum


class AILOpType(Enum):
    """The operation type of an AIL statement or expression."""
    # Statements
    ASSIGN = "[ASSIGN]"
    STORE = "[STORE]"
    JUMP = "[JUMP]"
    CJUMP = "[CJUMP]"  # Conditional jump
    CALL = "[CALL]"
    RETURN = "[RETURN]"
    LABEL = "[LABEL]"
    DIRTYSTMT = "[DIRTYSTMT]"  # Dirty statement

    # Expressions
    LOAD = "[LOAD]"
    ITE = "[ITE]"  # If-then-else
    DIRTYEXPR = "[DIRTYEXPR]"  # Dirty expression

    # Unary operations
    CONVERT = "[CONVERT]"
    REINTERPRET = "[REINTERPRET]"
    NOT = "[NOT]"
    NEG = "[NEG]"
    ABS = "[ABS]"
    SQRT = "[SQRT]"
    GETMSBS = "[GETMSBS]"
    UNPACK = "[UNPACK]"

    # Binary operations
    ADD = "[ADD]"
    ADDF = "[ADDF]"
    SUB = "[SUB]"
    SUBF = "[SUBF]"
    MUL = "[MUL]"
    MULF = "[MULF]"
    DIV = "[DIV]"
    DIVF = "[DIVF]"
    DIVMOD = "[DIVMOD]"
    MOD = "[MOD]"
    XOR = "[XOR]"
    AND = "[AND]"
    LOGICAL_AND = "[LOGICAL_AND]"
    OR = "[OR]"
    LOGICAL_OR = "[LOGICAL_OR]"
    SHL = "[SHL]"
    SHR = "[SHR]"
    SAR = "[SAR]"
    CMPF = "[CMPF]"
    CMPEQ = "[CMPEQ]"
    CMPNE = "[CMPNE]"
    CMPLT = "[CMPLT]"
    CMPLE = "[CMPLE]"
    CMPGT = "[CMPGT]"
    CMPGE = "[CMPGE]"
    CMPLTs = "[CMPLTs]"
    CMPLEs = "[CMPLEs]"
    CMPGTs = "[CMPGTs]"
    CMPGEs = "[CMPGEs]"
    CONCAT = "[CONCAT]"
    MULL = "[MULL]"
    ROUND = "[ROUND]"
    SHRN = "[SHRN]"
    SHLN = "[SHLN]"
    SARN = "[SARN]"
    CATEVENLANE = "[CATEVENLANE]"
    CATODDLANE = "[CATODDLANE]"
    PERM = "[PERM]"
    CLZ = "[CLZ]"  # ???(yancong): What is this? Unary or binary?
    CTZ = "[CTZ]"
    INTERLEAVELO = "[INTERLEAVELO]"
    INTERLEAVEHI = "[INTERLEAVEHI]"
    MAX = "[MAX]"
    MIN = "[MIN]"
    CASCMPNE = "[CASCMPNE]"
    EXPCMPNE = "[EXPCMPNE]"
    QNARROWBIN = "[QNARROWBIN]"
    MULHI = "[MULHI]"
    QADD = "[QADD]"
    QSUB = "[QSUB]"

    # Atomic operations
    REG = "[REG]"
    TMP = "[TMP]"
    STKOFFSET = "[STKOFFSET]"  # Stack base offset
    
    # UNKNOWN
    UNKOPTYPE = "[UNKOPTYPE]"
    
    # None Op
    NONE = "[NONE]"
    
    # variable mask and padding token
    VMASK = "<mask>"
    VPAD = "<vpad>"


UNARY_OP_MAPPING = {
    "Convert": AILOpType.CONVERT,
    "Reinterpret": AILOpType.REINTERPRET,
    "Not": AILOpType.NOT,
    "Neg": AILOpType.NEG,
    "Abs": AILOpType.ABS,
    "Sqrt": AILOpType.SQRT,
    "Clz": AILOpType.CLZ,
    "Ctz": AILOpType.CTZ,
    "GetMSBs": AILOpType.GETMSBS,
    "unpack": AILOpType.UNPACK,
}

BINARY_OP_MAPPING = {
    "Add": AILOpType.ADD,
    "AddF": AILOpType.ADDF,
    "Sub": AILOpType.SUB,
    "SubF": AILOpType.SUBF,
    "Mul": AILOpType.MUL,
    "MulF": AILOpType.MULF,
    "Div": AILOpType.DIV,
    "DivF": AILOpType.DIVF,
    "DivMod": AILOpType.DIVMOD,
    "Mod": AILOpType.MOD,
    "Xor": AILOpType.XOR,
    "And": AILOpType.AND,
    "LogicalAnd": AILOpType.LOGICAL_AND,
    "Or": AILOpType.OR,
    "LogicalOr": AILOpType.LOGICAL_OR,
    "Shl": AILOpType.SHL,
    "Shr": AILOpType.SHR,
    "Sar": AILOpType.SAR,
    "CmpF": AILOpType.CMPF,
    "CmpEQ": AILOpType.CMPEQ,
    "CmpNE": AILOpType.CMPNE,
    "CmpLT": AILOpType.CMPLT,
    "CmpLE": AILOpType.CMPLE,
    "CmpGT": AILOpType.CMPGT,
    "CmpGE": AILOpType.CMPGE,
    "CmpLTs": AILOpType.CMPLTs,
    "CmpLEs": AILOpType.CMPLEs,
    "CmpGTs": AILOpType.CMPGTs,
    "CmpGEs": AILOpType.CMPGEs,
    "Concat": AILOpType.CONCAT,
    "Mull": AILOpType.MULL,
    "Round": AILOpType.ROUND,
    "ShrN": AILOpType.SHRN,
    "ShlN": AILOpType.SHLN,
    "SarN": AILOpType.SARN,
    "CatEvenLane": AILOpType.CATEVENLANE,
    "CatOddLane": AILOpType.CATODDLANE,
    "Perm": AILOpType.PERM,
    "Clz": AILOpType.CLZ,
    "InterleaveLO": AILOpType.INTERLEAVELO,
    "InterleaveHI": AILOpType.INTERLEAVEHI,
    "Max": AILOpType.MAX,
    "Min": AILOpType.MIN,
    "CasCmpNE": AILOpType.CASCMPNE,
    "ExpCmpNE": AILOpType.EXPCMPNE,
    "QNarrowBin": AILOpType.QNARROWBIN,
    "MulHi": AILOpType.MULHI,
    "QAdd": AILOpType.QADD,
    "QSub": AILOpType.QSUB,
}
