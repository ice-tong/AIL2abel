from typing import Dict, List, Optional, TYPE_CHECKING
from collections import Counter

from .cfg import Config

if TYPE_CHECKING:
    from tokenizers import Tokenizer


VAR_LABELS_COUNTER = Counter()


class TokenerContext:
    
    def __init__(self) -> None:
        self.code_loc_map: Dict[int, str] = {}
        self.mem_loc_map: Dict[int, str] = {}
        self.var_ident_map: Dict[str, str] = {}
        self.arg_var_ident_map: Dict[str, str] = {}
        self.str_set = set()

    def add_code_loc(self, addr: int, check_range: bool = True) -> str:
        if check_range:
            assert addr >= 0x400000 and addr <= 0x400000 + 4*1024*1024*1024, \
                f"Invalid code location: {hex(addr)}"
        if addr not in self.code_loc_map:
            self.code_loc_map[addr] = f"<code_loc:{len(self.code_loc_map)}>"
        return self.code_loc_map[addr]

    def get_code_loc(self, addr: int) -> Optional[str]:
        assert addr > 0x400000 and addr < 0x400000 + 4*1024*1024*1024
        if addr not in self.code_loc_map:
            return f"<code_loc:unk>"
        else:
            return self.code_loc_map[addr]

    def add_mem_loc(self, addr: int, check_range: bool = True) -> str:
        if check_range:
            assert addr >= 0x400000 and addr <= 0x400000 + 4*1024*1024*1024, \
                f"Invalid memory location: {hex(addr)}"
        if addr not in self.mem_loc_map:
            self.mem_loc_map[addr] = f"<mem_loc:{len(self.mem_loc_map)}>"
        return self.mem_loc_map[addr]

    def get_mem_loc(self, addr: int) -> Optional[str]:
        assert addr > 0x400000 and addr < 0x400000 + 4*1024*1024*1024
        if addr not in self.mem_loc_map:
            return f"<mem_loc:unk>"
        else:
            return self.mem_loc_map[addr]

    def normalize_var_ident(self, ident: str, is_func_arg: bool) -> str:
        # return f"<vid:{ident}>"
        # Temporary disable the normalization
        if is_func_arg:
            if ident not in self.arg_var_ident_map:
                self.arg_var_ident_map[ident] = f"<arg_vid:{len(self.arg_var_ident_map)}>"
            return self.arg_var_ident_map[ident]
        else:
            if ident not in self.var_ident_map:
                self.var_ident_map[ident] = f"<vid:{len(self.var_ident_map)}>"
            return self.var_ident_map[ident]

    def add_string(self, s: str) -> str:
        """Record the string for BPE tokenizer training."""
        self.str_set.add(s)
    
    def tokenize_string(self, s: str) -> str:
        """Tokenize the string with the BPE tokenizer."""
        if Config.TOKENIZER is None:
            self.add_string(s[:Config.STR_TRUNCATE_LEN])
            return [s]
        else:
            return Config.TOKENIZER.encode(s[:Config.STR_TRUNCATE_LEN]).tokens[:Config.STR_TOKENS_MAX_LEN]
