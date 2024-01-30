import os
from tokenizers import Tokenizer


class _Config:

    # This is used as the token length for the variable tokens
    NUM_VAR_TOKENS = 4
    
    # This is used to truncate the string when recording it for BPE tokenizer training
    STR_TRUNCATE_LEN = 100
    # This is used to limit the length of the string tokens
    STR_TOKENS_MAX_LEN = 10

    # This is used as the byteorder when checking if a number is a valid string
    ENDIANNESS = "little"
    
    TOKENIZER: "Tokenizer" = None

    def load_tokenizer(self, tokenizer_fpath: str):
        """Load the BPE tokenizer from the given file path.

        Args:
            tokenizer_fpath (str): The file path of the BPE tokenizer.
        """
        self.TOKENIZER = Tokenizer.from_file(tokenizer_fpath)
        print("Loaded tokenizer from", tokenizer_fpath)


Config = _Config()
Config.load_tokenizer(os.path.join(os.path.dirname(__file__), "ail_bpe_tokenizer.json"))
