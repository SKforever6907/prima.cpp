#!/usr/bin/env python3
"""
Python bindings for the advanced llama tokenizer API.

This module provides a Python interface to the advanced tokenization
features implemented in llama-tokenizer.h/cpp.
"""

import ctypes
import os
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path

# Load the llama library
def _load_llama_library():
    """Load the llama shared library."""
    # Try different possible library names and paths
    library_names = [
        "libllama.so",
        "libllama.dylib", 
        "llama.dll",
        "libllama.so.0"
    ]
    
    # Try current directory and common library paths
    search_paths = [
        ".",
        "../build/src",
        "../build",
        "/usr/local/lib",
        "/usr/lib"
    ]
    
    for path in search_paths:
        for name in library_names:
            lib_path = os.path.join(path, name)
            if os.path.exists(lib_path):
                try:
                    return ctypes.CDLL(lib_path)
                except OSError:
                    continue
    
    raise RuntimeError("Could not find llama library. Please build llama.cpp first.")

# Load library
_lib = _load_llama_library()

# Define C structures
class LlamaTokenSplitOptions(ctypes.Structure):
    _fields_ = [
        ("add_special", ctypes.c_bool),
        ("parse_special", ctypes.c_bool),
        ("include_token_texts", ctypes.c_bool),
        ("remove_whitespace_tokens", ctypes.c_bool),
        ("max_tokens", ctypes.c_int32),
        ("split_by_words", ctypes.c_bool),
        ("preserve_spaces", ctypes.c_bool),
    ]

class LlamaTokenSplitResult(ctypes.Structure):
    _fields_ = [
        ("tokens", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_int32),
        ("token_texts", ctypes.POINTER(ctypes.c_char_p)),
        ("token_lengths", ctypes.POINTER(ctypes.c_int32)),
        ("success", ctypes.c_bool),
        ("error_message", ctypes.c_char_p),
    ]

class LlamaTokenInfo(ctypes.Structure):
    _fields_ = [
        ("token_id", ctypes.c_int32),
        ("text", ctypes.c_char_p),
        ("text_length", ctypes.c_int32),
        ("start_pos", ctypes.c_int32),
        ("end_pos", ctypes.c_int32),
        ("is_special", ctypes.c_bool),
        ("is_whitespace", ctypes.c_bool),
    ]

class LlamaTokenSplitDetailed(ctypes.Structure):
    _fields_ = [
        ("token_infos", ctypes.POINTER(LlamaTokenInfo)),
        ("n_tokens", ctypes.c_int32),
        ("original_text", ctypes.c_char_p),
        ("text_length", ctypes.c_int32),
        ("success", ctypes.c_bool),
        ("error_message", ctypes.c_char_p),
    ]

# Define function prototypes
_lib.llama_tokenize_split.argtypes = [
    ctypes.c_void_p,  # model
    ctypes.c_char_p,  # text
    ctypes.c_int32,   # text_len
]
_lib.llama_tokenize_split.restype = LlamaTokenSplitResult

_lib.llama_tokenize_split_ex.argtypes = [
    ctypes.c_void_p,  # model
    ctypes.c_char_p,  # text
    ctypes.c_int32,   # text_len
    ctypes.POINTER(LlamaTokenSplitOptions),  # options
]
_lib.llama_tokenize_split_ex.restype = LlamaTokenSplitResult

_lib.llama_tokenize_split_detailed.argtypes = [
    ctypes.c_void_p,  # model
    ctypes.c_char_p,  # text
    ctypes.c_int32,   # text_len
    ctypes.POINTER(LlamaTokenSplitOptions),  # options
]
_lib.llama_tokenize_split_detailed.restype = LlamaTokenSplitDetailed

_lib.llama_token_split_options_default.argtypes = []
_lib.llama_token_split_options_default.restype = LlamaTokenSplitOptions

_lib.llama_token_split_result_free.argtypes = [ctypes.POINTER(LlamaTokenSplitResult)]
_lib.llama_token_split_result_free.restype = None

_lib.llama_token_split_detailed_free.argtypes = [ctypes.POINTER(LlamaTokenSplitDetailed)]
_lib.llama_token_split_detailed_free.restype = None

_lib.llama_estimate_token_count.argtypes = [
    ctypes.c_void_p,  # model
    ctypes.c_char_p,  # text
    ctypes.c_int32,   # text_len
]
_lib.llama_estimate_token_count.restype = ctypes.c_int32

# Python wrapper classes
@dataclass
class TokenInfo:
    """Information about a single token."""
    token_id: int
    text: str
    text_length: int
    start_pos: int
    end_pos: int
    is_special: bool
    is_whitespace: bool

@dataclass
class TokenizationOptions:
    """Options for tokenization."""
    add_special: bool = False
    parse_special: bool = False
    include_token_texts: bool = True
    remove_whitespace_tokens: bool = False
    max_tokens: int = 0  # 0 = no limit
    split_by_words: bool = False
    preserve_spaces: bool = True

class TokenizationResult:
    """Result of tokenization operation."""
    
    def __init__(self, success: bool, tokens: List[int] = None, 
                 token_texts: List[str] = None, error_message: str = None):
        self.success = success
        self.tokens = tokens or []
        self.token_texts = token_texts or []
        self.error_message = error_message
    
    @property
    def n_tokens(self) -> int:
        return len(self.tokens)
    
    def __len__(self) -> int:
        return self.n_tokens
    
    def __iter__(self):
        return iter(self.tokens)
    
    def __getitem__(self, index):
        return self.tokens[index]

class DetailedTokenizationResult:
    """Detailed result of tokenization operation."""
    
    def __init__(self, success: bool, token_infos: List[TokenInfo] = None,
                 original_text: str = "", error_message: str = None):
        self.success = success
        self.token_infos = token_infos or []
        self.original_text = original_text
        self.error_message = error_message
    
    @property
    def n_tokens(self) -> int:
        return len(self.token_infos)
    
    @property
    def tokens(self) -> List[int]:
        return [info.token_id for info in self.token_infos]
    
    @property
    def token_texts(self) -> List[str]:
        return [info.text for info in self.token_infos]
    
    def __len__(self) -> int:
        return self.n_tokens
    
    def __iter__(self):
        return iter(self.token_infos)
    
    def __getitem__(self, index):
        return self.token_infos[index]

class LlamaTokenizer:
    """Advanced tokenizer for llama models."""
    
    def __init__(self, model_ptr: int):
        """Initialize tokenizer with model pointer.
        
        Args:
            model_ptr: Pointer to loaded llama model (from llama.cpp)
        """
        self.model_ptr = ctypes.c_void_p(model_ptr)
    
    def tokenize(self, text: str, options: TokenizationOptions = None) -> TokenizationResult:
        """Tokenize text with basic options.
        
        Args:
            text: Text to tokenize
            options: Tokenization options
            
        Returns:
            TokenizationResult with tokens and metadata
        """
        if options is None:
            options = TokenizationOptions()
        
        # Convert options to C structure
        c_options = LlamaTokenSplitOptions(
            add_special=options.add_special,
            parse_special=options.parse_special,
            include_token_texts=options.include_token_texts,
            remove_whitespace_tokens=options.remove_whitespace_tokens,
            max_tokens=options.max_tokens,
            split_by_words=options.split_by_words,
            preserve_spaces=options.preserve_spaces,
        )
        
        # Call C function
        text_bytes = text.encode('utf-8')
        result = _lib.llama_tokenize_split_ex(
            self.model_ptr,
            text_bytes,
            len(text_bytes),
            ctypes.byref(c_options)
        )
        
        try:
            if not result.success:
                error_msg = result.error_message.decode('utf-8') if result.error_message else "Unknown error"
                return TokenizationResult(False, error_message=error_msg)
            
            # Extract tokens
            tokens = []
            token_texts = []
            
            for i in range(result.n_tokens):
                tokens.append(result.tokens[i])
                
                if result.token_texts and options.include_token_texts:
                    token_text = result.token_texts[i].decode('utf-8') if result.token_texts[i] else ""
                    token_texts.append(token_text)
            
            return TokenizationResult(True, tokens, token_texts if token_texts else None)
        
        finally:
            # Free C memory
            _lib.llama_token_split_result_free(ctypes.byref(result))
    
    def tokenize_detailed(self, text: str, options: TokenizationOptions = None) -> DetailedTokenizationResult:
        """Tokenize text with detailed token information.
        
        Args:
            text: Text to tokenize
            options: Tokenization options
            
        Returns:
            DetailedTokenizationResult with comprehensive token info
        """
        if options is None:
            options = TokenizationOptions()
        
        # Convert options to C structure
        c_options = LlamaTokenSplitOptions(
            add_special=options.add_special,
            parse_special=options.parse_special,
            include_token_texts=options.include_token_texts,
            remove_whitespace_tokens=options.remove_whitespace_tokens,
            max_tokens=options.max_tokens,
            split_by_words=options.split_by_words,
            preserve_spaces=options.preserve_spaces,
        )
        
        # Call C function
        text_bytes = text.encode('utf-8')
        result = _lib.llama_tokenize_split_detailed(
            self.model_ptr,
            text_bytes,
            len(text_bytes),
            ctypes.byref(c_options)
        )
        
        try:
            if not result.success:
                error_msg = result.error_message.decode('utf-8') if result.error_message else "Unknown error"
                return DetailedTokenizationResult(False, error_message=error_msg)
            
            # Extract token information
            token_infos = []
            
            for i in range(result.n_tokens):
                info = result.token_infos[i]
                token_info = TokenInfo(
                    token_id=info.token_id,
                    text=info.text.decode('utf-8') if info.text else "",
                    text_length=info.text_length,
                    start_pos=info.start_pos,
                    end_pos=info.end_pos,
                    is_special=info.is_special,
                    is_whitespace=info.is_whitespace,
                )
                token_infos.append(token_info)
            
            original_text = result.original_text.decode('utf-8') if result.original_text else text
            
            return DetailedTokenizationResult(True, token_infos, original_text)
        
        finally:
            # Free C memory
            _lib.llama_token_split_detailed_free(ctypes.byref(result))
    
    def estimate_token_count(self, text: str) -> int:
        """Estimate token count without full tokenization.
        
        Args:
            text: Text to estimate
            
        Returns:
            Estimated number of tokens
        """
        text_bytes = text.encode('utf-8')
        return _lib.llama_estimate_token_count(
            self.model_ptr,
            text_bytes,
            len(text_bytes)
        )

# Convenience functions
def create_tokenizer(model_ptr: int) -> LlamaTokenizer:
    """Create a tokenizer instance.
    
    Args:
        model_ptr: Pointer to loaded llama model
        
    Returns:
        LlamaTokenizer instance
    """
    return LlamaTokenizer(model_ptr)

def tokenize_text(model_ptr: int, text: str, **kwargs) -> TokenizationResult:
    """Quick tokenization function.
    
    Args:
        model_ptr: Pointer to loaded llama model
        text: Text to tokenize
        **kwargs: Options for TokenizationOptions
        
    Returns:
        TokenizationResult
    """
    tokenizer = LlamaTokenizer(model_ptr)
    options = TokenizationOptions(**kwargs)
    return tokenizer.tokenize(text, options)

def tokenize_detailed(model_ptr: int, text: str, **kwargs) -> DetailedTokenizationResult:
    """Quick detailed tokenization function.
    
    Args:
        model_ptr: Pointer to loaded llama model
        text: Text to tokenize
        **kwargs: Options for TokenizationOptions
        
    Returns:
        DetailedTokenizationResult
    """
    tokenizer = LlamaTokenizer(model_ptr)
    options = TokenizationOptions(**kwargs)
    return tokenizer.tokenize_detailed(text, options)

# Example usage
if __name__ == "__main__":
    print("Llama Tokenizer Python Bindings")
    print("This module provides Python access to the advanced tokenization API.")
    print("Usage:")
    print("  from llama_tokenizer import LlamaTokenizer, TokenizationOptions")
    print("  tokenizer = LlamaTokenizer(model_ptr)")
    print("  result = tokenizer.tokenize('Hello, world!')")
    print("  print(f'Tokens: {result.tokens}')")
    print("  print(f'Token texts: {result.token_texts}')")