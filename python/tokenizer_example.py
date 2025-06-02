#!/usr/bin/env python3
"""
Example usage of the llama tokenizer Python bindings.

This script demonstrates how to use the advanced tokenization API
from Python.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from llama_tokenizer import (
        LlamaTokenizer, 
        TokenizationOptions, 
        tokenize_text, 
        tokenize_detailed
    )
except ImportError as e:
    print(f"Error importing llama_tokenizer: {e}")
    print("Make sure llama.cpp is built with LLAMA_TOKENIZER=ON")
    sys.exit(1)

def demo_basic_tokenization(model_ptr: int):
    """Demonstrate basic tokenization features."""
    print("=== Basic Tokenization Demo ===")
    
    tokenizer = LlamaTokenizer(model_ptr)
    
    # Test texts
    texts = [
        "Hello, world!",
        "This is a test of the tokenization system.",
        "How many tokens will this sentence have?",
        "Special characters: @#$%^&*()",
        "Numbers: 123 456 789",
        "Mixed: Hello123 World456!"
    ]
    
    for text in texts:
        print(f"\nText: '{text}'")
        
        # Estimate token count first
        estimated = tokenizer.estimate_token_count(text)
        print(f"Estimated tokens: {estimated}")
        
        # Basic tokenization
        result = tokenizer.tokenize(text)
        
        if result.success:
            print(f"Actual tokens: {result.n_tokens}")
            print(f"Token IDs: {result.tokens}")
            if result.token_texts:
                print("Token texts:")
                for i, (token_id, token_text) in enumerate(zip(result.tokens, result.token_texts)):
                    print(f"  {i:2d}: {token_id:6d} -> '{token_text}'")
        else:
            print(f"Error: {result.error_message}")

def demo_detailed_tokenization(model_ptr: int):
    """Demonstrate detailed tokenization features."""
    print("\n=== Detailed Tokenization Demo ===")
    
    tokenizer = LlamaTokenizer(model_ptr)
    
    text = "Hello, world! This is a test."
    print(f"Text: '{text}'")
    
    # Detailed tokenization
    result = tokenizer.tokenize_detailed(text)
    
    if result.success:
        print(f"Original text length: {len(result.original_text)} characters")
        print(f"Number of tokens: {result.n_tokens}")
        print("\nDetailed token information:")
        
        for i, info in enumerate(result.token_infos):
            flags = []
            if info.is_special:
                flags.append("SPECIAL")
            if info.is_whitespace:
                flags.append("WHITESPACE")
            
            flag_str = f" [{', '.join(flags)}]" if flags else ""
            
            print(f"  {i:2d}: ID={info.token_id:6d} "
                  f"Text='{info.text}' "
                  f"Len={info.text_length} "
                  f"Pos={info.start_pos}-{info.end_pos}"
                  f"{flag_str}")
    else:
        print(f"Error: {result.error_message}")

def demo_tokenization_options(model_ptr: int):
    """Demonstrate different tokenization options."""
    print("\n=== Tokenization Options Demo ===")
    
    tokenizer = LlamaTokenizer(model_ptr)
    text = "  Hello,   world!  "  # Text with extra whitespace
    
    print(f"Text: '{text}'")
    
    # Test different options
    options_list = [
        ("Default", TokenizationOptions()),
        ("Add special tokens", TokenizationOptions(add_special=True)),
        ("Remove whitespace", TokenizationOptions(remove_whitespace_tokens=True)),
        ("Max 5 tokens", TokenizationOptions(max_tokens=5)),
        ("No token texts", TokenizationOptions(include_token_texts=False)),
    ]
    
    for name, options in options_list:
        print(f"\n{name}:")
        result = tokenizer.tokenize(text, options)
        
        if result.success:
            print(f"  Tokens: {result.n_tokens}")
            print(f"  Token IDs: {result.tokens}")
            if result.token_texts:
                print(f"  Token texts: {result.token_texts}")
        else:
            print(f"  Error: {result.error_message}")

def demo_convenience_functions(model_ptr: int):
    """Demonstrate convenience functions."""
    print("\n=== Convenience Functions Demo ===")
    
    text = "Quick tokenization test"
    print(f"Text: '{text}'")
    
    # Using convenience functions
    result = tokenize_text(model_ptr, text, include_token_texts=True)
    print(f"Quick tokenize: {result.n_tokens} tokens")
    print(f"Tokens: {result.tokens}")
    
    detailed = tokenize_detailed(model_ptr, text, add_special=True)
    print(f"Quick detailed: {detailed.n_tokens} tokens")
    for i, info in enumerate(detailed.token_infos):
        print(f"  {i}: '{info.text}' (special: {info.is_special})")

def main():
    """Main demo function."""
    print("Llama Tokenizer Python Bindings Demo")
    print("=====================================")
    
    # Note: In a real application, you would load a model using llama.cpp
    # and get the model pointer. For this demo, we'll use a placeholder.
    
    # This is just a placeholder - in real usage you would have:
    # import llama_cpp
    # model = llama_cpp.Llama(model_path="path/to/model.gguf")
    # model_ptr = model._model.model  # Get the actual model pointer
    
    model_ptr = None  # Placeholder
    
    if model_ptr is None:
        print("Error: No model loaded.")
        print("To run this demo, you need to:")
        print("1. Build llama.cpp with LLAMA_TOKENIZER=ON")
        print("2. Load a model and get its pointer")
        print("3. Pass the model pointer to these functions")
        print("\nExample integration with llama-cpp-python:")
        print("```python")
        print("import llama_cpp")
        print("from llama_tokenizer import LlamaTokenizer")
        print("")
        print("# Load model")
        print("model = llama_cpp.Llama(model_path='model.gguf')")
        print("model_ptr = model._model.model")
        print("")
        print("# Create tokenizer")
        print("tokenizer = LlamaTokenizer(model_ptr)")
        print("result = tokenizer.tokenize('Hello, world!')")
        print("print(f'Tokens: {result.tokens}')")
        print("```")
        return
    
    try:
        demo_basic_tokenization(model_ptr)
        demo_detailed_tokenization(model_ptr)
        demo_tokenization_options(model_ptr)
        demo_convenience_functions(model_ptr)
        
        print("\n=== Demo Complete ===")
        print("All tokenization features demonstrated successfully!")
        
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()