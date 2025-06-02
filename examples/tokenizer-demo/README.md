# Tokenizer Demo

This example demonstrates the advanced tokenization API for llama.cpp.

## Features

The tokenizer API provides:

- **Basic tokenization**: Split text into tokens with customizable options
- **Detailed tokenization**: Get comprehensive token information including positions and types
- **Batch processing**: Tokenize multiple texts efficiently
- **Text chunking**: Split long texts into token-limited chunks
- **Token utilities**: Check token types, get text representations, estimate counts

## Building

```bash
mkdir build && cd build
cmake .. -DLLAMA_TOKENIZER=ON
make tokenizer-demo
```

## Usage

### Basic Usage

```bash
# Interactive mode
./tokenizer-demo -m /path/to/model.gguf

# Tokenize specific text
./tokenizer-demo -m /path/to/model.gguf -t "Hello, world!"

# Tokenize text from file
./tokenizer-demo -m /path/to/model.gguf -f input.txt
```

### Advanced Options

```bash
# Show detailed token information
./tokenizer-demo -m model.gguf -t "Hello world" --detailed

# Add special tokens (BOS/EOS)
./tokenizer-demo -m model.gguf -t "Hello world" --add-special

# Remove whitespace-only tokens
./tokenizer-demo -m model.gguf -t "Hello   world" --remove-whitespace

# Limit maximum tokens
./tokenizer-demo -m model.gguf -t "Long text..." --max-tokens 10

# Split into chunks
./tokenizer-demo -m model.gguf -t "Very long text..." --chunk-size 5

# Batch processing demo
./tokenizer-demo -m model.gguf --batch
```

## API Usage Examples

### Basic Tokenization

```cpp
#include "llama-tokenizer.h"

// Load model
llama_model * model = llama_load_model_from_file("model.gguf", params);

// Basic tokenization
llama_token_split_result result = llama_tokenize_split(
    model, "Hello, world!", -1);

if (result.success) {
    printf("Tokens: %d\n", result.n_tokens);
    for (int i = 0; i < result.n_tokens; i++) {
        printf("Token %d: ID=%d Text=\"%s\"\n", 
               i, result.tokens[i], result.token_texts[i]);
    }
}

llama_token_split_result_free(&result);
```

### Advanced Tokenization

```cpp
// Custom options
llama_token_split_options options = llama_token_split_options_default();
options.add_special = true;
options.remove_whitespace_tokens = true;
options.max_tokens = 100;

// Detailed tokenization
llama_token_split_detailed detailed = llama_tokenize_split_detailed(
    model, text, text_len, &options);

if (detailed.success) {
    for (int i = 0; i < detailed.n_tokens; i++) {
        llama_token_info * info = &detailed.token_infos[i];
        printf("Token %d: ID=%d Text=\"%s\" Pos=%d-%d Type=%s\n",
               i, info->token_id, info->text, 
               info->start_pos, info->end_pos,
               info->is_special ? "special" : "normal");
    }
}

llama_token_split_detailed_free(&detailed);
```

### Batch Processing

```cpp
const char * texts[] = {
    "First text",
    "Second text", 
    "Third text"
};

llama_token_split_result * results = llama_tokenize_split_batch(
    model, texts, nullptr, 3, &options);

for (int i = 0; i < 3; i++) {
    printf("Text %d: %d tokens\n", i, results[i].n_tokens);
}

llama_token_split_batch_free(results, 3);
```

### Text Chunking

```cpp
int32_t n_chunks = 0;
llama_token_split_result * chunks = llama_split_text_by_token_limit(
    model, long_text, -1, 512, &n_chunks);

printf("Split into %d chunks\n", n_chunks);

for (int i = 0; i < n_chunks; i++) {
    printf("Chunk %d: %d tokens\n", i, chunks[i].n_tokens);
    
    // Convert back to text
    char buffer[2048];
    int len = llama_tokens_merge_to_text(
        model, chunks[i].tokens, chunks[i].n_tokens,
        buffer, sizeof(buffer), false);
    
    printf("Text: %.*s\n", len, buffer);
}

llama_token_split_batch_free(chunks, n_chunks);
```

### Utility Functions

```cpp
// Estimate token count without full tokenization
int32_t estimated = llama_estimate_token_count(model, text, text_len);

// Check token properties
bool is_special = llama_token_is_special(model, token);
bool is_whitespace = llama_token_is_whitespace(model, token);
const char * type = llama_token_get_type_name(model, token);

// Get token text
char buffer[256];
int len = llama_token_get_text(model, token, buffer, sizeof(buffer));
```

## Output Examples

### Basic Output
```
=== Basic Tokenization Results ===
Success: Yes
Number of tokens: 4

   0: ID=  8279 Text="Hello" Len=5 Type=normal
   1: ID=    11 Text="," Len=1 Type=normal
   2: ID=  1917 Text=" world" Len=6 Type=normal
   3: ID=    33 Text="!" Len=1 Type=normal
```

### Detailed Output
```
=== Detailed Tokenization Results ===
Success: Yes
Original text length: 13
Number of tokens: 4

   0: ID=  8279 Text="Hello" Len=5 Pos=0-5
   1: ID=    11 Text="," Len=1 Pos=5-6
   2: ID=  1917 Text=" world" Len=6 Pos=6-12
   3: ID=    33 Text="!" Len=1 Pos=12-13
```

## Error Handling

All functions return success/failure status and error messages:

```cpp
llama_token_split_result result = llama_tokenize_split(model, text, -1);

if (!result.success) {
    printf("Error: %s\n", result.error_message ? result.error_message : "Unknown error");
}

// Always free results
llama_token_split_result_free(&result);
```

## Memory Management

- All `*_free()` functions must be called to prevent memory leaks
- Results can be safely freed even on error
- Batch results require special batch free functions
- All strings are null-terminated and safe to use with standard C functions