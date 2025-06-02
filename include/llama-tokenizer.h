#ifndef LLAMA_TOKENIZER_H
#define LLAMA_TOKENIZER_H

#include "llama.h"
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Token splitting result structure
typedef struct llama_token_split_result {
    llama_token * tokens;           // Array of tokens
    int32_t       n_tokens;         // Number of tokens
    char       ** token_texts;      // Array of token text representations
    int32_t    * token_lengths;     // Length of each token text
    bool          success;          // Whether tokenization was successful
    char        * error_message;    // Error message if failed
} llama_token_split_result;

// Token splitting options
typedef struct llama_token_split_options {
    bool add_special;               // Add special tokens (BOS/EOS)
    bool parse_special;             // Parse special/control tokens
    bool include_token_texts;       // Include text representation of tokens
    bool remove_whitespace_tokens;  // Filter out pure whitespace tokens
    int32_t max_tokens;             // Maximum number of tokens (0 = no limit)
    bool split_by_words;            // Split into word-level tokens when possible
    bool preserve_spaces;           // Preserve leading/trailing spaces
} llama_token_split_options;

// Token information structure
typedef struct llama_token_info {
    llama_token token_id;           // Token ID
    char      * text;               // Token text representation
    int32_t     text_length;        // Length of token text
    int32_t     start_pos;          // Start position in original text
    int32_t     end_pos;            // End position in original text
    bool        is_special;         // Whether this is a special token
    bool        is_whitespace;      // Whether this is a whitespace-only token
} llama_token_info;

// Advanced token splitting result with detailed information
typedef struct llama_token_split_detailed {
    llama_token_info * token_infos; // Array of detailed token information
    int32_t            n_tokens;    // Number of tokens
    char             * original_text; // Copy of original input text
    int32_t            text_length; // Length of original text
    bool               success;     // Whether tokenization was successful
    char             * error_message; // Error message if failed
} llama_token_split_detailed;

//
// Basic Token Splitting API
//

// Split text into tokens using default options
LLAMA_API llama_token_split_result llama_tokenize_split(
    const struct llama_model * model,
    const char               * text,
    int32_t                    text_len);

// Split text into tokens with custom options
LLAMA_API llama_token_split_result llama_tokenize_split_ex(
    const struct llama_model        * model,
    const char                      * text,
    int32_t                           text_len,
    const llama_token_split_options * options);

// Split text into tokens with detailed information
LLAMA_API llama_token_split_detailed llama_tokenize_split_detailed(
    const struct llama_model        * model,
    const char                      * text,
    int32_t                           text_len,
    const llama_token_split_options * options);

//
// Utility Functions
//

// Create default token splitting options
LLAMA_API llama_token_split_options llama_token_split_options_default(void);

// Free token split result memory
LLAMA_API void llama_token_split_result_free(llama_token_split_result * result);

// Free detailed token split result memory
LLAMA_API void llama_token_split_detailed_free(llama_token_split_detailed * result);

// Get token text representation
LLAMA_API int32_t llama_token_get_text(
    const struct llama_model * model,
    llama_token                token,
    char                     * buffer,
    int32_t                    buffer_size);

// Check if token is special token
LLAMA_API bool llama_token_is_special(
    const struct llama_model * model,
    llama_token                token);

// Check if token is whitespace-only
LLAMA_API bool llama_token_is_whitespace(
    const struct llama_model * model,
    llama_token                token);

// Get token type information
LLAMA_API const char * llama_token_get_type_name(
    const struct llama_model * model,
    llama_token                token);

//
// Batch Processing API
//

// Split multiple texts in batch
LLAMA_API llama_token_split_result * llama_tokenize_split_batch(
    const struct llama_model        * model,
    const char                     ** texts,
    const int32_t                   * text_lens,
    int32_t                           n_texts,
    const llama_token_split_options * options);

// Free batch results
LLAMA_API void llama_token_split_batch_free(
    llama_token_split_result * results,
    int32_t                    n_results);

//
// Advanced Features
//

// Split text with token position mapping
LLAMA_API llama_token_split_detailed llama_tokenize_split_with_positions(
    const struct llama_model * model,
    const char               * text,
    int32_t                    text_len,
    bool                       byte_level_positions);

// Merge tokens back to text
LLAMA_API int32_t llama_tokens_merge_to_text(
    const struct llama_model * model,
    const llama_token        * tokens,
    int32_t                    n_tokens,
    char                     * output_buffer,
    int32_t                    buffer_size,
    bool                       remove_special);

// Count tokens without full tokenization (estimate)
LLAMA_API int32_t llama_estimate_token_count(
    const struct llama_model * model,
    const char               * text,
    int32_t                    text_len);

// Split text into chunks with maximum token count
LLAMA_API llama_token_split_result * llama_split_text_by_token_limit(
    const struct llama_model * model,
    const char               * text,
    int32_t                    text_len,
    int32_t                    max_tokens_per_chunk,
    int32_t                  * n_chunks);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_TOKENIZER_H