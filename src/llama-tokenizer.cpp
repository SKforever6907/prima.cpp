#include "llama-tokenizer.h"
#include "llama.h"
#include "llama-vocab.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
#include <cctype>

// Helper function to check if a string is whitespace only
static bool is_whitespace_only(const char * text, int32_t len) {
    for (int32_t i = 0; i < len; i++) {
        if (!std::isspace(static_cast<unsigned char>(text[i]))) {
            return false;
        }
    }
    return len > 0;
}

// Helper function to duplicate string
static char * duplicate_string(const char * src, int32_t len = -1) {
    if (!src) return nullptr;
    
    if (len < 0) {
        len = static_cast<int32_t>(strlen(src));
    }
    
    char * dst = static_cast<char *>(malloc(len + 1));
    if (!dst) return nullptr;
    
    memcpy(dst, src, len);
    dst[len] = '\0';
    return dst;
}

// Helper function to create error result
static llama_token_split_result create_error_result(const char * error_msg) {
    llama_token_split_result result = {};
    result.success = false;
    result.error_message = duplicate_string(error_msg);
    return result;
}

// Helper function to create detailed error result
static llama_token_split_detailed create_detailed_error_result(const char * error_msg) {
    llama_token_split_detailed result = {};
    result.success = false;
    result.error_message = duplicate_string(error_msg);
    return result;
}

//
// Basic Token Splitting API Implementation
//

llama_token_split_result llama_tokenize_split(
    const struct llama_model * model,
    const char               * text,
    int32_t                    text_len) {
    
    llama_token_split_options options = llama_token_split_options_default();
    return llama_tokenize_split_ex(model, text, text_len, &options);
}

llama_token_split_result llama_tokenize_split_ex(
    const struct llama_model        * model,
    const char                      * text,
    int32_t                           text_len,
    const llama_token_split_options * options) {
    
    if (!model || !text || !options) {
        return create_error_result("Invalid input parameters");
    }
    
    if (text_len < 0) {
        text_len = static_cast<int32_t>(strlen(text));
    }
    
    if (text_len == 0) {
        llama_token_split_result result = {};
        result.success = true;
        result.n_tokens = 0;
        return result;
    }
    
    // Determine maximum possible tokens (conservative estimate)
    int32_t max_tokens = options->max_tokens > 0 ? options->max_tokens : text_len + 10;
    
    // Allocate token buffer
    std::vector<llama_token> tokens(max_tokens);
    
    // Tokenize the text
    int32_t n_tokens = llama_tokenize(
        model,
        text,
        text_len,
        tokens.data(),
        max_tokens,
        options->add_special,
        options->parse_special
    );
    
    if (n_tokens < 0) {
        return create_error_result("Tokenization failed: buffer too small");
    }
    
    // Resize to actual number of tokens
    tokens.resize(n_tokens);
    
    // Filter tokens if requested
    std::vector<llama_token> filtered_tokens;
    std::vector<std::string> token_texts;
    
    for (int32_t i = 0; i < n_tokens; i++) {
        llama_token token = tokens[i];
        
        // Get token text if needed
        std::string token_text;
        if (options->include_token_texts || options->remove_whitespace_tokens) {
            char buffer[256];
            int32_t len = llama_token_to_piece(model, token, buffer, sizeof(buffer), 0, false);
            if (len > 0) {
                token_text = std::string(buffer, len);
            }
        }
        
        // Check if we should skip whitespace tokens
        if (options->remove_whitespace_tokens && 
            is_whitespace_only(token_text.c_str(), static_cast<int32_t>(token_text.length()))) {
            continue;
        }
        
        filtered_tokens.push_back(token);
        if (options->include_token_texts) {
            token_texts.push_back(token_text);
        }
    }
    
    // Create result structure
    llama_token_split_result result = {};
    result.success = true;
    result.n_tokens = static_cast<int32_t>(filtered_tokens.size());
    
    if (result.n_tokens > 0) {
        // Allocate and copy tokens
        result.tokens = static_cast<llama_token *>(malloc(result.n_tokens * sizeof(llama_token)));
        if (!result.tokens) {
            return create_error_result("Memory allocation failed");
        }
        memcpy(result.tokens, filtered_tokens.data(), result.n_tokens * sizeof(llama_token));
        
        // Allocate and copy token texts if requested
        if (options->include_token_texts) {
            result.token_texts = static_cast<char **>(malloc(result.n_tokens * sizeof(char *)));
            result.token_lengths = static_cast<int32_t *>(malloc(result.n_tokens * sizeof(int32_t)));
            
            if (!result.token_texts || !result.token_lengths) {
                free(result.tokens);
                free(result.token_texts);
                free(result.token_lengths);
                return create_error_result("Memory allocation failed");
            }
            
            for (int32_t i = 0; i < result.n_tokens; i++) {
                result.token_texts[i] = duplicate_string(token_texts[i].c_str());
                result.token_lengths[i] = static_cast<int32_t>(token_texts[i].length());
            }
        }
    }
    
    return result;
}

llama_token_split_detailed llama_tokenize_split_detailed(
    const struct llama_model        * model,
    const char                      * text,
    int32_t                           text_len,
    const llama_token_split_options * options) {
    
    if (!model || !text || !options) {
        return create_detailed_error_result("Invalid input parameters");
    }
    
    if (text_len < 0) {
        text_len = static_cast<int32_t>(strlen(text));
    }
    
    // First get basic tokenization
    llama_token_split_result basic_result = llama_tokenize_split_ex(model, text, text_len, options);
    if (!basic_result.success) {
        llama_token_split_detailed detailed_result = {};
        detailed_result.success = false;
        detailed_result.error_message = basic_result.error_message;
        basic_result.error_message = nullptr; // Transfer ownership
        llama_token_split_result_free(&basic_result);
        return detailed_result;
    }
    
    // Create detailed result
    llama_token_split_detailed result = {};
    result.success = true;
    result.n_tokens = basic_result.n_tokens;
    result.original_text = duplicate_string(text, text_len);
    result.text_length = text_len;
    
    if (result.n_tokens > 0) {
        result.token_infos = static_cast<llama_token_info *>(
            calloc(result.n_tokens, sizeof(llama_token_info)));
        
        if (!result.token_infos) {
            llama_token_split_result_free(&basic_result);
            return create_detailed_error_result("Memory allocation failed");
        }
        
        // Fill in detailed token information
        int32_t current_pos = 0;
        for (int32_t i = 0; i < result.n_tokens; i++) {
            llama_token_info * info = &result.token_infos[i];
            info->token_id = basic_result.tokens[i];
            
            // Get token text
            char buffer[256];
            int32_t len = llama_token_to_piece(model, info->token_id, buffer, sizeof(buffer), 0, false);
            if (len > 0) {
                info->text = duplicate_string(buffer, len);
                info->text_length = len;
                info->is_whitespace = is_whitespace_only(buffer, len);
            }
            
            // Estimate positions (this is approximate)
            info->start_pos = current_pos;
            info->end_pos = current_pos + info->text_length;
            current_pos = info->end_pos;
            
            // Check if special token
            info->is_special = llama_token_is_special(model, info->token_id);
        }
    }
    
    llama_token_split_result_free(&basic_result);
    return result;
}

//
// Utility Functions Implementation
//

llama_token_split_options llama_token_split_options_default(void) {
    llama_token_split_options options = {};
    options.add_special = false;
    options.parse_special = false;
    options.include_token_texts = true;
    options.remove_whitespace_tokens = false;
    options.max_tokens = 0; // No limit
    options.split_by_words = false;
    options.preserve_spaces = true;
    return options;
}

void llama_token_split_result_free(llama_token_split_result * result) {
    if (!result) return;
    
    free(result->tokens);
    free(result->error_message);
    free(result->token_lengths);
    
    if (result->token_texts) {
        for (int32_t i = 0; i < result->n_tokens; i++) {
            free(result->token_texts[i]);
        }
        free(result->token_texts);
    }
    
    memset(result, 0, sizeof(*result));
}

void llama_token_split_detailed_free(llama_token_split_detailed * result) {
    if (!result) return;
    
    free(result->original_text);
    free(result->error_message);
    
    if (result->token_infos) {
        for (int32_t i = 0; i < result->n_tokens; i++) {
            free(result->token_infos[i].text);
        }
        free(result->token_infos);
    }
    
    memset(result, 0, sizeof(*result));
}

int32_t llama_token_get_text(
    const struct llama_model * model,
    llama_token                token,
    char                     * buffer,
    int32_t                    buffer_size) {
    
    if (!model || !buffer || buffer_size <= 0) {
        return -1;
    }
    
    return llama_token_to_piece(model, token, buffer, buffer_size, 0, false);
}

bool llama_token_is_special(
    const struct llama_model * model,
    llama_token                token) {
    
    if (!model) return false;
    
    // Check common special tokens
    return (token == llama_token_bos(model) ||
            token == llama_token_eos(model) ||
            token == llama_token_pad(model) ||
            token == llama_token_unk(model) ||
            token == llama_token_cls(model) ||
            token == llama_token_sep(model) ||
            token == llama_token_nl(model) ||
            token == llama_token_prefix(model) ||
            token == llama_token_middle(model) ||
            token == llama_token_suffix(model) ||
            token == llama_token_eot(model));
}

bool llama_token_is_whitespace(
    const struct llama_model * model,
    llama_token                token) {
    
    if (!model) return false;
    
    char buffer[256];
    int32_t len = llama_token_to_piece(model, token, buffer, sizeof(buffer), 0, false);
    
    return len > 0 && is_whitespace_only(buffer, len);
}

const char * llama_token_get_type_name(
    const struct llama_model * model,
    llama_token                token) {
    
    if (!model) return "unknown";
    
    if (llama_token_is_special(model, token)) {
        if (token == llama_token_bos(model)) return "bos";
        if (token == llama_token_eos(model)) return "eos";
        if (token == llama_token_pad(model)) return "pad";
        if (token == llama_token_unk(model)) return "unk";
        if (token == llama_token_cls(model)) return "cls";
        if (token == llama_token_sep(model)) return "sep";
        if (token == llama_token_nl(model)) return "newline";
        if (token == llama_token_prefix(model)) return "prefix";
        if (token == llama_token_middle(model)) return "middle";
        if (token == llama_token_suffix(model)) return "suffix";
        if (token == llama_token_eot(model)) return "eot";
        return "special";
    }
    
    if (llama_token_is_whitespace(model, token)) {
        return "whitespace";
    }
    
    return "normal";
}

//
// Batch Processing Implementation
//

llama_token_split_result * llama_tokenize_split_batch(
    const struct llama_model        * model,
    const char                     ** texts,
    const int32_t                   * text_lens,
    int32_t                           n_texts,
    const llama_token_split_options * options) {
    
    if (!model || !texts || !options || n_texts <= 0) {
        return nullptr;
    }
    
    llama_token_split_result * results = static_cast<llama_token_split_result *>(
        calloc(n_texts, sizeof(llama_token_split_result)));
    
    if (!results) return nullptr;
    
    for (int32_t i = 0; i < n_texts; i++) {
        int32_t text_len = text_lens ? text_lens[i] : -1;
        results[i] = llama_tokenize_split_ex(model, texts[i], text_len, options);
    }
    
    return results;
}

void llama_token_split_batch_free(
    llama_token_split_result * results,
    int32_t                    n_results) {
    
    if (!results) return;
    
    for (int32_t i = 0; i < n_results; i++) {
        llama_token_split_result_free(&results[i]);
    }
    
    free(results);
}

//
// Advanced Features Implementation
//

llama_token_split_detailed llama_tokenize_split_with_positions(
    const struct llama_model * model,
    const char               * text,
    int32_t                    text_len,
    bool                       byte_level_positions) {
    
    llama_token_split_options options = llama_token_split_options_default();
    return llama_tokenize_split_detailed(model, text, text_len, &options);
}

int32_t llama_tokens_merge_to_text(
    const struct llama_model * model,
    const llama_token        * tokens,
    int32_t                    n_tokens,
    char                     * output_buffer,
    int32_t                    buffer_size,
    bool                       remove_special) {
    
    if (!model || !tokens || !output_buffer || buffer_size <= 0 || n_tokens <= 0) {
        return -1;
    }
    
    return llama_detokenize(model, tokens, n_tokens, output_buffer, buffer_size, remove_special, false);
}

int32_t llama_estimate_token_count(
    const struct llama_model * model,
    const char               * text,
    int32_t                    text_len) {
    
    if (!model || !text) return -1;
    
    if (text_len < 0) {
        text_len = static_cast<int32_t>(strlen(text));
    }
    
    // Simple estimation: average 3-4 characters per token for most languages
    // This is a rough estimate and actual count may vary
    return (text_len + 2) / 3;
}

llama_token_split_result * llama_split_text_by_token_limit(
    const struct llama_model * model,
    const char               * text,
    int32_t                    text_len,
    int32_t                    max_tokens_per_chunk,
    int32_t                  * n_chunks) {
    
    if (!model || !text || max_tokens_per_chunk <= 0 || !n_chunks) {
        if (n_chunks) *n_chunks = 0;
        return nullptr;
    }
    
    if (text_len < 0) {
        text_len = static_cast<int32_t>(strlen(text));
    }
    
    // First, tokenize the entire text
    llama_token_split_options options = llama_token_split_options_default();
    llama_token_split_result full_result = llama_tokenize_split_ex(model, text, text_len, &options);
    
    if (!full_result.success) {
        *n_chunks = 0;
        llama_token_split_result_free(&full_result);
        return nullptr;
    }
    
    // Calculate number of chunks needed
    *n_chunks = (full_result.n_tokens + max_tokens_per_chunk - 1) / max_tokens_per_chunk;
    
    if (*n_chunks == 0) {
        llama_token_split_result_free(&full_result);
        return nullptr;
    }
    
    // Allocate results array
    llama_token_split_result * results = static_cast<llama_token_split_result *>(
        calloc(*n_chunks, sizeof(llama_token_split_result)));
    
    if (!results) {
        *n_chunks = 0;
        llama_token_split_result_free(&full_result);
        return nullptr;
    }
    
    // Split tokens into chunks
    for (int32_t chunk = 0; chunk < *n_chunks; chunk++) {
        int32_t start_token = chunk * max_tokens_per_chunk;
        int32_t end_token = std::min(start_token + max_tokens_per_chunk, full_result.n_tokens);
        int32_t chunk_size = end_token - start_token;
        
        results[chunk].success = true;
        results[chunk].n_tokens = chunk_size;
        
        if (chunk_size > 0) {
            results[chunk].tokens = static_cast<llama_token *>(malloc(chunk_size * sizeof(llama_token)));
            if (results[chunk].tokens) {
                memcpy(results[chunk].tokens, 
                       full_result.tokens + start_token, 
                       chunk_size * sizeof(llama_token));
            } else {
                results[chunk].success = false;
                results[chunk].error_message = duplicate_string("Memory allocation failed");
            }
        }
    }
    
    llama_token_split_result_free(&full_result);
    return results;
}