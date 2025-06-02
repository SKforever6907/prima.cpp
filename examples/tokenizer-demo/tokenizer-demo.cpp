#include "llama.h"
#include "llama-tokenizer.h"
#include "common.h"
#include "arg.h"

#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <fstream>
#include <cstdlib>

static void print_usage(const char * program_name) {
    std::cout << "Usage: " << program_name << " [options]\n";
    std::cout << "Options:\n";
    std::cout << "  -m, --model PATH        Model file path (required)\n";
    std::cout << "  -t, --text TEXT         Text to tokenize (default: interactive mode)\n";
    std::cout << "  -f, --file PATH         Read text from file\n";
    std::cout << "  --add-special           Add special tokens (BOS/EOS)\n";
    std::cout << "  --parse-special         Parse special/control tokens\n";
    std::cout << "  --remove-whitespace     Remove whitespace-only tokens\n";
    std::cout << "  --max-tokens N          Maximum number of tokens\n";
    std::cout << "  --detailed              Show detailed token information\n";
    std::cout << "  --batch                 Batch processing demo\n";
    std::cout << "  --chunk-size N          Split text into chunks of N tokens\n";
    std::cout << "  -h, --help              Show this help message\n";
}

static void print_token_info(const llama_token_info & info, int index) {
    std::cout << std::setw(4) << index << ": ";
    std::cout << "ID=" << std::setw(6) << info.token_id;
    std::cout << " Text=\"" << (info.text ? info.text : "") << "\"";
    std::cout << " Len=" << info.text_length;
    std::cout << " Pos=" << info.start_pos << "-" << info.end_pos;
    
    if (info.is_special) std::cout << " [SPECIAL]";
    if (info.is_whitespace) std::cout << " [WHITESPACE]";
    
    std::cout << "\n";
}

static void print_basic_tokens(const llama_model * model, const llama_token_split_result & result) {
    std::cout << "\n=== Basic Tokenization Results ===\n";
    std::cout << "Success: " << (result.success ? "Yes" : "No") << "\n";
    
    if (!result.success) {
        std::cout << "Error: " << (result.error_message ? result.error_message : "Unknown error") << "\n";
        return;
    }
    
    std::cout << "Number of tokens: " << result.n_tokens << "\n\n";
    
    for (int32_t i = 0; i < result.n_tokens; i++) {
        std::cout << std::setw(4) << i << ": ";
        std::cout << "ID=" << std::setw(6) << result.tokens[i];
        
        if (result.token_texts && result.token_texts[i]) {
            std::cout << " Text=\"" << result.token_texts[i] << "\"";
            std::cout << " Len=" << result.token_lengths[i];
        } else {
            // Get token text manually
            char buffer[256];
            int32_t len = llama_token_get_text(model, result.tokens[i], buffer, sizeof(buffer));
            if (len > 0) {
                std::cout << " Text=\"" << std::string(buffer, len) << "\"";
                std::cout << " Len=" << len;
            }
        }
        
        // Show token type
        const char * type = llama_token_get_type_name(model, result.tokens[i]);
        std::cout << " Type=" << type;
        
        std::cout << "\n";
    }
}

static void print_detailed_tokens(const llama_token_split_detailed & result) {
    std::cout << "\n=== Detailed Tokenization Results ===\n";
    std::cout << "Success: " << (result.success ? "Yes" : "No") << "\n";
    
    if (!result.success) {
        std::cout << "Error: " << (result.error_message ? result.error_message : "Unknown error") << "\n";
        return;
    }
    
    std::cout << "Original text length: " << result.text_length << "\n";
    std::cout << "Number of tokens: " << result.n_tokens << "\n\n";
    
    for (int32_t i = 0; i < result.n_tokens; i++) {
        print_token_info(result.token_infos[i], i);
    }
}

static void demo_batch_processing(const llama_model * model) {
    std::cout << "\n=== Batch Processing Demo ===\n";
    
    const char * texts[] = {
        "Hello, world!",
        "This is a test.",
        "Batch processing is efficient.",
        "Multiple texts at once."
    };
    
    int32_t n_texts = sizeof(texts) / sizeof(texts[0]);
    llama_token_split_options options = llama_token_split_options_default();
    
    llama_token_split_result * results = llama_tokenize_split_batch(
        model, texts, nullptr, n_texts, &options);
    
    if (results) {
        for (int32_t i = 0; i < n_texts; i++) {
            std::cout << "\nText " << i << ": \"" << texts[i] << "\"\n";
            std::cout << "Tokens: " << results[i].n_tokens << "\n";
            
            if (results[i].success) {
                for (int32_t j = 0; j < results[i].n_tokens; j++) {
                    if (results[i].token_texts && results[i].token_texts[j]) {
                        std::cout << "  " << j << ": \"" << results[i].token_texts[j] << "\"\n";
                    }
                }
            }
        }
        
        llama_token_split_batch_free(results, n_texts);
    }
}

static void demo_text_chunking(const llama_model * model, const std::string & text, int32_t chunk_size) {
    std::cout << "\n=== Text Chunking Demo ===\n";
    std::cout << "Chunk size: " << chunk_size << " tokens\n";
    
    int32_t n_chunks = 0;
    llama_token_split_result * chunks = llama_split_text_by_token_limit(
        model, text.c_str(), static_cast<int32_t>(text.length()), chunk_size, &n_chunks);
    
    if (chunks) {
        std::cout << "Number of chunks: " << n_chunks << "\n";
        
        for (int32_t i = 0; i < n_chunks; i++) {
            std::cout << "\nChunk " << i << ": " << chunks[i].n_tokens << " tokens\n";
            
            if (chunks[i].success && chunks[i].tokens) {
                // Convert tokens back to text
                char buffer[1024];
                int32_t len = llama_tokens_merge_to_text(
                    model, chunks[i].tokens, chunks[i].n_tokens, 
                    buffer, sizeof(buffer), false);
                
                if (len > 0) {
                    std::cout << "Text: \"" << std::string(buffer, len) << "\"\n";
                }
            }
        }
        
        llama_token_split_batch_free(chunks, n_chunks);
    }
}

int main(int argc, char ** argv) {
    gpt_params params;
    std::string input_text;
    std::string input_file;
    bool detailed_mode = false;
    bool batch_demo = false;
    int32_t chunk_size = 0;
    
    llama_token_split_options options = llama_token_split_options_default();
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-m" || arg == "--model") {
            if (i + 1 < argc) {
                params.model = argv[++i];
            } else {
                std::cerr << "Error: --model requires a path\n";
                return 1;
            }
        } else if (arg == "-t" || arg == "--text") {
            if (i + 1 < argc) {
                input_text = argv[++i];
            } else {
                std::cerr << "Error: --text requires text input\n";
                return 1;
            }
        } else if (arg == "-f" || arg == "--file") {
            if (i + 1 < argc) {
                input_file = argv[++i];
            } else {
                std::cerr << "Error: --file requires a path\n";
                return 1;
            }
        } else if (arg == "--add-special") {
            options.add_special = true;
        } else if (arg == "--parse-special") {
            options.parse_special = true;
        } else if (arg == "--remove-whitespace") {
            options.remove_whitespace_tokens = true;
        } else if (arg == "--max-tokens") {
            if (i + 1 < argc) {
                options.max_tokens = std::atoi(argv[++i]);
            } else {
                std::cerr << "Error: --max-tokens requires a number\n";
                return 1;
            }
        } else if (arg == "--detailed") {
            detailed_mode = true;
        } else if (arg == "--batch") {
            batch_demo = true;
        } else if (arg == "--chunk-size") {
            if (i + 1 < argc) {
                chunk_size = std::atoi(argv[++i]);
            } else {
                std::cerr << "Error: --chunk-size requires a number\n";
                return 1;
            }
        }
    }
    
    if (params.model.empty()) {
        std::cerr << "Error: Model path is required. Use -m or --model.\n";
        print_usage(argv[0]);
        return 1;
    }
    
    // Initialize llama
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);
    
    // Load model
    llama_model_params model_params = llama_model_default_params();
    llama_model * model = llama_load_model_from_file(params.model.c_str(), model_params);
    
    if (!model) {
        std::cerr << "Error: Failed to load model from " << params.model << "\n";
        llama_backend_free();
        return 1;
    }
    
    std::cout << "Model loaded successfully!\n";
    std::cout << "Vocab type: " << llama_vocab_type(model) << "\n";
    std::cout << "Vocab size: " << llama_n_vocab(model) << "\n";
    
    // Read input text
    if (!input_file.empty()) {
        std::ifstream file(input_file);
        if (file.is_open()) {
            std::string line;
            while (std::getline(file, line)) {
                input_text += line + "\n";
            }
            file.close();
        } else {
            std::cerr << "Error: Could not open file " << input_file << "\n";
            llama_free_model(model);
            llama_backend_free();
            return 1;
        }
    }
    
    // Interactive mode if no text provided
    if (input_text.empty() && !batch_demo) {
        std::cout << "\nEnter text to tokenize (or 'quit' to exit):\n";
        std::getline(std::cin, input_text);
        
        if (input_text == "quit") {
            llama_free_model(model);
            llama_backend_free();
            return 0;
        }
    }
    
    // Run demos
    if (batch_demo) {
        demo_batch_processing(model);
    }
    
    if (!input_text.empty()) {
        std::cout << "\nInput text: \"" << input_text << "\"\n";
        std::cout << "Text length: " << input_text.length() << " characters\n";
        
        // Estimate token count
        int32_t estimated_tokens = llama_estimate_token_count(
            model, input_text.c_str(), static_cast<int32_t>(input_text.length()));
        std::cout << "Estimated tokens: " << estimated_tokens << "\n";
        
        if (detailed_mode) {
            // Detailed tokenization
            llama_token_split_detailed result = llama_tokenize_split_detailed(
                model, input_text.c_str(), static_cast<int32_t>(input_text.length()), &options);
            
            print_detailed_tokens(result);
            llama_token_split_detailed_free(&result);
        } else {
            // Basic tokenization
            llama_token_split_result result = llama_tokenize_split_ex(
                model, input_text.c_str(), static_cast<int32_t>(input_text.length()), &options);
            
            print_basic_tokens(model, result);
            llama_token_split_result_free(&result);
        }
        
        // Text chunking demo
        if (chunk_size > 0) {
            demo_text_chunking(model, input_text, chunk_size);
        }
    }
    
    // Cleanup
    llama_free_model(model);
    llama_backend_free();
    
    return 0;
}