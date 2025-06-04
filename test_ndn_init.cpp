#include <iostream>
#include <cstdint>

// Mock llama_context structure for testing
struct llama_context {
    void* ndn_ctx;
};

// Include NDN initialization function
extern "C" {
    void llama_init_sockets(struct llama_context * ctx, uint32_t n_world, uint32_t my_rank);
}

int main(int argc, char* argv[]) {
    std::cout << "🧪 Testing NDN initialization directly..." << std::endl;
    
    // Create a mock context
    llama_context ctx = {0};
    
    // Test single node (should skip)
    std::cout << "\n--- Test 1: Single node mode ---" << std::endl;
    llama_init_sockets(&ctx, 1, 0);
    
    // Test multi-node (should initialize NDN)
    std::cout << "\n--- Test 2: Multi-node mode ---" << std::endl;
    llama_init_sockets(&ctx, 2, 0);
    
    std::cout << "\n✅ NDN initialization test completed!" << std::endl;
    return 0;
}