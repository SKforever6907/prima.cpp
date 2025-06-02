// NDN implementation for prima.cpp distributed inference
// Replaces ZeroMQ with Named Data Networking

#include "ndn_prima.h"
#include "llama-impl.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <chrono>
#include <random>

#ifdef USE_NDN_INSTEAD_OF_ZMQ

namespace ndn_prima {

// NDN Context Implementation
ndn_context::ndn_context(const std::string& prefix) 
    : scheduler(face.getIoService()), prefix_base(prefix) {
    session_id = std::chrono::duration_cast<std::chrono::seconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

ndn_context::~ndn_context() {
    stop();
}

void ndn_context::start() {
    if (running.load()) {
        return;
    }
    
    running = true;
    face_thread = std::thread([this]() {
        while (running.load()) {
            try {
                face.processEvents(ndn::time::milliseconds(10));
            } catch (const std::exception& e) {
                std::cerr << "NDN Face processing error: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    });
    
    std::cout << "NDN context started for rank " << rank << std::endl;
}

void ndn_context::stop() {
    if (!running.load()) {
        return;
    }
    
    running = false;
    
    if (face_thread.joinable()) {
        face_thread.join();
    }
    
    // Unregister all prefixes
    for (auto& handle : registered_prefixes) {
        handle.unregister();
    }
    registered_prefixes.clear();
    
    std::cout << "NDN context stopped for rank " << rank << std::endl;
}

void ndn_context::cache_data(const std::string& name, const ndn::Data& data) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    
    if (data_cache.size() >= max_cache_size) {
        // Simple LRU: remove oldest entry
        auto oldest = data_cache.begin();
        data_cache.erase(oldest);
    }
    
    data_cache[name] = data;
}

bool ndn_context::get_cached_data(const std::string& name, ndn::Data& data) {
    std::lock_guard<std::mutex> lock(cache_mutex);
    auto it = data_cache.find(name);
    if (it != data_cache.end()) {
        data = it->second;
        return true;
    }
    return false;
}

void ndn_context::cleanup_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    data_cache.clear();
}

void ndn_context::print_stats() const {
    std::cout << "=== NDN Statistics for Rank " << rank << " ===" << std::endl;
    std::cout << "Interests sent: " << interests_sent.load() << std::endl;
    std::cout << "Interests received: " << interests_received.load() << std::endl;
    std::cout << "Data sent: " << data_sent.load() << std::endl;
    std::cout << "Data received: " << data_received.load() << std::endl;
    std::cout << "Timeouts: " << timeouts.load() << std::endl;
    std::cout << "NACKs: " << nacks.load() << std::endl;
    std::cout << "Cache size: " << data_cache.size() << std::endl;
    std::cout << "=========================================" << std::endl;
}

void ndn_context::reset_stats() {
    interests_sent = 0;
    interests_received = 0;
    data_sent = 0;
    data_received = 0;
    timeouts = 0;
    nacks = 0;
}

// Name building functions
std::string build_meta_name(const ndn_context* ctx, uint64_t sequence) {
    std::ostringstream oss;
    oss << ctx->prefix_base << "/session-" << ctx->session_id 
        << "/rank-" << ctx->rank << "/meta/seq-" << sequence;
    return oss.str();
}

std::string build_tensor_name(const ndn_context* ctx, uint32_t layer, uint64_t sequence) {
    std::ostringstream oss;
    oss << ctx->prefix_base << "/session-" << ctx->session_id 
        << "/rank-" << ctx->rank << "/tensor/layer-" << layer << "/seq-" << sequence;
    return oss.str();
}

std::string build_device_info_name(const ndn_context* ctx) {
    std::ostringstream oss;
    oss << ctx->prefix_base << "/session-" << ctx->session_id 
        << "/rank-" << ctx->rank << "/device-info";
    return oss.str();
}

std::string build_broadcast_name(const ndn_context* ctx, const std::string& type) {
    std::ostringstream oss;
    oss << ctx->prefix_base << "/session-" << ctx->session_id 
        << "/broadcast/" << type;
    return oss.str();
}

std::string build_kv_cache_name(const ndn_context* ctx, const std::string& operation, uint64_t sequence) {
    std::ostringstream oss;
    oss << ctx->prefix_base << "/session-" << ctx->session_id 
        << "/rank-" << ctx->rank << "/kv-cache/" << operation << "/seq-" << sequence;
    return oss.str();
}

// Serialization functions
void serialize_meta_to_buffer(const struct sync_meta* meta, ndn::encoding::BufferStream& os) {
    // Write sync_meta fields in order
    os.write(reinterpret_cast<const char*>(&meta->n_tokens), sizeof(meta->n_tokens));
    os.write(reinterpret_cast<const char*>(&meta->all_pos_0), sizeof(meta->all_pos_0));
    os.write(reinterpret_cast<const char*>(&meta->all_pos_1), sizeof(meta->all_pos_1));
    os.write(reinterpret_cast<const char*>(&meta->seq_id), sizeof(meta->seq_id));
    os.write(reinterpret_cast<const char*>(&meta->kv_cache_clear), sizeof(meta->kv_cache_clear));
    os.write(reinterpret_cast<const char*>(&meta->kv_cache_seq_rm_n), sizeof(meta->kv_cache_seq_rm_n));
    
    // Write variable-length arrays
    for (int i = 0; i < meta->kv_cache_seq_rm_n; ++i) {
        os.write(reinterpret_cast<const char*>(&meta->kv_cache_seq_rm[i]), sizeof(meta->kv_cache_seq_rm[i]));
    }
    
    os.write(reinterpret_cast<const char*>(&meta->kv_cache_seq_cp_n), sizeof(meta->kv_cache_seq_cp_n));
    for (int i = 0; i < meta->kv_cache_seq_cp_n; ++i) {
        os.write(reinterpret_cast<const char*>(&meta->kv_cache_seq_cp[i]), sizeof(meta->kv_cache_seq_cp[i]));
    }
    
    os.write(reinterpret_cast<const char*>(&meta->kv_cache_seq_add_n), sizeof(meta->kv_cache_seq_add_n));
    for (int i = 0; i < meta->kv_cache_seq_add_n; ++i) {
        os.write(reinterpret_cast<const char*>(&meta->kv_cache_seq_add[i]), sizeof(meta->kv_cache_seq_add[i]));
    }
    
    os.write(reinterpret_cast<const char*>(&meta->kv_cache_seq_div_n), sizeof(meta->kv_cache_seq_div_n));
    for (int i = 0; i < meta->kv_cache_seq_div_n; ++i) {
        os.write(reinterpret_cast<const char*>(&meta->kv_cache_seq_div[i]), sizeof(meta->kv_cache_seq_div[i]));
    }
}

void deserialize_meta_from_buffer(const uint8_t* data, size_t size, struct sync_meta* meta) {
    size_t offset = 0;
    
    if (offset + sizeof(meta->n_tokens) > size) return;
    std::memcpy(&meta->n_tokens, data + offset, sizeof(meta->n_tokens));
    offset += sizeof(meta->n_tokens);
    
    if (offset + sizeof(meta->all_pos_0) > size) return;
    std::memcpy(&meta->all_pos_0, data + offset, sizeof(meta->all_pos_0));
    offset += sizeof(meta->all_pos_0);
    
    if (offset + sizeof(meta->all_pos_1) > size) return;
    std::memcpy(&meta->all_pos_1, data + offset, sizeof(meta->all_pos_1));
    offset += sizeof(meta->all_pos_1);
    
    if (offset + sizeof(meta->seq_id) > size) return;
    std::memcpy(&meta->seq_id, data + offset, sizeof(meta->seq_id));
    offset += sizeof(meta->seq_id);
    
    if (offset + sizeof(meta->kv_cache_clear) > size) return;
    std::memcpy(&meta->kv_cache_clear, data + offset, sizeof(meta->kv_cache_clear));
    offset += sizeof(meta->kv_cache_clear);
    
    if (offset + sizeof(meta->kv_cache_seq_rm_n) > size) return;
    std::memcpy(&meta->kv_cache_seq_rm_n, data + offset, sizeof(meta->kv_cache_seq_rm_n));
    offset += sizeof(meta->kv_cache_seq_rm_n);
    
    // Read variable-length arrays
    for (int i = 0; i < meta->kv_cache_seq_rm_n && i < LLAMA_MAX_SEQ_RM; ++i) {
        if (offset + sizeof(meta->kv_cache_seq_rm[i]) > size) return;
        std::memcpy(&meta->kv_cache_seq_rm[i], data + offset, sizeof(meta->kv_cache_seq_rm[i]));
        offset += sizeof(meta->kv_cache_seq_rm[i]);
    }
    
    if (offset + sizeof(meta->kv_cache_seq_cp_n) > size) return;
    std::memcpy(&meta->kv_cache_seq_cp_n, data + offset, sizeof(meta->kv_cache_seq_cp_n));
    offset += sizeof(meta->kv_cache_seq_cp_n);
    
    for (int i = 0; i < meta->kv_cache_seq_cp_n && i < LLAMA_MAX_SEQ_CP; ++i) {
        if (offset + sizeof(meta->kv_cache_seq_cp[i]) > size) return;
        std::memcpy(&meta->kv_cache_seq_cp[i], data + offset, sizeof(meta->kv_cache_seq_cp[i]));
        offset += sizeof(meta->kv_cache_seq_cp[i]);
    }
    
    if (offset + sizeof(meta->kv_cache_seq_add_n) > size) return;
    std::memcpy(&meta->kv_cache_seq_add_n, data + offset, sizeof(meta->kv_cache_seq_add_n));
    offset += sizeof(meta->kv_cache_seq_add_n);
    
    for (int i = 0; i < meta->kv_cache_seq_add_n && i < LLAMA_MAX_SEQ_ADD; ++i) {
        if (offset + sizeof(meta->kv_cache_seq_add[i]) > size) return;
        std::memcpy(&meta->kv_cache_seq_add[i], data + offset, sizeof(meta->kv_cache_seq_add[i]));
        offset += sizeof(meta->kv_cache_seq_add[i]);
    }
    
    if (offset + sizeof(meta->kv_cache_seq_div_n) > size) return;
    std::memcpy(&meta->kv_cache_seq_div_n, data + offset, sizeof(meta->kv_cache_seq_div_n));
    offset += sizeof(meta->kv_cache_seq_div_n);
    
    for (int i = 0; i < meta->kv_cache_seq_div_n && i < LLAMA_MAX_SEQ_DIV; ++i) {
        if (offset + sizeof(meta->kv_cache_seq_div[i]) > size) return;
        std::memcpy(&meta->kv_cache_seq_div[i], data + offset, sizeof(meta->kv_cache_seq_div[i]));
        offset += sizeof(meta->kv_cache_seq_div[i]);
    }
}

void serialize_device_info_to_buffer(const struct device_info* info, ndn::encoding::BufferStream& os) {
    os.write(reinterpret_cast<const char*>(&info->rank), sizeof(info->rank));
    os.write(reinterpret_cast<const char*>(&info->n_gpu_layers), sizeof(info->n_gpu_layers));
    os.write(reinterpret_cast<const char*>(&info->n_layer_window), sizeof(info->n_layer_window));
    os.write(reinterpret_cast<const char*>(&info->memory_total), sizeof(info->memory_total));
    os.write(reinterpret_cast<const char*>(&info->memory_free), sizeof(info->memory_free));
    os.write(reinterpret_cast<const char*>(&info->compute_capability), sizeof(info->compute_capability));
}

void deserialize_device_info_from_buffer(const uint8_t* data, size_t size, struct device_info* info) {
    size_t offset = 0;
    
    if (offset + sizeof(info->rank) > size) return;
    std::memcpy(&info->rank, data + offset, sizeof(info->rank));
    offset += sizeof(info->rank);
    
    if (offset + sizeof(info->n_gpu_layers) > size) return;
    std::memcpy(&info->n_gpu_layers, data + offset, sizeof(info->n_gpu_layers));
    offset += sizeof(info->n_gpu_layers);
    
    if (offset + sizeof(info->n_layer_window) > size) return;
    std::memcpy(&info->n_layer_window, data + offset, sizeof(info->n_layer_window));
    offset += sizeof(info->n_layer_window);
    
    if (offset + sizeof(info->memory_total) > size) return;
    std::memcpy(&info->memory_total, data + offset, sizeof(info->memory_total));
    offset += sizeof(info->memory_total);
    
    if (offset + sizeof(info->memory_free) > size) return;
    std::memcpy(&info->memory_free, data + offset, sizeof(info->memory_free));
    offset += sizeof(info->memory_free);
    
    if (offset + sizeof(info->compute_capability) > size) return;
    std::memcpy(&info->compute_capability, data + offset, sizeof(info->compute_capability));
    offset += sizeof(info->compute_capability);
}

void serialize_startup_args_to_buffer(const struct startup_args* args, ndn::encoding::BufferStream& os) {
    os.write(reinterpret_cast<const char*>(&args->n_world), sizeof(args->n_world));
    os.write(reinterpret_cast<const char*>(&args->n_layer_window), sizeof(args->n_layer_window));
    os.write(reinterpret_cast<const char*>(&args->n_gpu_layers), sizeof(args->n_gpu_layers));
    os.write(reinterpret_cast<const char*>(&args->batch_size), sizeof(args->batch_size));
    os.write(reinterpret_cast<const char*>(&args->ctx_size), sizeof(args->ctx_size));
}

void deserialize_startup_args_from_buffer(const uint8_t* data, size_t size, struct startup_args* args) {
    size_t offset = 0;
    
    if (offset + sizeof(args->n_world) > size) return;
    std::memcpy(&args->n_world, data + offset, sizeof(args->n_world));
    offset += sizeof(args->n_world);
    
    if (offset + sizeof(args->n_layer_window) > size) return;
    std::memcpy(&args->n_layer_window, data + offset, sizeof(args->n_layer_window));
    offset += sizeof(args->n_layer_window);
    
    if (offset + sizeof(args->n_gpu_layers) > size) return;
    std::memcpy(&args->n_gpu_layers, data + offset, sizeof(args->n_gpu_layers));
    offset += sizeof(args->n_gpu_layers);
    
    if (offset + sizeof(args->batch_size) > size) return;
    std::memcpy(&args->batch_size, data + offset, sizeof(args->batch_size));
    offset += sizeof(args->batch_size);
    
    if (offset + sizeof(args->ctx_size) > size) return;
    std::memcpy(&args->ctx_size, data + offset, sizeof(args->ctx_size));
    offset += sizeof(args->ctx_size);
}

// Core communication functions
void send_meta_ndn(ndn_context* ctx, struct sync_meta* meta) {
    uint64_t seq = ctx->sequence_number.fetch_add(1);
    std::string name = build_meta_name(ctx, seq);
    
    ndn::Interest interest(name);
    interest.setMustBeFresh(true);
    interest.setInterestLifetime(ndn::time::seconds(5));
    interest.setCanBePrefix(false);
    
    // Serialize meta data
    ndn::encoding::BufferStream os;
    serialize_meta_to_buffer(meta, os);
    auto buffer = os.buf();
    interest.setApplicationParameters(buffer->data(), buffer->size());
    
    // Set operation as pending
    {
        std::lock_guard<std::mutex> lock(ctx->sync_mutex);
        ctx->pending_operations[name] = true;
    }
    
    ctx->interests_sent++;
    
    ctx->face.expressInterest(interest,
        [ctx, name](const ndn::Interest&, const ndn::Data& data) {
            ctx->data_received++;
            {
                std::lock_guard<std::mutex> lock(ctx->sync_mutex);
                ctx->pending_operations[name] = false;
            }
            ctx->sync_cv.notify_all();
        },
        [ctx, name](const ndn::Interest&, const ndn::lp::Nack& nack) {
            ctx->nacks++;
            std::cerr << "Received NACK for " << name << ": " 
                     << ndn::lp::getNackReasonString(nack.getReason()) << std::endl;
            {
                std::lock_guard<std::mutex> lock(ctx->sync_mutex);
                ctx->pending_operations[name] = false;
            }
            ctx->sync_cv.notify_all();
        },
        [ctx, name](const ndn::Interest&) {
            ctx->timeouts++;
            std::cerr << "Interest timeout for " << name << std::endl;
            {
                std::lock_guard<std::mutex> lock(ctx->sync_mutex);
                ctx->pending_operations[name] = false;
            }
            ctx->sync_cv.notify_all();
        });
}

void send_tensor_ndn(ndn_context* ctx, struct llama_ubatch* ubatch, struct input_tensors* tensors) {
    uint64_t seq = ctx->sequence_number.fetch_add(1);
    std::string name = build_tensor_name(ctx, 0, seq); // Layer 0 for now
    
    ndn::Interest interest(name);
    interest.setMustBeFresh(true);
    interest.setInterestLifetime(ndn::time::seconds(10)); // Longer timeout for large data
    interest.setCanBePrefix(false);
    
    // Serialize tensor data (simplified - would need proper tensor serialization)
    ndn::encoding::BufferStream os;
    // This is a placeholder - actual tensor serialization would be more complex
    size_t tensor_size = tensors->n_tokens * sizeof(float);
    os.write(reinterpret_cast<const char*>(&tensor_size), sizeof(tensor_size));
    os.write(reinterpret_cast<const char*>(tensors->data), tensor_size);
    
    auto buffer = os.buf();
    interest.setApplicationParameters(buffer->data(), buffer->size());
    
    ctx->interests_sent++;
    
    ctx->face.expressInterest(interest,
        [ctx](const ndn::Interest&, const ndn::Data& data) {
            ctx->data_received++;
        },
        [ctx](const ndn::Interest&, const ndn::lp::Nack& nack) {
            ctx->nacks++;
        },
        [ctx](const ndn::Interest&) {
            ctx->timeouts++;
        });
}

void send_device_info_ndn(ndn_context* ctx, struct device_info* dev_info) {
    std::string name = build_device_info_name(ctx);
    
    ndn::Interest interest(name);
    interest.setMustBeFresh(true);
    interest.setInterestLifetime(ndn::time::seconds(10));
    interest.setCanBePrefix(false);
    
    ndn::encoding::BufferStream os;
    serialize_device_info_to_buffer(dev_info, os);
    auto buffer = os.buf();
    interest.setApplicationParameters(buffer->data(), buffer->size());
    
    ctx->interests_sent++;
    
    ctx->face.expressInterest(interest,
        [ctx](const ndn::Interest&, const ndn::Data& data) {
            ctx->data_received++;
        },
        [ctx](const ndn::Interest&, const ndn::lp::Nack& nack) {
            ctx->nacks++;
        },
        [ctx](const ndn::Interest&) {
            ctx->timeouts++;
        });
}

void broadcast_startup_args_ndn(ndn_context* ctx, struct startup_args* args) {
    std::string name = build_broadcast_name(ctx, "startup-args");
    
    ndn::Interest interest(name);
    interest.setMustBeFresh(true);
    interest.setInterestLifetime(ndn::time::seconds(15));
    interest.setCanBePrefix(false);
    
    ndn::encoding::BufferStream os;
    serialize_startup_args_to_buffer(args, os);
    auto buffer = os.buf();
    interest.setApplicationParameters(buffer->data(), buffer->size());
    
    ctx->interests_sent++;
    
    ctx->face.expressInterest(interest,
        [ctx](const ndn::Interest&, const ndn::Data& data) {
            ctx->data_received++;
        },
        [ctx](const ndn::Interest&, const ndn::lp::Nack& nack) {
            ctx->nacks++;
        },
        [ctx](const ndn::Interest&) {
            ctx->timeouts++;
        });
}

// Interest filter setup functions
void setup_meta_interest_filter(ndn_context* ctx) {
    std::string prefix = ctx->prefix_base + "/session-" + std::to_string(ctx->session_id) + "/rank-" + std::to_string(ctx->rank) + "/meta";
    
    auto handle = ctx->face.setInterestFilter(prefix,
        [ctx](const ndn::InterestFilter&, const ndn::Interest& interest) {
            ctx->interests_received++;
            
            if (!interest.hasApplicationParameters()) {
                return;
            }
            
            // Deserialize meta data
            auto params = interest.getApplicationParameters();
            struct sync_meta meta = {};
            deserialize_meta_from_buffer(params.data(), params.size(), &meta);
            
            // Call callback if set
            if (ctx->meta_callback) {
                ctx->meta_callback(&meta);
            }
            
            // Send ACK response
            ndn::Data data(interest.getName());
            std::string content = "ACK";
            data.setContent(ndn::encoding::makeStringBlock(ndn::tlv::Content, content));
            data.setFreshnessPeriod(ndn::time::seconds(1));
            ctx->keyChain.sign(data);
            ctx->face.put(data);
            ctx->data_sent++;
        },
        [](const ndn::Name& prefix, const std::string& reason) {
            std::cerr << "Failed to register prefix " << prefix.toUri() << ": " << reason << std::endl;
        });
        
    ctx->registered_prefixes.push_back(handle);
}

void setup_tensor_interest_filter(ndn_context* ctx) {
    std::string prefix = ctx->prefix_base + "/session-" + std::to_string(ctx->session_id) + "/rank-" + std::to_string(ctx->rank) + "/tensor";
    
    auto handle = ctx->face.setInterestFilter(prefix,
        [ctx](const ndn::InterestFilter&, const ndn::Interest& interest) {
            ctx->interests_received++;
            
            if (!interest.hasApplicationParameters()) {
                return;
            }
            
            // Deserialize tensor data (simplified)
            auto params = interest.getApplicationParameters();
            struct input_tensors tensors = {};
            // This would need proper tensor deserialization
            
            // Call callback if set
            if (ctx->tensor_callback) {
                ctx->tensor_callback(&tensors);
            }
            
            // Send ACK response
            ndn::Data data(interest.getName());
            std::string content = "ACK";
            data.setContent(ndn::encoding::makeStringBlock(ndn::tlv::Content, content));
            data.setFreshnessPeriod(ndn::time::seconds(1));
            ctx->keyChain.sign(data);
            ctx->face.put(data);
            ctx->data_sent++;
        },
        [](const ndn::Name& prefix, const std::string& reason) {
            std::cerr << "Failed to register prefix " << prefix.toUri() << ": " << reason << std::endl;
        });
        
    ctx->registered_prefixes.push_back(handle);
}

void setup_device_info_interest_filter(ndn_context* ctx) {
    std::string prefix = ctx->prefix_base + "/session-" + std::to_string(ctx->session_id) + "/rank-" + std::to_string(ctx->rank) + "/device-info";
    
    auto handle = ctx->face.setInterestFilter(prefix,
        [ctx](const ndn::InterestFilter&, const ndn::Interest& interest) {
            ctx->interests_received++;
            
            if (!interest.hasApplicationParameters()) {
                return;
            }
            
            // Deserialize device info
            auto params = interest.getApplicationParameters();
            struct device_info dev_info = {};
            deserialize_device_info_from_buffer(params.data(), params.size(), &dev_info);
            
            // Call callback if set
            if (ctx->device_info_callback) {
                ctx->device_info_callback(&dev_info);
            }
            
            // Send ACK response
            ndn::Data data(interest.getName());
            std::string content = "ACK";
            data.setContent(ndn::encoding::makeStringBlock(ndn::tlv::Content, content));
            data.setFreshnessPeriod(ndn::time::seconds(1));
            ctx->keyChain.sign(data);
            ctx->face.put(data);
            ctx->data_sent++;
        },
        [](const ndn::Name& prefix, const std::string& reason) {
            std::cerr << "Failed to register prefix " << prefix.toUri() << ": " << reason << std::endl;
        });
        
    ctx->registered_prefixes.push_back(handle);
}

void setup_broadcast_interest_filter(ndn_context* ctx) {
    std::string prefix = ctx->prefix_base + "/session-" + std::to_string(ctx->session_id) + "/broadcast";
    
    auto handle = ctx->face.setInterestFilter(prefix,
        [ctx](const ndn::InterestFilter&, const ndn::Interest& interest) {
            ctx->interests_received++;
            
            // Handle broadcast messages
            std::string name = interest.getName().toUri();
            
            // Send ACK response
            ndn::Data data(interest.getName());
            std::string content = "BROADCAST_ACK";
            data.setContent(ndn::encoding::makeStringBlock(ndn::tlv::Content, content));
            data.setFreshnessPeriod(ndn::time::seconds(5));
            ctx->keyChain.sign(data);
            ctx->face.put(data);
            ctx->data_sent++;
        },
        [](const ndn::Name& prefix, const std::string& reason) {
            std::cerr << "Failed to register prefix " << prefix.toUri() << ": " << reason << std::endl;
        });
        
    ctx->registered_prefixes.push_back(handle);
}

void setup_kv_cache_interest_filters(ndn_context* ctx) {
    std::string prefix = ctx->prefix_base + "/session-" + std::to_string(ctx->session_id) + "/rank-" + std::to_string(ctx->rank) + "/kv-cache";
    
    auto handle = ctx->face.setInterestFilter(prefix,
        [ctx](const ndn::InterestFilter&, const ndn::Interest& interest) {
            ctx->interests_received++;
            
            // Handle KV cache operations
            std::string name = interest.getName().toUri();
            
            // Send ACK response
            ndn::Data data(interest.getName());
            std::string content = "KV_CACHE_ACK";
            data.setContent(ndn::encoding::makeStringBlock(ndn::tlv::Content, content));
            data.setFreshnessPeriod(ndn::time::seconds(1));
            ctx->keyChain.sign(data);
            ctx->face.put(data);
            ctx->data_sent++;
        },
        [](const ndn::Name& prefix, const std::string& reason) {
            std::cerr << "Failed to register prefix " << prefix.toUri() << ": " << reason << std::endl;
        });
        
    ctx->registered_prefixes.push_back(handle);
}

// KV cache operations
void send_kv_cache_clear_ndn(ndn_context* ctx) {
    uint64_t seq = ctx->sequence_number.fetch_add(1);
    std::string name = build_kv_cache_name(ctx, "clear", seq);
    
    ndn::Interest interest(name);
    interest.setMustBeFresh(true);
    interest.setInterestLifetime(ndn::time::seconds(5));
    
    ctx->interests_sent++;
    
    ctx->face.expressInterest(interest,
        [ctx](const ndn::Interest&, const ndn::Data& data) {
            ctx->data_received++;
        },
        [ctx](const ndn::Interest&, const ndn::lp::Nack& nack) {
            ctx->nacks++;
        },
        [ctx](const ndn::Interest&) {
            ctx->timeouts++;
        });
}

void send_kv_cache_seq_rm_ndn(ndn_context* ctx, int seq_id, int p0, int p1) {
    uint64_t seq = ctx->sequence_number.fetch_add(1);
    std::string name = build_kv_cache_name(ctx, "seq-rm", seq);
    
    ndn::Interest interest(name);
    interest.setMustBeFresh(true);
    interest.setInterestLifetime(ndn::time::seconds(5));
    
    // Add parameters
    ndn::encoding::BufferStream os;
    os.write(reinterpret_cast<const char*>(&seq_id), sizeof(seq_id));
    os.write(reinterpret_cast<const char*>(&p0), sizeof(p0));
    os.write(reinterpret_cast<const char*>(&p1), sizeof(p1));
    auto buffer = os.buf();
    interest.setApplicationParameters(buffer->data(), buffer->size());
    
    ctx->interests_sent++;
    
    ctx->face.expressInterest(interest,
        [ctx](const ndn::Interest&, const ndn::Data& data) {
            ctx->data_received++;
        },
        [ctx](const ndn::Interest&, const ndn::lp::Nack& nack) {
            ctx->nacks++;
        },
        [ctx](const ndn::Interest&) {
            ctx->timeouts++;
        });
}

void send_kv_cache_seq_cp_ndn(ndn_context* ctx, int seq_id_src, int seq_id_dst, int p0, int p1) {
    uint64_t seq = ctx->sequence_number.fetch_add(1);
    std::string name = build_kv_cache_name(ctx, "seq-cp", seq);
    
    ndn::Interest interest(name);
    interest.setMustBeFresh(true);
    interest.setInterestLifetime(ndn::time::seconds(5));
    
    // Add parameters
    ndn::encoding::BufferStream os;
    os.write(reinterpret_cast<const char*>(&seq_id_src), sizeof(seq_id_src));
    os.write(reinterpret_cast<const char*>(&seq_id_dst), sizeof(seq_id_dst));
    os.write(reinterpret_cast<const char*>(&p0), sizeof(p0));
    os.write(reinterpret_cast<const char*>(&p1), sizeof(p1));
    auto buffer = os.buf();
    interest.setApplicationParameters(buffer->data(), buffer->size());
    
    ctx->interests_sent++;
    
    ctx->face.expressInterest(interest,
        [ctx](const ndn::Interest&, const ndn::Data& data) {
            ctx->data_received++;
        },
        [ctx](const ndn::Interest&, const ndn::lp::Nack& nack) {
            ctx->nacks++;
        },
        [ctx](const ndn::Interest&) {
            ctx->timeouts++;
        });
}

void send_kv_cache_seq_add_ndn(ndn_context* ctx, int seq_id, int p0, int p1, int delta) {
    uint64_t seq = ctx->sequence_number.fetch_add(1);
    std::string name = build_kv_cache_name(ctx, "seq-add", seq);
    
    ndn::Interest interest(name);
    interest.setMustBeFresh(true);
    interest.setInterestLifetime(ndn::time::seconds(5));
    
    // Add parameters
    ndn::encoding::BufferStream os;
    os.write(reinterpret_cast<const char*>(&seq_id), sizeof(seq_id));
    os.write(reinterpret_cast<const char*>(&p0), sizeof(p0));
    os.write(reinterpret_cast<const char*>(&p1), sizeof(p1));
    os.write(reinterpret_cast<const char*>(&delta), sizeof(delta));
    auto buffer = os.buf();
    interest.setApplicationParameters(buffer->data(), buffer->size());
    
    ctx->interests_sent++;
    
    ctx->face.expressInterest(interest,
        [ctx](const ndn::Interest&, const ndn::Data& data) {
            ctx->data_received++;
        },
        [ctx](const ndn::Interest&, const ndn::lp::Nack& nack) {
            ctx->nacks++;
        },
        [ctx](const ndn::Interest&) {
            ctx->timeouts++;
        });
}

void send_kv_cache_seq_div_ndn(ndn_context* ctx, int seq_id, int p0, int p1, int d) {
    uint64_t seq = ctx->sequence_number.fetch_add(1);
    std::string name = build_kv_cache_name(ctx, "seq-div", seq);
    
    ndn::Interest interest(name);
    interest.setMustBeFresh(true);
    interest.setInterestLifetime(ndn::time::seconds(5));
    
    // Add parameters
    ndn::encoding::BufferStream os;
    os.write(reinterpret_cast<const char*>(&seq_id), sizeof(seq_id));
    os.write(reinterpret_cast<const char*>(&p0), sizeof(p0));
    os.write(reinterpret_cast<const char*>(&p1), sizeof(p1));
    os.write(reinterpret_cast<const char*>(&d), sizeof(d));
    auto buffer = os.buf();
    interest.setApplicationParameters(buffer->data(), buffer->size());
    
    ctx->interests_sent++;
    
    ctx->face.expressInterest(interest,
        [ctx](const ndn::Interest&, const ndn::Data& data) {
            ctx->data_received++;
        },
        [ctx](const ndn::Interest&, const ndn::lp::Nack& nack) {
            ctx->nacks++;
        },
        [ctx](const ndn::Interest&) {
            ctx->timeouts++;
        });
}

// Synchronization functions
bool wait_for_operation(ndn_context* ctx, const std::string& operation_id, int timeout_ms) {
    std::unique_lock<std::mutex> lock(ctx->sync_mutex);
    
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    
    return ctx->sync_cv.wait_until(lock, deadline, [ctx, &operation_id]() {
        auto it = ctx->pending_operations.find(operation_id);
        return it == ctx->pending_operations.end() || !it->second;
    });
}

void wait_for_all_operations(ndn_context* ctx, int timeout_ms) {
    std::unique_lock<std::mutex> lock(ctx->sync_mutex);
    
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    
    ctx->sync_cv.wait_until(lock, deadline, [ctx]() {
        for (const auto& op : ctx->pending_operations) {
            if (op.second) return false;  // Still has pending operations
        }
        return true;  // All operations completed
    });
}

// Error handling
void handle_interest_timeout(ndn_context* ctx, const std::string& name) {
    std::cerr << "Interest timeout for " << name << std::endl;
    ctx->timeouts++;
}

void handle_nack(ndn_context* ctx, const std::string& name, const ndn::lp::Nack& nack) {
    std::cerr << "Received NACK for " << name << ": " 
              << ndn::lp::getNackReasonString(nack.getReason()) << std::endl;
    ctx->nacks++;
}

// Performance optimization
void optimize_interest_pipeline(ndn_context* ctx) {
    // Implement Interest pipelining optimizations
    // This could include adjusting Interest lifetimes, implementing Interest aggregation, etc.
}

void adjust_interest_lifetime(ndn_context* ctx, const std::string& name_type, int success_rate) {
    // Dynamically adjust Interest lifetimes based on success rates
    // This is a placeholder for adaptive timeout mechanisms
}

} // namespace ndn_prima

// C interface implementation
extern "C" {

void llama_init_ndn(struct llama_context* ctx, uint32_t n_world, uint32_t my_rank) {
    ctx->ndn_ctx = new ndn_prima::ndn_context();
    ctx->ndn_ctx->rank = my_rank;
    ctx->ndn_ctx->n_world = n_world;
    
    // Start NDN Face processing
    ctx->ndn_ctx->start();
    
    // Setup Interest filters
    ndn_prima::setup_meta_interest_filter(ctx->ndn_ctx);
    ndn_prima::setup_tensor_interest_filter(ctx->ndn_ctx);
    ndn_prima::setup_device_info_interest_filter(ctx->ndn_ctx);
    ndn_prima::setup_broadcast_interest_filter(ctx->ndn_ctx);
    ndn_prima::setup_kv_cache_interest_filters(ctx->ndn_ctx);
    
    std::cout << "NDN context initialized for rank " << my_rank << " in world of " << n_world << std::endl;
}

void llama_free_ndn(struct llama_context* ctx, char** msg) {
    if (ctx->ndn_ctx) {
        ctx->ndn_ctx->stop();
        delete ctx->ndn_ctx;
        ctx->ndn_ctx = nullptr;
    }
    
    if (msg) {
        *msg = strdup("NDN context freed");
    }
}

int llama_gather_device_info_ndn(struct llama_context* ctx, struct device_info* dev_info_set) {
    // Gather device info from all nodes
    // This would involve sending Interests to all other ranks and collecting responses
    return 0; // Success
}

int llama_send_device_info_ndn(struct llama_context* ctx, struct device_info* dev_info) {
    if (!ctx->ndn_ctx) {
        return -1;
    }
    
    ndn_prima::send_device_info_ndn(ctx->ndn_ctx, dev_info);
    return 0;
}

int llama_bcast_startup_args_ndn(struct llama_context* ctx, uint32_t rank, struct startup_args* args) {
    if (!ctx->ndn_ctx) {
        return -1;
    }
    
    ndn_prima::broadcast_startup_args_ndn(ctx->ndn_ctx, args);
    return 0;
}

int llama_bcast_layer_setup_ndn(struct llama_context* ctx, uint32_t* n_layer_window, uint32_t* n_gpu_layers) {
    // Broadcast layer setup information
    return 0;
}

int llama_recv_layer_setup_ndn(struct llama_context* ctx, uint32_t* n_layer_window, uint32_t* n_gpu_layers) {
    // Receive layer setup information
    return 0;
}

void llama_send_kv_cache_clear_ndn(struct llama_context* ctx) {
    if (ctx->ndn_ctx) {
        ndn_prima::send_kv_cache_clear_ndn(ctx->ndn_ctx);
    }
}

void llama_send_kv_cache_seq_rm_ndn(struct llama_context* ctx, int seq_id, int p0, int p1) {
    if (ctx->ndn_ctx) {
        ndn_prima::send_kv_cache_seq_rm_ndn(ctx->ndn_ctx, seq_id, p0, p1);
    }
}

void llama_send_kv_cache_seq_cp_ndn(struct llama_context* ctx, int seq_id_src, int seq_id_dst, int p0, int p1) {
    if (ctx->ndn_ctx) {
        ndn_prima::send_kv_cache_seq_cp_ndn(ctx->ndn_ctx, seq_id_src, seq_id_dst, p0, p1);
    }
}

void llama_send_kv_cache_seq_add_ndn(struct llama_context* ctx, int seq_id, int p0, int p1, int delta) {
    if (ctx->ndn_ctx) {
        ndn_prima::send_kv_cache_seq_add_ndn(ctx->ndn_ctx, seq_id, p0, p1, delta);
    }
}

void llama_send_kv_cache_seq_div_ndn(struct llama_context* ctx, int seq_id, int p0, int p1, int d) {
    if (ctx->ndn_ctx) {
        ndn_prima::send_kv_cache_seq_div_ndn(ctx->ndn_ctx, seq_id, p0, p1, d);
    }
}

void llama_ndn_print_stats(struct llama_context* ctx) {
    if (ctx->ndn_ctx) {
        ctx->ndn_ctx->print_stats();
    }
}

void llama_ndn_reset_stats(struct llama_context* ctx) {
    if (ctx->ndn_ctx) {
        ctx->ndn_ctx->reset_stats();
    }
}

} // extern "C"

#endif // USE_NDN_INSTEAD_OF_ZMQ