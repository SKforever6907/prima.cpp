// NDN implementation for prima.cpp distributed inference
// Replaces ZeroMQ with Named Data Networking

#include "llama-impl.h"
#include "ndn_prima.h"
#include <iostream>
#include <sstream>
#include <cstring>
#include <chrono>
#include <random>

#ifdef USE_NDN_INSTEAD_OF_ZMQ

namespace ndn_prima {

// 序列化函数实现
void serialize_device_info_to_buffer(const device_info* dev_info, std::ostringstream& os) {
    if (!dev_info) return;
    
    // 简单的二进制序列化 - 使用实际的结构体字段
    os.write(reinterpret_cast<const char*>(&dev_info->rank), sizeof(dev_info->rank));
    // 注意：字符串指针需要特殊处理，这里简化处理
    size_t name_len = dev_info->device_name ? strlen(dev_info->device_name) : 0;
    os.write(reinterpret_cast<const char*>(&name_len), sizeof(name_len));
    if (name_len > 0) {
        os.write(dev_info->device_name, name_len);
    }
    
    size_t os_len = dev_info->device_os ? strlen(dev_info->device_os) : 0;
    os.write(reinterpret_cast<const char*>(&os_len), sizeof(os_len));
    if (os_len > 0) {
        os.write(dev_info->device_os, os_len);
    }
}

void serialize_startup_args_to_buffer(const startup_args* args, std::ostringstream& os) {
    if (!args) return;
    
    // 简单的二进制序列化 - 使用实际的结构体字段
    os.write(reinterpret_cast<const char*>(&args->should_profile), sizeof(args->should_profile));
    os.write(reinterpret_cast<const char*>(&args->n_ctx), sizeof(args->n_ctx));
}

void deserialize_device_info_from_buffer(const uint8_t* buffer, size_t size, device_info* dev_info) {
    if (!buffer || !dev_info || size < sizeof(uint32_t)) return;
    
    // 简单的二进制反序列化
    const char* data = reinterpret_cast<const char*>(buffer);
    size_t offset = 0;
    
    memcpy(&dev_info->rank, data + offset, sizeof(dev_info->rank));
    offset += sizeof(dev_info->rank);
    
    // 简化处理：设置默认值而不是反序列化字符串
    dev_info->device_name = "unknown";
    dev_info->device_os = "unknown";
}

// NDN Context Implementation
ndn_context::ndn_context(const std::string& prefix) 
    : scheduler(face.getIoContext()), prefix_base(prefix) {
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
void serialize_meta_to_buffer(const struct sync_meta* meta, std::ostringstream& os) {
    // Write sync_meta fields in order
    os.write(reinterpret_cast<const char*>(&meta->n_tokens), sizeof(meta->n_tokens));
    os.write(reinterpret_cast<const char*>(&meta->all_pos_0), sizeof(meta->all_pos_0));
    os.write(reinterpret_cast<const char*>(&meta->all_pos_1), sizeof(meta->all_pos_1));
    os.write(reinterpret_cast<const char*>(&meta->n_ctx), sizeof(meta->n_ctx));
    
    // Write boolean flags
    os.write(reinterpret_cast<const char*>(&meta->clear_kv_cache), sizeof(meta->clear_kv_cache));
    os.write(reinterpret_cast<const char*>(&meta->kv_seq_rm), sizeof(meta->kv_seq_rm));
    os.write(reinterpret_cast<const char*>(&meta->rm_seq_id), sizeof(meta->rm_seq_id));
    os.write(reinterpret_cast<const char*>(&meta->rm_p0), sizeof(meta->rm_p0));
    os.write(reinterpret_cast<const char*>(&meta->rm_p1), sizeof(meta->rm_p1));
    
    os.write(reinterpret_cast<const char*>(&meta->kv_seq_add), sizeof(meta->kv_seq_add));
    os.write(reinterpret_cast<const char*>(&meta->add_seq_id), sizeof(meta->add_seq_id));
    os.write(reinterpret_cast<const char*>(&meta->add_p0), sizeof(meta->add_p0));
    os.write(reinterpret_cast<const char*>(&meta->add_p1), sizeof(meta->add_p1));
    os.write(reinterpret_cast<const char*>(&meta->add_delta), sizeof(meta->add_delta));
    
    os.write(reinterpret_cast<const char*>(&meta->kv_seq_cp), sizeof(meta->kv_seq_cp));
    os.write(reinterpret_cast<const char*>(&meta->cp_src_seq_id), sizeof(meta->cp_src_seq_id));
    os.write(reinterpret_cast<const char*>(&meta->cp_dst_seq_id), sizeof(meta->cp_dst_seq_id));
    os.write(reinterpret_cast<const char*>(&meta->cp_p0), sizeof(meta->cp_p0));
    os.write(reinterpret_cast<const char*>(&meta->cp_p1), sizeof(meta->cp_p1));
    
    os.write(reinterpret_cast<const char*>(&meta->kv_seq_div), sizeof(meta->kv_seq_div));
    os.write(reinterpret_cast<const char*>(&meta->div_seq_id), sizeof(meta->div_seq_id));
    os.write(reinterpret_cast<const char*>(&meta->div_p0), sizeof(meta->div_p0));
    os.write(reinterpret_cast<const char*>(&meta->div_p1), sizeof(meta->div_p1));
    os.write(reinterpret_cast<const char*>(&meta->div_factor), sizeof(meta->div_factor));
}

void deserialize_meta_from_buffer(const uint8_t* data, size_t size, struct sync_meta* meta) {
    size_t offset = 0;
    
    // Read basic fields
    if (offset + sizeof(meta->n_tokens) > size) return;
    std::memcpy(&meta->n_tokens, data + offset, sizeof(meta->n_tokens));
    offset += sizeof(meta->n_tokens);
    
    if (offset + sizeof(meta->all_pos_0) > size) return;
    std::memcpy(&meta->all_pos_0, data + offset, sizeof(meta->all_pos_0));
    offset += sizeof(meta->all_pos_0);
    
    if (offset + sizeof(meta->all_pos_1) > size) return;
    std::memcpy(&meta->all_pos_1, data + offset, sizeof(meta->all_pos_1));
    offset += sizeof(meta->all_pos_1);
    
    if (offset + sizeof(meta->n_ctx) > size) return;
    std::memcpy(&meta->n_ctx, data + offset, sizeof(meta->n_ctx));
    offset += sizeof(meta->n_ctx);
    
    // Read boolean flags and related data
    if (offset + sizeof(meta->clear_kv_cache) > size) return;
    std::memcpy(&meta->clear_kv_cache, data + offset, sizeof(meta->clear_kv_cache));
    offset += sizeof(meta->clear_kv_cache);
    
    if (offset + sizeof(meta->kv_seq_rm) > size) return;
    std::memcpy(&meta->kv_seq_rm, data + offset, sizeof(meta->kv_seq_rm));
    offset += sizeof(meta->kv_seq_rm);
    
    if (offset + sizeof(meta->rm_seq_id) > size) return;
    std::memcpy(&meta->rm_seq_id, data + offset, sizeof(meta->rm_seq_id));
    offset += sizeof(meta->rm_seq_id);
    
    if (offset + sizeof(meta->rm_p0) > size) return;
    std::memcpy(&meta->rm_p0, data + offset, sizeof(meta->rm_p0));
    offset += sizeof(meta->rm_p0);
    
    if (offset + sizeof(meta->rm_p1) > size) return;
    std::memcpy(&meta->rm_p1, data + offset, sizeof(meta->rm_p1));
    offset += sizeof(meta->rm_p1);
    
    if (offset + sizeof(meta->kv_seq_add) > size) return;
    std::memcpy(&meta->kv_seq_add, data + offset, sizeof(meta->kv_seq_add));
    offset += sizeof(meta->kv_seq_add);
    
    if (offset + sizeof(meta->add_seq_id) > size) return;
    std::memcpy(&meta->add_seq_id, data + offset, sizeof(meta->add_seq_id));
    offset += sizeof(meta->add_seq_id);
    
    if (offset + sizeof(meta->add_p0) > size) return;
    std::memcpy(&meta->add_p0, data + offset, sizeof(meta->add_p0));
    offset += sizeof(meta->add_p0);
    
    if (offset + sizeof(meta->add_p1) > size) return;
    std::memcpy(&meta->add_p1, data + offset, sizeof(meta->add_p1));
    offset += sizeof(meta->add_p1);
    
    if (offset + sizeof(meta->add_delta) > size) return;
    std::memcpy(&meta->add_delta, data + offset, sizeof(meta->add_delta));
    offset += sizeof(meta->add_delta);
    
    if (offset + sizeof(meta->kv_seq_cp) > size) return;
    std::memcpy(&meta->kv_seq_cp, data + offset, sizeof(meta->kv_seq_cp));
    offset += sizeof(meta->kv_seq_cp);
    
    if (offset + sizeof(meta->cp_src_seq_id) > size) return;
    std::memcpy(&meta->cp_src_seq_id, data + offset, sizeof(meta->cp_src_seq_id));
    offset += sizeof(meta->cp_src_seq_id);
    
    if (offset + sizeof(meta->cp_dst_seq_id) > size) return;
    std::memcpy(&meta->cp_dst_seq_id, data + offset, sizeof(meta->cp_dst_seq_id));
    offset += sizeof(meta->cp_dst_seq_id);
    
    if (offset + sizeof(meta->cp_p0) > size) return;
    std::memcpy(&meta->cp_p0, data + offset, sizeof(meta->cp_p0));
    offset += sizeof(meta->cp_p0);
    
    if (offset + sizeof(meta->cp_p1) > size) return;
    std::memcpy(&meta->cp_p1, data + offset, sizeof(meta->cp_p1));
    offset += sizeof(meta->cp_p1);
    
    if (offset + sizeof(meta->kv_seq_div) > size) return;
    std::memcpy(&meta->kv_seq_div, data + offset, sizeof(meta->kv_seq_div));
    offset += sizeof(meta->kv_seq_div);
    
    if (offset + sizeof(meta->div_seq_id) > size) return;
    std::memcpy(&meta->div_seq_id, data + offset, sizeof(meta->div_seq_id));
    offset += sizeof(meta->div_seq_id);
    
    if (offset + sizeof(meta->div_p0) > size) return;
    std::memcpy(&meta->div_p0, data + offset, sizeof(meta->div_p0));
    offset += sizeof(meta->div_p0);
    
    if (offset + sizeof(meta->div_p1) > size) return;
    std::memcpy(&meta->div_p1, data + offset, sizeof(meta->div_p1));
    offset += sizeof(meta->div_p1);
    
    if (offset + sizeof(meta->div_factor) > size) return;
    std::memcpy(&meta->div_factor, data + offset, sizeof(meta->div_factor));
    offset += sizeof(meta->div_factor);
}

// Device info serialization functions removed - using existing definitions




// Core communication functions
void send_meta_ndn(ndn_context* ctx, struct sync_meta* meta) {
    uint64_t seq = ctx->sequence_number.fetch_add(1);
    std::string name = build_meta_name(ctx, seq);
    
    ndn::Interest interest(name);
    interest.setMustBeFresh(true);
    interest.setInterestLifetime(ndn::time::seconds(5));
    interest.setCanBePrefix(false);
    
    // Serialize meta data
    std::ostringstream os;
    serialize_meta_to_buffer(meta, os);
    std::string buffer = os.str();
    interest.setApplicationParameters(std::string_view(buffer));
    
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
                     << static_cast<int>(nack.getReason()) << std::endl;
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
    std::ostringstream os;
    // This is a placeholder - actual tensor serialization would be more complex
    // For now, just serialize basic tensor info
    if (tensors->sub_gf_out) {
        size_t tensor_size = ggml_nbytes(tensors->sub_gf_out);
        os.write(reinterpret_cast<const char*>(&tensor_size), sizeof(tensor_size));
        os.write(reinterpret_cast<const char*>(tensors->sub_gf_out->data), tensor_size);
    }
    if (tensors->inp_pos) {
        size_t tensor_size = ggml_nbytes(tensors->inp_pos);
        os.write(reinterpret_cast<const char*>(&tensor_size), sizeof(tensor_size));
        os.write(reinterpret_cast<const char*>(tensors->inp_pos->data), tensor_size);
    }
    
    std::string buffer = os.str();
    interest.setApplicationParameters(std::string_view(buffer));
    
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
    
    std::ostringstream os;
    serialize_device_info_to_buffer(dev_info, os);
    std::string buffer = os.str();
    interest.setApplicationParameters(std::string_view(buffer));
    
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
    
    std::ostringstream os;
    serialize_startup_args_to_buffer(args, os);
    std::string buffer = os.str();
    interest.setApplicationParameters(std::string_view(buffer));
    
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
    std::ostringstream os;
    os.write(reinterpret_cast<const char*>(&seq_id), sizeof(seq_id));
    os.write(reinterpret_cast<const char*>(&p0), sizeof(p0));
    os.write(reinterpret_cast<const char*>(&p1), sizeof(p1));
    std::string buffer = os.str();
    interest.setApplicationParameters(std::string_view(buffer));
    
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
    std::ostringstream os;
    os.write(reinterpret_cast<const char*>(&seq_id_src), sizeof(seq_id_src));
    os.write(reinterpret_cast<const char*>(&seq_id_dst), sizeof(seq_id_dst));
    os.write(reinterpret_cast<const char*>(&p0), sizeof(p0));
    os.write(reinterpret_cast<const char*>(&p1), sizeof(p1));
    std::string buffer = os.str();
    interest.setApplicationParameters(std::string_view(buffer));
    
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
    std::ostringstream os;
    os.write(reinterpret_cast<const char*>(&seq_id), sizeof(seq_id));
    os.write(reinterpret_cast<const char*>(&p0), sizeof(p0));
    os.write(reinterpret_cast<const char*>(&p1), sizeof(p1));
    os.write(reinterpret_cast<const char*>(&delta), sizeof(delta));
    std::string buffer = os.str();
    interest.setApplicationParameters(std::string_view(buffer));
    
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
    std::ostringstream os;
    os.write(reinterpret_cast<const char*>(&seq_id), sizeof(seq_id));
    os.write(reinterpret_cast<const char*>(&p0), sizeof(p0));
    os.write(reinterpret_cast<const char*>(&p1), sizeof(p1));
    os.write(reinterpret_cast<const char*>(&d), sizeof(d));
    std::string buffer = os.str();
    interest.setApplicationParameters(std::string_view(buffer));
    
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
              << static_cast<int>(nack.getReason()) << std::endl;
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

void receive_tensor_ndn(ndn_context* ctx, struct llama_ubatch* ubatch, bool is_out_embd) {
    // This is a placeholder for receiving tensor data via NDN
    // In a real implementation, this would wait for tensor data from other nodes
    // For now, just increment the counter
    ctx->data_received++;
}

} // namespace ndn_prima

#endif // USE_NDN_INSTEAD_OF_ZMQ

// C interface implementation moved to llama.cpp to access complete llama_context definition

