// NDN替换ZeroMQ的实现示例
// 需要安装 ndn-cxx 库

#include <ndn-cxx/face.hpp>
#include <ndn-cxx/security/key-chain.hpp>
#include <ndn-cxx/util/scheduler.hpp>
#include <ndn-cxx/encoding/buffer-stream.hpp>

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <unordered_map>

// 替换原有的zmq_addon.hpp
namespace ndn_prima {

struct ndn_context {
    ndn::Face face;
    ndn::KeyChain keyChain;
    ndn::Scheduler scheduler;
    
    std::string prefix_base;  // /prima-inference
    uint32_t session_id;
    uint32_t rank;
    uint32_t n_world;
    
    // Interest过滤器句柄
    std::vector<ndn::RegisteredPrefixHandle> registered_prefixes;
    
    // 数据缓存
    std::unordered_map<std::string, ndn::Data> data_cache;
    std::mutex cache_mutex;
    
    // 同步机制
    std::atomic<uint64_t> sequence_number{0};
    std::mutex sync_mutex;
    std::condition_variable sync_cv;
    std::unordered_map<std::string, bool> pending_operations;
    
    // Face处理线程
    std::thread face_thread;
    std::atomic<bool> running{false};
    
    ndn_context(const std::string& prefix = "/prima-inference") 
        : scheduler(face.getIoService()), prefix_base(prefix) {}
    
    ~ndn_context() {
        stop();
    }
    
    void start() {
        running = true;
        face_thread = std::thread([this]() {
            while (running) {
                try {
                    face.processEvents(ndn::time::milliseconds(10));
                } catch (const std::exception& e) {
                    // 处理异常
                }
            }
        });
    }
    
    void stop() {
        running = false;
        if (face_thread.joinable()) {
            face_thread.join();
        }
        
        // 清理注册的前缀
        for (auto& handle : registered_prefixes) {
            handle.unregister();
        }
        registered_prefixes.clear();
    }
};

// 构建NDN名称
std::string build_name(const ndn_context* ctx, const std::string& type, 
                      const std::string& suffix = "") {
    std::ostringstream oss;
    oss << ctx->prefix_base << "/session-" << ctx->session_id 
        << "/rank-" << ctx->rank << "/" << type;
    if (!suffix.empty()) {
        oss << "/" << suffix;
    }
    return oss.str();
}

std::string build_broadcast_name(const ndn_context* ctx, const std::string& type) {
    std::ostringstream oss;
    oss << ctx->prefix_base << "/session-" << ctx->session_id 
        << "/broadcast/" << type;
    return oss.str();
}

// 序列化sync_meta到Interest
void serialize_meta_to_interest(ndn::Interest& interest, const struct sync_meta* meta) {
    ndn::encoding::BufferStream os;
    
    // 写入meta数据
    os.write(reinterpret_cast<const char*>(&meta->n_tokens), sizeof(meta->n_tokens));
    os.write(reinterpret_cast<const char*>(&meta->all_pos_0), sizeof(meta->all_pos_0));
    os.write(reinterpret_cast<const char*>(&meta->all_pos_1), sizeof(meta->all_pos_1));
    
    // 添加其他字段...
    
    auto buffer = os.buf();
    interest.setApplicationParameters(buffer->data(), buffer->size());
}

// 从Interest反序列化sync_meta
void deserialize_meta_from_interest(const ndn::Interest& interest, struct sync_meta* meta) {
    if (!interest.hasApplicationParameters()) {
        return;
    }
    
    auto params = interest.getApplicationParameters();
    const char* data = reinterpret_cast<const char*>(params.data());
    size_t offset = 0;
    
    std::memcpy(&meta->n_tokens, data + offset, sizeof(meta->n_tokens));
    offset += sizeof(meta->n_tokens);
    
    std::memcpy(&meta->all_pos_0, data + offset, sizeof(meta->all_pos_0));
    offset += sizeof(meta->all_pos_0);
    
    std::memcpy(&meta->all_pos_1, data + offset, sizeof(meta->all_pos_1));
    offset += sizeof(meta->all_pos_1);
    
    // 反序列化其他字段...
}

// 替换llama_send_meta
void llama_send_meta_ndn(ndn_context* ndn_ctx, struct sync_meta* meta) {
    uint64_t seq = ndn_ctx->sequence_number.fetch_add(1);
    std::string name = build_name(ndn_ctx, "meta", "seq-" + std::to_string(seq));
    
    ndn::Interest interest(name);
    interest.setMustBeFresh(true);
    interest.setInterestLifetime(ndn::time::seconds(5));
    interest.setCanBePrefix(false);
    
    // 序列化meta数据
    serialize_meta_to_interest(interest, meta);
    
    // 设置操作为pending
    {
        std::lock_guard<std::mutex> lock(ndn_ctx->sync_mutex);
        ndn_ctx->pending_operations[name] = true;
    }
    
    ndn_ctx->face.expressInterest(interest,
        [ndn_ctx, name](const ndn::Interest&, const ndn::Data& data) {
            // 成功接收到响应
            {
                std::lock_guard<std::mutex> lock(ndn_ctx->sync_mutex);
                ndn_ctx->pending_operations[name] = false;
            }
            ndn_ctx->sync_cv.notify_all();
        },
        [ndn_ctx, name](const ndn::Interest&, const ndn::lp::Nack& nack) {
            // 处理NACK
            printf("Received NACK for %s: %s\n", name.c_str(), 
                   ndn::lp::getNackReasonString(nack.getReason()).c_str());
            {
                std::lock_guard<std::mutex> lock(ndn_ctx->sync_mutex);
                ndn_ctx->pending_operations[name] = false;
            }
            ndn_ctx->sync_cv.notify_all();
        },
        [ndn_ctx, name](const ndn::Interest&) {
            // 处理超时
            printf("Interest timeout for %s\n", name.c_str());
            {
                std::lock_guard<std::mutex> lock(ndn_ctx->sync_mutex);
                ndn_ctx->pending_operations[name] = false;
            }
            ndn_ctx->sync_cv.notify_all();
        });
}

// 设置meta数据的Interest过滤器
void setup_meta_interest_filter(ndn_context* ndn_ctx, 
                                std::function<void(struct sync_meta*)> callback) {
    std::string prefix = build_name(ndn_ctx, "meta");
    
    auto handle = ndn_ctx->face.setInterestFilter(prefix,
        [ndn_ctx, callback](const ndn::InterestFilter&, const ndn::Interest& interest) {
            // 解析Interest中的meta数据
            struct sync_meta meta = {};
            deserialize_meta_from_interest(interest, &meta);
            
            // 调用回调处理meta数据
            callback(&meta);
            
            // 发送ACK响应
            ndn::Data data(interest.getName());
            std::string content = "ACK";
            data.setContent(ndn::encoding::makeStringBlock(ndn::tlv::Content, content));
            data.setFreshnessPeriod(ndn::time::seconds(1));
            ndn_ctx->keyChain.sign(data);
            ndn_ctx->face.put(data);
        },
        [](const ndn::Name& prefix, const std::string& reason) {
            printf("Failed to register prefix %s: %s\n", prefix.toUri().c_str(), reason.c_str());
        });
        
    ndn_ctx->registered_prefixes.push_back(handle);
}

// 广播设备信息
void llama_bcast_device_info_ndn(ndn_context* ndn_ctx, struct device_info* dev_info) {
    std::string name = build_broadcast_name(ndn_ctx, "device-info");
    
    ndn::Interest interest(name);
    interest.setMustBeFresh(true);
    interest.setInterestLifetime(ndn::time::seconds(10));
    
    // 序列化设备信息
    ndn::encoding::BufferStream os;
    // 这里需要实现device_info的序列化
    // serialize_device_info(os, dev_info);
    
    auto buffer = os.buf();
    interest.setApplicationParameters(buffer->data(), buffer->size());
    
    ndn_ctx->face.expressInterest(interest,
        [](const ndn::Interest&, const ndn::Data& data) {
            printf("Device info broadcast successful\n");
        },
        [](const ndn::Interest&, const ndn::lp::Nack& nack) {
            printf("Device info broadcast failed: NACK\n");
        },
        [](const ndn::Interest&) {
            printf("Device info broadcast timeout\n");
        });
}

// 等待所有pending操作完成
void wait_for_sync(ndn_context* ndn_ctx, int timeout_ms = 5000) {
    std::unique_lock<std::mutex> lock(ndn_ctx->sync_mutex);
    
    auto deadline = std::chrono::steady_clock::now() + 
                   std::chrono::milliseconds(timeout_ms);
    
    ndn_ctx->sync_cv.wait_until(lock, deadline, [ndn_ctx]() {
        for (const auto& op : ndn_ctx->pending_operations) {
            if (op.second) return false;  // 还有pending操作
        }
        return true;  // 所有操作都完成了
    });
}

// 替换llama_init_sockets
void llama_init_ndn(struct llama_context* ctx, uint32_t n_world, uint32_t my_rank) {
    ctx->ndn_ctx = new ndn_context();
    ctx->ndn_ctx->session_id = std::time(nullptr);  // 简单的session ID
    ctx->ndn_ctx->rank = my_rank;
    ctx->ndn_ctx->n_world = n_world;
    
    // 启动Face处理
    ctx->ndn_ctx->start();
    
    // 设置Interest过滤器
    setup_meta_interest_filter(ctx->ndn_ctx, [ctx](struct sync_meta* meta) {
        // 处理接收到的meta数据
        // 这里需要根据原有逻辑处理meta
    });
    
    printf("NDN context initialized for rank %d\n", my_rank);
}

// 替换llama_free_sockets
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

} // namespace ndn_prima

// 使用示例
/*
int main() {
    // 初始化NDN上下文
    struct llama_context ctx = {};
    llama_init_ndn(&ctx, 4, 0);  // 4个节点，当前是rank 0
    
    // 发送meta数据
    struct sync_meta meta = {};
    meta.n_tokens = 10;
    meta.all_pos_0 = 0;
    meta.all_pos_1 = 9;
    
    llama_send_meta_ndn(ctx.ndn_ctx, &meta);
    
    // 等待同步完成
    wait_for_sync(ctx.ndn_ctx);
    
    // 清理
    llama_free_ndn(&ctx, nullptr);
    
    return 0;
}
*/