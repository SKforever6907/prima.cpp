#ifndef NDN_PRIMA_H
#define NDN_PRIMA_H

// NDN替换ZeroMQ的头文件
// 用于prima.cpp的分布式推理

#ifdef USE_NDN_INSTEAD_OF_ZMQ

#include <ndn-cxx/face.hpp>
#include <ndn-cxx/security/key-chain.hpp>
#include <ndn-cxx/util/scheduler.hpp>
#include <ndn-cxx/encoding/buffer-stream.hpp>
#include <ndn-cxx/lp/nack.hpp>

#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <functional>
#include <chrono>

// 前向声明
struct llama_context;
struct sync_meta;
struct device_info;
struct startup_args;
struct input_tensors;
struct llama_ubatch;

namespace ndn_prima {

// NDN上下文结构
struct ndn_context {
    // NDN核心组件
    ndn::Face face;
    ndn::KeyChain keyChain;
    ndn::Scheduler scheduler;
    
    // 配置参数
    std::string prefix_base;      // 基础前缀，如 /prima-inference
    uint32_t session_id;          // 会话ID
    uint32_t rank;                // 当前节点rank
    uint32_t n_world;             // 总节点数
    
    // 网络配置
    std::string master_ip;        // master节点IP（用于路由提示）
    std::string next_node_ip;     // 下一个节点IP
    uint32_t data_port;           // 数据端口（用于路由提示）
    uint32_t signal_port;         // 信号端口
    
    // Interest过滤器管理
    std::vector<ndn::RegisteredPrefixHandle> registered_prefixes;
    
    // 数据缓存
    std::unordered_map<std::string, ndn::Data> data_cache;
    std::mutex cache_mutex;
    size_t max_cache_size = 1000;
    
    // 同步机制
    std::atomic<uint64_t> sequence_number{0};
    std::mutex sync_mutex;
    std::condition_variable sync_cv;
    std::unordered_map<std::string, bool> pending_operations;
    
    // 回调函数
    std::function<void(struct sync_meta*)> meta_callback;
    std::function<void(struct input_tensors*)> tensor_callback;
    std::function<void(struct device_info*)> device_info_callback;
    
    // Face处理线程
    std::thread face_thread;
    std::atomic<bool> running{false};
    
    // 性能统计
    std::atomic<uint64_t> interests_sent{0};
    std::atomic<uint64_t> interests_received{0};
    std::atomic<uint64_t> data_sent{0};
    std::atomic<uint64_t> data_received{0};
    std::atomic<uint64_t> timeouts{0};
    std::atomic<uint64_t> nacks{0};
    
    // 构造函数
    ndn_context(const std::string& prefix = "/prima-inference");
    
    // 析构函数
    ~ndn_context();
    
    // 启动和停止
    void start();
    void stop();
    
    // 缓存管理
    void cache_data(const std::string& name, const ndn::Data& data);
    bool get_cached_data(const std::string& name, ndn::Data& data);
    void cleanup_cache();
    
    // 统计信息
    void print_stats() const;
    void reset_stats();
};

// 名称构建函数
std::string build_meta_name(const ndn_context* ctx, uint64_t sequence);
std::string build_tensor_name(const ndn_context* ctx, uint32_t layer, uint64_t sequence);
std::string build_device_info_name(const ndn_context* ctx);
std::string build_broadcast_name(const ndn_context* ctx, const std::string& type);
std::string build_kv_cache_name(const ndn_context* ctx, const std::string& operation, uint64_t sequence);

// 序列化/反序列化函数
void serialize_meta_to_buffer(const struct sync_meta* meta, ndn::encoding::BufferStream& os);
void deserialize_meta_from_buffer(const uint8_t* data, size_t size, struct sync_meta* meta);

void serialize_device_info_to_buffer(const struct device_info* info, ndn::encoding::BufferStream& os);
void deserialize_device_info_from_buffer(const uint8_t* data, size_t size, struct device_info* info);

void serialize_startup_args_to_buffer(const struct startup_args* args, ndn::encoding::BufferStream& os);
void deserialize_startup_args_from_buffer(const uint8_t* data, size_t size, struct startup_args* args);

// 核心通信函数
void send_meta_ndn(ndn_context* ctx, struct sync_meta* meta);
void send_tensor_ndn(ndn_context* ctx, struct llama_ubatch* ubatch, struct input_tensors* tensors);
void send_device_info_ndn(ndn_context* ctx, struct device_info* dev_info);
void broadcast_startup_args_ndn(ndn_context* ctx, struct startup_args* args);

// Interest过滤器设置
void setup_meta_interest_filter(ndn_context* ctx);
void setup_tensor_interest_filter(ndn_context* ctx);
void setup_device_info_interest_filter(ndn_context* ctx);
void setup_broadcast_interest_filter(ndn_context* ctx);
void setup_kv_cache_interest_filters(ndn_context* ctx);

// KV缓存操作
void send_kv_cache_clear_ndn(ndn_context* ctx);
void send_kv_cache_seq_rm_ndn(ndn_context* ctx, int seq_id, int p0, int p1);
void send_kv_cache_seq_cp_ndn(ndn_context* ctx, int seq_id_src, int seq_id_dst, int p0, int p1);
void send_kv_cache_seq_add_ndn(ndn_context* ctx, int seq_id, int p0, int p1, int delta);
void send_kv_cache_seq_div_ndn(ndn_context* ctx, int seq_id, int p0, int p1, int d);

// 同步和等待函数
bool wait_for_operation(ndn_context* ctx, const std::string& operation_id, int timeout_ms = 5000);
void wait_for_all_operations(ndn_context* ctx, int timeout_ms = 10000);

// 错误处理
void handle_interest_timeout(ndn_context* ctx, const std::string& name);
void handle_nack(ndn_context* ctx, const std::string& name, const ndn::lp::Nack& nack);

// 性能优化
void optimize_interest_pipeline(ndn_context* ctx);
void adjust_interest_lifetime(ndn_context* ctx, const std::string& name_type, int success_rate);

} // namespace ndn_prima

// C接口函数（替换原有的ZMQ函数）
extern "C" {

// 初始化和清理
void llama_init_ndn(struct llama_context* ctx, uint32_t n_world, uint32_t my_rank);
void llama_free_ndn(struct llama_context* ctx, char** msg);

// 设备信息交换
int llama_gather_device_info_ndn(struct llama_context* ctx, struct device_info* dev_info_set);
int llama_send_device_info_ndn(struct llama_context* ctx, struct device_info* dev_info);

// 启动参数广播
int llama_bcast_startup_args_ndn(struct llama_context* ctx, uint32_t rank, struct startup_args* args);

// 层设置
int llama_bcast_layer_setup_ndn(struct llama_context* ctx, uint32_t* n_layer_window, uint32_t* n_gpu_layers);
int llama_recv_layer_setup_ndn(struct llama_context* ctx, uint32_t* n_layer_window, uint32_t* n_gpu_layers);

// KV缓存同步
void llama_send_kv_cache_clear_ndn(struct llama_context* ctx);
void llama_send_kv_cache_seq_rm_ndn(struct llama_context* ctx, int seq_id, int p0, int p1);
void llama_send_kv_cache_seq_cp_ndn(struct llama_context* ctx, int seq_id_src, int seq_id_dst, int p0, int p1);
void llama_send_kv_cache_seq_add_ndn(struct llama_context* ctx, int seq_id, int p0, int p1, int delta);
void llama_send_kv_cache_seq_div_ndn(struct llama_context* ctx, int seq_id, int p0, int p1, int d);

// 性能和调试
void llama_ndn_print_stats(struct llama_context* ctx);
void llama_ndn_reset_stats(struct llama_context* ctx);

}

// 宏定义，用于条件编译
#ifdef USE_NDN_INSTEAD_OF_ZMQ
#define llama_init_sockets llama_init_ndn
#define llama_free_sockets llama_free_ndn
#define llama_gather_device_info llama_gather_device_info_ndn
#define llama_send_device_info llama_send_device_info_ndn
#define llama_bcast_startup_args llama_bcast_startup_args_ndn
#define llama_bcast_layer_setup llama_bcast_layer_setup_ndn
#define llama_recv_layer_setup llama_recv_layer_setup_ndn
#define llama_send_kv_cache_clear llama_send_kv_cache_clear_ndn
#define llama_send_kv_cache_seq_rm llama_send_kv_cache_seq_rm_ndn
#define llama_send_kv_cache_seq_cp llama_send_kv_cache_seq_cp_ndn
#define llama_send_kv_cache_seq_add llama_send_kv_cache_seq_add_ndn
#define llama_send_kv_cache_seq_div llama_send_kv_cache_seq_div_ndn
#endif

#else
// 如果不使用NDN，包含原有的ZMQ头文件
#include "zmq_addon.hpp"
#endif // USE_NDN_INSTEAD_OF_ZMQ

#endif // NDN_PRIMA_H