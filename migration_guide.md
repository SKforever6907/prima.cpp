# Prima.cpp ZeroMQ到NDN迁移指南

## 概述

本指南详细说明如何将prima.cpp中的ZeroMQ通信替换为NDN（Named Data Networking）。

## 迁移步骤

### 1. 环境准备

#### 安装NDN-CXX库

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install libndn-cxx-dev
```

**从源码编译:**
```bash
git clone https://github.com/named-data/ndn-cxx.git
cd ndn-cxx
./waf configure
./waf
sudo ./waf install
sudo ldconfig
```

**macOS (Homebrew):**
```bash
brew install ndn-cxx
```

#### 启动NFD (NDN Forwarding Daemon)
```bash
# 安装NFD
sudo apt-get install nfd  # Ubuntu/Debian
# 或
brew install nfd          # macOS

# 启动NFD
nfd-start
```

### 2. 代码修改

#### 2.1 修改CMakeLists.txt

将原有的ZeroMQ依赖替换为NDN-CXX：

```cmake
# 移除ZeroMQ
# find_package(PkgConfig REQUIRED)
# pkg_check_modules(ZMQ REQUIRED libzmq)

# 添加NDN-CXX
find_package(PkgConfig REQUIRED)
pkg_check_modules(NDN_CXX REQUIRED libndn-cxx)

# 更新编译选项
add_definitions(-DUSE_NDN_INSTEAD_OF_ZMQ)
target_link_libraries(prima ${NDN_CXX_LIBRARIES})
```

#### 2.2 修改头文件包含

在`src/llama.cpp`中：

```cpp
// 替换
// #include "zmq_addon.hpp"

// 为
#include "ndn_prima.h"
```

#### 2.3 修改llama_context结构

在`llama.cpp`中的`llama_context`结构中：

```cpp
struct llama_context {
    // 其他成员...
    
#ifdef USE_NDN_INSTEAD_OF_ZMQ
    ndn_prima::ndn_context * ndn_ctx = nullptr;
#else
    // 原有的ZMQ成员
    zmq::context_t * sock_context  = nullptr;
    zmq::socket_t  * send_socket   = nullptr;
    zmq::socket_t  * recv_socket   = nullptr;
    zmq::socket_t  * master_socket = nullptr;
    zmq::socket_t  * signal_socket = nullptr;
#endif
};
```

### 3. 函数替换映射

| ZeroMQ函数 | NDN函数 | 说明 |
|------------|---------|------|
| `llama_init_sockets` | `llama_init_ndn` | 初始化通信 |
| `llama_free_sockets` | `llama_free_ndn` | 清理资源 |
| `llama_send_meta` | `send_meta_ndn` | 发送元数据 |
| `llama_recv_meta` | Interest过滤器 | 接收元数据 |
| `llama_send_tensors` | `send_tensor_ndn` | 发送张量 |
| `llama_recv_tensors` | Interest过滤器 | 接收张量 |

### 4. 配置参数调整

#### 4.1 命令行参数

原有参数保持不变，但含义略有调整：

```bash
# 原有ZMQ版本
./prima-main --world 4 --rank 0 --master-ip 192.168.1.100 --next-ip 192.168.1.101

# NDN版本（IP用作路由提示）
./prima-main-ndn --world 4 --rank 0 --master-ip 192.168.1.100 --next-ip 192.168.1.101
```

#### 4.2 NDN特定配置

可以通过环境变量配置NDN参数：

```bash
export NDN_PRIMA_PREFIX="/prima-inference"
export NDN_PRIMA_CACHE_SIZE="1000"
export NDN_PRIMA_INTEREST_LIFETIME="5000"  # 毫秒
```

### 5. 网络拓扑配置

#### 5.1 NFD路由配置

为了支持分布式推理，需要配置NFD路由：

```bash
# 在master节点上
nfdc route add /prima-inference tcp4://192.168.1.101:6363
nfdc route add /prima-inference tcp4://192.168.1.102:6363
nfdc route add /prima-inference tcp4://192.168.1.103:6363

# 在worker节点上
nfdc route add /prima-inference tcp4://192.168.1.100:6363  # master
```

#### 5.2 自动路由发现

可以使用NLSR (Named-data Link State Routing)进行自动路由：

```bash
# 安装NLSR
sudo apt-get install nlsr

# 配置NLSR
sudo nano /etc/ndn/nlsr.conf
```

### 6. 性能优化

#### 6.1 Interest聚合

NDN会自动聚合相同的Interest，减少网络流量：

```cpp
// 在ndn_context中配置
ctx->face.getIoService().post([ctx]() {
    // 批量处理Interest
});
```

#### 6.2 缓存策略

配置合适的缓存策略：

```cpp
// 设置缓存大小
ctx->max_cache_size = 2000;

// 设置数据新鲜度
data.setFreshnessPeriod(ndn::time::seconds(10));
```

#### 6.3 Interest生命周期

根据网络延迟调整Interest生命周期：

```cpp
interest.setInterestLifetime(ndn::time::milliseconds(5000));
```

### 7. 调试和监控

#### 7.1 启用调试日志

```cpp
// 在初始化时启用
#ifdef DEBUG_NDN
ndn::util::Logging::setLevel("ndn.Face", ndn::util::LogLevel::DEBUG);
#endif
```

#### 7.2 性能监控

```cpp
// 打印统计信息
llama_ndn_print_stats(ctx);
```

#### 7.3 网络监控

使用NDN工具监控网络状态：

```bash
# 查看NFD状态
nfd-status

# 查看路由表
nfd-status -r

# 查看面信息
nfd-status -f
```

### 8. 测试验证

#### 8.1 单元测试

```cpp
// 测试NDN通信
void test_ndn_communication() {
    struct llama_context ctx = {};
    llama_init_ndn(&ctx, 2, 0);
    
    // 测试meta发送
    struct sync_meta meta = {};
    meta.n_tokens = 10;
    send_meta_ndn(ctx.ndn_ctx, &meta);
    
    // 等待完成
    wait_for_all_operations(ctx.ndn_ctx);
    
    llama_free_ndn(&ctx, nullptr);
}
```

#### 8.2 集成测试

```bash
# 启动多个节点测试
./prima-main-ndn --world 4 --rank 0 &
./prima-main-ndn --world 4 --rank 1 &
./prima-main-ndn --world 4 --rank 2 &
./prima-main-ndn --world 4 --rank 3 &
```

### 9. 故障排除

#### 9.1 常见问题

**问题1: Interest超时**
```
解决方案: 检查NFD路由配置，增加Interest生命周期
```

**问题2: 数据包过大**
```
解决方案: 实现数据分片机制
```

**问题3: 缓存命中率低**
```
解决方案: 优化命名策略，增加缓存大小
```

#### 9.2 性能对比

| 指标 | ZeroMQ | NDN | 说明 |
|------|--------|-----|------|
| 延迟 | 低 | 中等 | NDN有额外的路由开销 |
| 吞吐量 | 高 | 中等 | 取决于缓存命中率 |
| 容错性 | 低 | 高 | NDN支持多路径 |
| 扩展性 | 中等 | 高 | NDN支持灵活拓扑 |

### 10. 迁移检查清单

- [ ] 安装NDN-CXX库
- [ ] 启动NFD守护进程
- [ ] 修改CMakeLists.txt
- [ ] 更新头文件包含
- [ ] 替换通信函数
- [ ] 配置NFD路由
- [ ] 测试基本通信
- [ ] 性能调优
- [ ] 集成测试
- [ ] 部署验证

### 11. 回滚计划

如果NDN迁移遇到问题，可以通过以下方式回滚：

```bash
# 重新编译ZMQ版本
cmake -DUSE_NDN_INSTEAD_OF_ZMQ=OFF ..
make

# 或者使用git分支
git checkout zmq-version
```

这个迁移过程需要仔细测试，特别是在生产环境中。建议先在测试环境中完成迁移和验证。