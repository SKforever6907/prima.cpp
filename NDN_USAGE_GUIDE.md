# NDN版本Prima.cpp使用指南

## 🎯 概述

这是prima.cpp的NDN（Named Data Networking）版本，将原有的ZMQ分布式通信替换为NDN网络通信。

## 📋 系统要求

- Ubuntu 20.04+ 或类似Linux发行版
- GCC 9+ 支持C++17
- CMake 3.16+
- 至少4GB内存用于编译

## 🚀 快速开始

### 1. 获取代码
```bash
git clone https://github.com/SKforever6907/prima.cpp.git
cd prima.cpp
git checkout ndn-migration
```

### 2. 安装依赖
```bash
# 安装NDN-CXX库和依赖
./install_ndn_deps.sh
```

### 3. 编译NDN版本
```bash
# 编译NDN版本
./build_ndn.sh
```

### 4. 测试功能
```bash
# 运行完整测试
./test_ndn_complete.sh
```

## 🔧 手动编译步骤

如果自动脚本失败，可以手动执行：

### 安装NDN-CXX
```bash
# 安装基础依赖
sudo apt update
sudo apt install -y build-essential cmake pkg-config \
    libboost-all-dev libssl-dev libsqlite3-dev git

# 编译NDN-CXX
cd /tmp
git clone https://github.com/named-data/ndn-cxx.git
cd ndn-cxx
./waf configure --with-examples
./waf
sudo ./waf install
sudo ldconfig
```

### 编译Prima NDN版本
```bash
cd prima.cpp
mkdir -p build-ndn
cd build-ndn
cmake -f ../CMakeLists_ndn.txt \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_NDN_INSTEAD_OF_ZMQ=ON
make -j$(nproc)
```

## 🎮 使用方法

### 单节点模式（本地运行）
```bash
cd build-ndn
./prima-main-ndn --help                    # 查看帮助
./prima-main-ndn model.gguf                # 运行模型
```

### 分布式模式（NDN网络）
```bash
# 节点0（主节点）
./prima-main-ndn --n-world 2 --rank 0 model.gguf

# 节点1（工作节点）
./prima-main-ndn --n-world 2 --rank 1 model.gguf
```

## 🧪 测试和验证

### 基础功能测试
```bash
cd build-ndn
./test_ndn_init                           # NDN初始化测试
./prima-main-ndn --help                   # 基础功能测试
```

### NDN符号验证
```bash
nm prima-main-ndn | grep -i ndn           # 检查NDN符号
ldd prima-main-ndn | grep ndn              # 检查NDN库链接
```

### 调试模式
```bash
# 启用详细调试输出
export PRIMA_DEBUG=1
./prima-main-ndn --n-world 2 --rank 0 model.gguf
```

## 🔍 故障排除

### 编译错误
1. **NDN-CXX未找到**
   ```bash
   export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
   ```

2. **缺少依赖**
   ```bash
   sudo apt install libboost-all-dev libssl-dev
   ```

3. **CMake配置失败**
   ```bash
   rm -rf build-ndn
   mkdir build-ndn
   cd build-ndn
   cmake -f ../CMakeLists_ndn.txt
   ```

### 运行时错误
1. **NDN连接失败**
   - 这是正常的，因为没有运行NFD守护进程
   - NDN代码路径仍然会被正确执行

2. **模型加载失败**
   - 确保模型文件存在且格式正确
   - 使用较小的测试模型进行验证

## 📊 性能对比

| 特性 | ZMQ版本 | NDN版本 |
|------|---------|---------|
| 编译大小 | ~3.2MB | ~3.5MB |
| 启动时间 | 快 | 稍慢（NDN初始化） |
| 网络协议 | TCP/IPC | NDN |
| 依赖库 | ZMQ | NDN-CXX |

## 🔗 相关链接

- [NDN-CXX文档](https://named-data.net/doc/ndn-cxx/)
- [Prima.cpp原项目](https://github.com/SKforever6907/prima.cpp)
- [NDN项目官网](https://named-data.net/)

## 📝 开发说明

### 代码结构
- `src/ndn-prima.cpp` - NDN核心实现
- `ndn_prima.h` - NDN接口定义
- `src/llama.cpp` - 主要逻辑（包含NDN条件编译）
- `CMakeLists_ndn.txt` - NDN版本构建配置

### 条件编译
所有NDN代码都使用 `#ifdef USE_NDN_INSTEAD_OF_ZMQ` 包装，确保与原版本兼容。

### 调试输出
NDN版本包含详细的调试输出，使用emoji标记便于识别：
- 🚀 初始化开始
- ✅ 操作成功  
- 📡 网络通信
- 🔧 配置设置
- 🎉 完成状态