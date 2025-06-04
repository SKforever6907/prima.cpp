#!/bin/bash

echo "🔧 安装NDN-CXX依赖..."

# 更新包管理器
sudo apt update

# 安装基础依赖
sudo apt install -y \
    build-essential \
    cmake \
    pkg-config \
    libboost-all-dev \
    libssl-dev \
    libsqlite3-dev \
    git

# 下载并编译NDN-CXX
echo "📦 下载NDN-CXX库..."
cd /tmp
git clone https://github.com/named-data/ndn-cxx.git
cd ndn-cxx

# 配置和编译
echo "🔨 编译NDN-CXX..."
./waf configure --with-examples
./waf
sudo ./waf install

# 更新库路径
sudo ldconfig

echo "✅ NDN-CXX安装完成！"
echo "📍 安装位置: /usr/local"
echo "📍 头文件: /usr/local/include/ndn-cxx"
echo "📍 库文件: /usr/local/lib"