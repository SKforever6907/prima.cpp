#!/bin/bash

echo "🧪 Testing Prima.cpp NDN Implementation"
echo "======================================="

# 检查NFD是否需要安装
if ! command -v nfd &> /dev/null; then
    echo "⚠️  NFD (NDN Forwarding Daemon) not found. Installing..."
    
    # 安装NFD依赖
    sudo apt update
    sudo apt install -y build-essential pkg-config libboost-all-dev libssl-dev libsqlite3-dev
    
    # 下载并编译NFD
    if [ ! -d "NFD" ]; then
        echo "📥 Downloading NFD..."
        git clone --depth 1 https://github.com/named-data/NFD.git
        cd NFD
        ./waf configure
        ./waf
        sudo ./waf install
        cd ..
    fi
fi

# 启动NFD
echo "🚀 Starting NFD..."
sudo pkill nfd 2>/dev/null || true
sleep 1
sudo nfd-start &
sleep 3

# 检查NFD状态
if pgrep nfd > /dev/null; then
    echo "✅ NFD is running"
else
    echo "❌ Failed to start NFD"
    exit 1
fi

# 创建测试目录
mkdir -p test_ndn_logs

echo ""
echo "🔧 Testing Prima NDN with debug output..."

# 测试1: 检查NDN初始化
echo "Test 1: NDN Initialization Test"
echo "--------------------------------"

cd build-ndn

# 运行prima-main-ndn并检查NDN相关输出
timeout 10s ./prima-main-ndn --help 2>&1 | tee ../test_ndn_logs/help_output.log

echo ""
echo "Test 2: NDN Socket Initialization Test"
echo "--------------------------------------"

# 创建一个简单的测试配置
cat > ../test_config.txt << EOF
-m models/test.gguf
-n 10
-p "Hello"
--rank 0
--world-size 2
EOF

# 测试NDN socket初始化（会失败但能看到NDN代码路径）
echo "Running with rank 0 (master node)..."
timeout 5s ./prima-main-ndn -m /nonexistent/model.gguf -n 1 -p "test" --rank 0 --world-size 2 2>&1 | tee ../test_ndn_logs/rank0_output.log &

sleep 2

echo "Running with rank 1 (worker node)..."
timeout 5s ./prima-main-ndn -m /nonexistent/model.gguf -n 1 -p "test" --rank 1 --world-size 2 2>&1 | tee ../test_ndn_logs/rank1_output.log &

wait

echo ""
echo "📊 Test Results Analysis"
echo "========================"

echo "Checking for NDN-specific code execution..."

# 检查是否执行了NDN代码路径
if grep -q "ndn" ../test_ndn_logs/*.log 2>/dev/null; then
    echo "✅ NDN code paths detected in logs"
else
    echo "⚠️  No explicit NDN mentions found in logs"
fi

# 检查是否有ZMQ相关错误（应该没有，因为用的是NDN）
if grep -q "zmq\|ZMQ" ../test_ndn_logs/*.log 2>/dev/null; then
    echo "❌ ZMQ code still being executed (NDN replacement incomplete)"
else
    echo "✅ No ZMQ code detected (good - using NDN instead)"
fi

# 检查socket初始化
if grep -q -i "socket\|connect\|bind" ../test_ndn_logs/*.log 2>/dev/null; then
    echo "✅ Socket/connection initialization detected"
else
    echo "⚠️  No socket initialization detected"
fi

echo ""
echo "📋 Log Files Created:"
ls -la ../test_ndn_logs/

echo ""
echo "🔍 To examine detailed logs:"
echo "  cat test_ndn_logs/rank0_output.log"
echo "  cat test_ndn_logs/rank1_output.log"

echo ""
echo "🧪 NDN Test Complete!"