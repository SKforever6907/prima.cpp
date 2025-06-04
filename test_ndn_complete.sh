#!/bin/bash

echo "🧪 NDN功能完整测试..."

# 确保在构建目录
if [ ! -f "build-ndn/prima-main-ndn" ]; then
    echo "❌ 错误: 请先运行 ./build_ndn.sh 编译NDN版本"
    exit 1
fi

cd build-ndn

echo ""
echo "=== 1. 基础功能测试 ==="
echo "🔍 测试可执行文件..."
if ./prima-main-ndn --help > /dev/null 2>&1; then
    echo "✅ 可执行文件运行正常"
else
    echo "❌ 可执行文件运行失败"
fi

echo ""
echo "=== 2. NDN符号验证 ==="
echo "🔍 检查NDN符号链接..."
NDN_SYMBOLS=$(nm prima-main-ndn | grep -i ndn | wc -l)
echo "📊 找到 $NDN_SYMBOLS 个NDN符号"
if [ $NDN_SYMBOLS -gt 0 ]; then
    echo "✅ NDN库已正确链接"
    echo "🔍 主要NDN符号:"
    nm prima-main-ndn | grep -i ndn | head -3
else
    echo "❌ 未找到NDN符号"
fi

echo ""
echo "=== 3. NDN初始化测试 ==="
if [ -f "test_ndn_init" ]; then
    echo "🧪 运行NDN初始化测试..."
    ./test_ndn_init
else
    echo "⚠️ NDN初始化测试程序不存在，编译中..."
    cd ..
    g++ -std=c++17 -o build-ndn/test_ndn_init test_ndn_init.cpp \
        -I/usr/local/include -L/usr/local/lib \
        -lndn-cxx -lboost_system -lssl -lcrypto -pthread
    cd build-ndn
    echo "🧪 运行NDN初始化测试..."
    ./test_ndn_init
fi

echo ""
echo "=== 4. 分布式模式测试 ==="
echo "🧪 测试分布式模式参数..."
echo "单节点模式 (应该跳过NDN):"
./prima-main-ndn --help 2>&1 | head -5

echo ""
echo "=== 5. 编译配置验证 ==="
echo "🔍 验证编译配置..."
if strings prima-main-ndn | grep -q "USE_NDN_INSTEAD_OF_ZMQ"; then
    echo "✅ NDN编译宏已启用"
else
    echo "⚠️ 未找到NDN编译宏"
fi

echo ""
echo "=== 测试总结 ==="
echo "📊 可执行文件大小: $(ls -lh prima-main-ndn | awk '{print $5}')"
echo "📊 NDN符号数量: $NDN_SYMBOLS"
echo "📍 可执行文件位置: $(pwd)/prima-main-ndn"
echo ""
echo "🎉 NDN版本测试完成！"
echo ""
echo "📝 使用说明:"
echo "   单节点运行: ./prima-main-ndn [模型文件]"
echo "   分布式运行: ./prima-main-ndn --n-world 2 --rank 0 [模型文件]"
echo "   查看帮助: ./prima-main-ndn --help"