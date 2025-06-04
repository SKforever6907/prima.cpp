#!/bin/bash

echo "Compiling NDN test program..."

# 编译NDN测试程序
g++ -std=c++17 -o test_ndn test_ndn.cpp \
    -I/usr/local/include \
    -L/usr/local/lib \
    -lndn-cxx \
    -lboost_system \
    -lboost_filesystem \
    -lboost_thread \
    -lssl -lcrypto \
    -pthread

if [ $? -eq 0 ]; then
    echo "✅ NDN test program compiled successfully!"
    echo ""
    echo "Usage:"
    echo "  Terminal 1: ./test_ndn producer"
    echo "  Terminal 2: ./test_ndn consumer"
else
    echo "❌ Compilation failed!"
    exit 1
fi