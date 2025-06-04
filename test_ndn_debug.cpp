#include <iostream>
#include <cstdlib>

// 简单测试程序来验证NDN宏定义
int main() {
    std::cout << "Testing Prima.cpp NDN compilation..." << std::endl;
    
#ifdef USE_NDN_INSTEAD_OF_ZMQ
    std::cout << "✅ USE_NDN_INSTEAD_OF_ZMQ is defined - NDN code will be used" << std::endl;
#else
    std::cout << "❌ USE_NDN_INSTEAD_OF_ZMQ is NOT defined - ZMQ code will be used" << std::endl;
#endif

    return 0;
}