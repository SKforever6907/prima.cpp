#include <ndn-cxx/face.hpp>
#include <ndn-cxx/interest.hpp>
#include <ndn-cxx/data.hpp>
#include <ndn-cxx/security/key-chain.hpp>
#include <iostream>
#include <thread>
#include <chrono>

using namespace ndn;

class NDNTest {
public:
    NDNTest() : m_face(m_ioContext) {}
    
    void runProducer() {
        std::cout << "Starting NDN Producer..." << std::endl;
        
        // 注册前缀
        m_face.setInterestFilter("/prima/test",
            [this](const InterestFilter&, const Interest& interest) {
                std::cout << "Received Interest: " << interest.getName() << std::endl;
                
                // 创建Data包
                auto data = std::make_shared<Data>(interest.getName());
                data->setContent("Hello from Prima NDN!");
                data->setFreshnessPeriod(10_s);
                
                // 签名
                m_keyChain.sign(*data);
                
                // 发送Data
                m_face.put(*data);
                std::cout << "Sent Data: " << data->getName() << std::endl;
            });
        
        std::cout << "Producer registered prefix: /prima/test" << std::endl;
        
        // 运行事件循环
        m_ioContext.run();
    }
    
    void runConsumer() {
        std::cout << "Starting NDN Consumer..." << std::endl;
        
        // 等待一下让Producer启动
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        // 发送Interest
        Interest interest("/prima/test/hello");
        interest.setMustBeFresh(true);
        interest.setInterestLifetime(5_s);
        
        std::cout << "Sending Interest: " << interest.getName() << std::endl;
        
        m_face.expressInterest(interest,
            [](const Interest&, const Data& data) {
                std::cout << "Received Data: " << data.getName() << std::endl;
                std::cout << "Content: " << std::string(reinterpret_cast<const char*>(data.getContent().value()), 
                                                       data.getContent().value_size()) << std::endl;
            },
            [](const Interest&, const lp::Nack& nack) {
                std::cout << "Received Nack: " << nack.getReason() << std::endl;
            },
            [](const Interest&) {
                std::cout << "Interest timeout" << std::endl;
            });
        
        // 运行一段时间
        m_ioContext.run_for(std::chrono::seconds(10));
    }

private:
    boost::asio::io_context m_ioContext;
    Face m_face;
    KeyChain m_keyChain;
};

int main(int argc, char* argv[]) {
    try {
        NDNTest test;
        
        if (argc > 1 && std::string(argv[1]) == "producer") {
            test.runProducer();
        } else if (argc > 1 && std::string(argv[1]) == "consumer") {
            test.runConsumer();
        } else {
            std::cout << "Usage: " << argv[0] << " [producer|consumer]" << std::endl;
            std::cout << "Run producer in one terminal, consumer in another" << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}