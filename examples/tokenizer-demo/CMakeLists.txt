set(TARGET tokenizer-demo)

add_executable(${TARGET} tokenizer-demo.cpp)

target_link_libraries(${TARGET} PRIVATE llama common)

target_compile_features(${TARGET} PRIVATE cxx_std_11)

if(TARGET BUILD_INFO)
    add_dependencies(${TARGET} BUILD_INFO)
endif()

if(LLAMA_TOKENIZER)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_TOKENIZER)
endif()