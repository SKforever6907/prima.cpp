# TODO: should not use this
if (WIN32)
    if (BUILD_SHARED_LIBS)
        set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()
endif()

#
# libraries
#

# llama

set(LLAMA_SOURCES
    ../include/llama.h
    llama.cpp
    llama-vocab.cpp
    llama-grammar.cpp
    llama-sampling.cpp
    unicode.h
    unicode.cpp
    unicode-data.cpp
)

# Add NDN source file if NDN is enabled
if (LLAMA_NDN)
    list(APPEND LLAMA_SOURCES ndn-prima.cpp)
endif()

# Add tokenizer source file if tokenizer is enabled
if (LLAMA_TOKENIZER)
    list(APPEND LLAMA_SOURCES llama-tokenizer.cpp)
endif()

add_library(llama ${LLAMA_SOURCES})

target_include_directories(llama PUBLIC . ../include)
target_compile_features   (llama PUBLIC cxx_std_11) # don't bump

target_link_libraries(llama PUBLIC ggml)

# Link NDN libraries if NDN is enabled
if (LLAMA_NDN)
    target_link_libraries(llama PUBLIC ${LLAMA_NDN_LIBRARIES})
    target_link_directories(llama PUBLIC ${LLAMA_NDN_LIBRARY_DIRS})
    # NDN requires C++17
    target_compile_features(llama PUBLIC cxx_std_17)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties(llama PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(llama PRIVATE LLAMA_SHARED LLAMA_BUILD)
endif()
