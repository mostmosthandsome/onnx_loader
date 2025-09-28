cmake_minimum_required(VERSION 3.10)
project(onnx_load)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Protobuf REQUIRED CONFIG)
# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -mfloat-abi=hard -mfpu=neon -mtune=cortex-a9 -fpermissive")

set(Custom_KERNEL_FILES
    src/main.cpp
    src/onnx.pb.cc
    src/CustomKernel.cpp
    src/OnnxLoader.cpp
)

# 包含路径
include_directories(
    /root/workspace/04_opencl_demo/vip_driver/sdk/include #OpenCL的include目录
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(CustomKernel ${Custom_KERNEL_FILES})

# 链接库路径
link_directories(
    /root/workspace/04_opencl_demo/vip_driver/sdk/drivers #libOpenCL.so所在目录
)

#编译成库
target_link_libraries(CustomKernel PRIVATE OpenCL protobuf::libprotobuf)

# 标准输入输出测试样例
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE CustomKernel)

#测试
add_executable(model_check test/model_check.cpp  src/onnx.pb.cc)
target_link_libraries(model_check PRIVATE protobuf::libprotobuf)
