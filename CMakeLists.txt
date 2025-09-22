cmake_minimum_required(VERSION 3.10)
project(onnx_load)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
find_package(Protobuf REQUIRED CONFIG)
# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -mfloat-abi=hard -mfpu=neon -mtune=cortex-a9 -fpermissive")

set(SRC_FILES
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

# 链接库路径
link_directories(
    /root/workspace/04_opencl_demo/vip_driver/sdk/drivers #libOpenCL.so所在目录
)

# 可执行文件
add_executable(onnx_load ${SRC_FILES})

# 链接
target_link_libraries(onnx_load PRIVATE OpenCL protobuf::libprotobuf)