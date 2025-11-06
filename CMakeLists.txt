cmake_minimum_required(VERSION 3.10)
project(onnx_load)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -mfloat-abi=hard -mfpu=neon -mtune=cortex-a9 -fpermissive")

set(Custom_KERNEL_FILES
    src/main.cpp
    src/onnx.pb.cc
    src/CustomKernel.cpp
    src/OnnxLoader.cpp
    src/ExploreVaeRunner.cpp
    src/PolicyHtOriRunner.cpp
)

set(OPENCL_LINK_LIB
-lOpenCL -lprotobuf  -lCLC -lVSC -lGAL -lSPIRV_viv 
)

set(PROTOBUF_LINK_LIB
-lprotobuf -labsl_log_internal_check_op -labsl_die_if_null -labsl_log_internal_conditions -labsl_log_internal_message -labsl_examine_stack -labsl_log_internal_format -labsl_log_internal_nullguard -labsl_log_internal_structured_proto -labsl_log_internal_log_sink_set -labsl_log_sink -labsl_log_entry -labsl_log_internal_proto -labsl_flags_internal -labsl_flags_marshalling -labsl_flags_reflection -labsl_flags_private_handle_accessor -labsl_flags_commandlineflag -labsl_flags_commandlineflag_internal -labsl_flags_config -labsl_flags_program_name -labsl_log_initialize -labsl_log_internal_globals -labsl_log_globals -labsl_vlog_config_internal -labsl_log_internal_fnmatch -labsl_raw_hash_set -labsl_hashtablez_sampler -labsl_random_distributions -labsl_random_seed_sequences -labsl_random_internal_entropy_pool -labsl_random_internal_randen -labsl_random_internal_randen_hwaes -labsl_random_internal_randen_hwaes_impl -labsl_random_internal_randen_slow -labsl_random_internal_platform -labsl_random_internal_seed_material -labsl_random_seed_gen_exception -labsl_statusor -labsl_status -labsl_cord -labsl_cordz_info -labsl_cord_internal -labsl_hash -labsl_city -labsl_cordz_functions -labsl_exponential_biased -labsl_cordz_handle -labsl_crc_cord_state -labsl_crc32c -labsl_crc_internal -labsl_crc_cpu_detect -labsl_leak_check -labsl_strerror -labsl_str_format_internal -labsl_synchronization -labsl_graphcycles_internal -labsl_kernel_timeout_internal -labsl_stacktrace -labsl_symbolize -labsl_debugging_internal -labsl_demangle_internal -labsl_demangle_rust -labsl_decode_rust_punycode -labsl_utf8_for_code_point -labsl_malloc_internal -labsl_tracing_internal -labsl_time -labsl_civil_time -labsl_time_zone -lutf8_validity -lutf8_range -labsl_strings -labsl_strings_internal -labsl_string_view -labsl_int128 -labsl_base -lrt -labsl_spinlock_wait -labsl_throw_delegate -labsl_raw_logging_internal -labsl_log_severity
)

# 包含路径
include_directories(
    ../vip_driver/sdk/include #OpenCL的include目录
    ../onnx_loader_third_parties/install/include #第三方库protobuf
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_library(CustomKernel ${Custom_KERNEL_FILES})

# 链接库路径
link_directories(
    ../vip_driver/sdk/drivers #libOpenCL.so所在目录
    ../onnx_loader_third_parties/install/lib
)

#编译成库
target_link_libraries(CustomKernel PRIVATE ${OPENCL_LINK_LIB} ${PROTOBUF_LINK_LIB})

# 标准输入输出测试样例
add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE CustomKernel)

#测试
add_executable(model_check test/model_check.cpp  src/onnx.pb.cc)
target_link_libraries(model_check PRIVATE ${PROTOBUF_LINK_LIB})

add_executable(kernel_test test/kernel_test.cpp)
target_link_libraries(kernel_test PRIVATE ${OPENCL_LINK_LIB})