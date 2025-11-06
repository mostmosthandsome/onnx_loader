# === 项目信息 ===
PROJECT = onnx_load
CXXFLAGS = -Wall -O3 -std=c++17 -mfloat-abi=hard -mfpu=neon -mtune=cortex-a9 -fpermissive

# Protobuf 安装路径
PROTOBUF_DIR = ../onnx_loader_third_parties/install

INCLUDES = \
    -I../vip_driver/sdk/include \
    -I./include \
    -I$(PROTOBUF_DIR)/include

LDFLAGS = \
    -L../vip_driver/sdk/drivers \
    -L$(PROTOBUF_DIR)/lib
LDLIBS = -lOpenCL -lprotobuf  -lCLC -lVSC -lGAL -lSPIRV_viv \
-labsl_log_internal_check_op -labsl_die_if_null -labsl_log_internal_conditions -labsl_log_internal_message -labsl_examine_stack -labsl_log_internal_format -labsl_log_internal_nullguard -labsl_log_internal_structured_proto -labsl_log_internal_log_sink_set -labsl_log_sink -labsl_log_entry -labsl_log_internal_proto -labsl_flags_internal -labsl_flags_marshalling -labsl_flags_reflection -labsl_flags_private_handle_accessor -labsl_flags_commandlineflag -labsl_flags_commandlineflag_internal -labsl_flags_config -labsl_flags_program_name -labsl_log_initialize -labsl_log_internal_globals -labsl_log_globals -labsl_vlog_config_internal -labsl_log_internal_fnmatch -labsl_raw_hash_set -labsl_hashtablez_sampler -labsl_random_distributions -labsl_random_seed_sequences -labsl_random_internal_entropy_pool -labsl_random_internal_randen -labsl_random_internal_randen_hwaes -labsl_random_internal_randen_hwaes_impl -labsl_random_internal_randen_slow -labsl_random_internal_platform -labsl_random_internal_seed_material -labsl_random_seed_gen_exception -labsl_statusor -labsl_status -labsl_cord -labsl_cordz_info -labsl_cord_internal -labsl_hash -labsl_city -labsl_cordz_functions -labsl_exponential_biased -labsl_cordz_handle -labsl_crc_cord_state -labsl_crc32c -labsl_crc_internal -labsl_crc_cpu_detect -labsl_leak_check -labsl_strerror -labsl_str_format_internal -labsl_synchronization -labsl_graphcycles_internal -labsl_kernel_timeout_internal -labsl_stacktrace -labsl_symbolize -labsl_debugging_internal -labsl_demangle_internal -labsl_demangle_rust -labsl_decode_rust_punycode -labsl_utf8_for_code_point -labsl_malloc_internal -labsl_tracing_internal -labsl_time -labsl_civil_time -labsl_time_zone -lutf8_validity -lutf8_range -labsl_strings -labsl_strings_internal -labsl_string_view -labsl_int128 -labsl_base -lrt -labsl_spinlock_wait -labsl_throw_delegate -labsl_raw_logging_internal -labsl_log_severity

# === 路径定义 ===
SRC_DIR = src
TEST_DIR = test
BUILD_DIR = build

# === 源文件定义 ===
CUSTOM_KERNEL_SRC = \
    $(SRC_DIR)/onnx.pb.cc \
    $(SRC_DIR)/CustomKernel.cpp \
    $(SRC_DIR)/OnnxLoader.cpp \
    $(SRC_DIR)/ExploreVaeRunner.cpp \
    $(SRC_DIR)/PolicyHtOriRunner.cpp



CUSTOM_KERNEL_OBJ = $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CUSTOM_KERNEL_SRC))
CUSTOM_KERNEL_OBJ := $(patsubst $(SRC_DIR)/%.cc,$(BUILD_DIR)/%.o,$(CUSTOM_KERNEL_OBJ))


MAIN_SRC = $(SRC_DIR)/main.cpp
MAIN_OBJ = $(MAIN_SRC:$(SRC_DIR)/%.cpp=$(BUILD_DIR)/%.o)

MODEL_CHECK_SRC = $(TEST_DIR)/model_check.cpp $(SRC_DIR)/onnx.pb.cc

MODEL_CHECK_OBJ = $(patsubst $(TEST_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(filter %.cpp,$(MODEL_CHECK_SRC)))
MODEL_CHECK_OBJ += $(patsubst $(SRC_DIR)/%.cc,$(BUILD_DIR)/%.o,$(filter %.cc,$(MODEL_CHECK_SRC)))


# === 目标文件（输出路径统一放到 build/ 下）===
# 静态库
LIB_TARGET = $(BUILD_DIR)/libCustomKernel.a
# main
MAIN_TARGET = $(BUILD_DIR)/main
# test
MODEL_CHECK_TARGET = $(BUILD_DIR)/model_check

# === 默认规则 ===
all: $(LIB_TARGET) $(MAIN_TARGET) $(MODEL_CHECK_TARGET)

# === 静态库 ===
$(LIB_TARGET): $(CUSTOM_KERNEL_OBJ)
	@echo " 生成静态库 $@"
	$(AR) rcs $@ $^

# === 主程序 ===
$(MAIN_TARGET): $(MAIN_OBJ) $(LIB_TARGET)
	@echo " 编译主程序 $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ $(LDFLAGS) $(LDLIBS) -o $@

# === 模型检查程序 ===
$(MODEL_CHECK_TARGET): $(MODEL_CHECK_OBJ)
	@echo "🧪 编译测试程序 $@"
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ $(LDFLAGS) $(LDLIBS) -o $@

# === 通用编译规则 ===
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	@echo "🧩 编译 $<"
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cc | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	@echo "🧩 编译 $<"
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(TEST_DIR)/%.cpp | $(BUILD_DIR)
	@mkdir -p $(dir $@)
	@echo "🧩 编译 $<"
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@


# === 创建 build 目录 ===
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

# === 清理规则 ===
clean:
	@echo "🧹 清理生成文件"
	rm -rf $(BUILD_DIR)

.PHONY: all clean
