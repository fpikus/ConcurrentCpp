# Machine-specific configuration -- compiler, -march target, standard, and the
# Google Benchmark / GoogleTest install paths -- comes from ../config.mk, shared
# by every benchmark directory and written once per machine. On Arm servers that
# config.mk resolves GBENCH_DIR/GTEST_DIR to the per-arch builds (gbench_arm,
# gtest_arm) via its own uname check; on x86 to gbench_x86; on a dev box to a
# flat path. Everything below this include is machine-independent and ships
# unchanged.
include ../config.mk

# Per-host build directory: several hosts build concurrently in the same
# NFS-shared tree, so binaries must not overwrite each other. Same convention as
# the Spinlock and DCLP directories.
BUILD_DIR = build/$(shell hostname)

# Common configuration
COMMON_FLAGS = -march=$(VEC) --std=$(STD) -g -Werror -Wall -Wextra -pedantic
COMMON_INCLUDES = -I.
# `?=` so a machine's config.mk can override: macOS has no librt.
COMMON_LIBS ?= -lpthread -lrt -lm

# Benchmark flags (optimized). NDEBUG keeps the queue's debug-only asserts out
# of the measured hot path; the tests (-O0, no NDEBUG) still run them.
CXXFLAGS_BM = $(COMMON_FLAGS) -O3 -DNDEBUG $(COMMON_INCLUDES) -I$(GBENCH_DIR)/include
LIBS_BM = $(GBENCH_DIR)/lib/libbenchmark.a $(COMMON_LIBS)

# Test flags
CXXFLAGS_TEST_ASAN = $(COMMON_FLAGS) -O0 -Wno-misleading-indentation -fsanitize=address $(COMMON_INCLUDES) -I$(GTEST_DIR)/include
CXXFLAGS_TEST_TSAN = $(COMMON_FLAGS) -O0 -Wno-misleading-indentation -fsanitize=thread $(COMMON_INCLUDES) -I$(GTEST_DIR)/include
LIBS_TEST = $(GTEST_DIR)/lib/libgtest.a $(GTEST_DIR)/lib/libgtest_main.a $(COMMON_LIBS)

# Target definitions (all under the per-host build directory)
TARGET_GMBM = $(BUILD_DIR)/concurrent_queue_gmbm
TARGET_LMBM = $(BUILD_DIR)/concurrent_queue_lmbm
TARGET_MBM = $(BUILD_DIR)/concurrent_queue_mbm
TARGET_PPMBM = $(BUILD_DIR)/concurrent_queue_ppmbm
TARGET_TEST_ASAN = $(BUILD_DIR)/concurrent_queue_test_asan
TARGET_TEST_TSAN = $(BUILD_DIR)/concurrent_queue_test_tsan

# `make benchmarks` needs only Google Benchmark; `make tests` needs GoogleTest
# and ASan/TSan. `make all` is both.
BENCHMARKS = $(TARGET_GMBM) $(TARGET_LMBM) $(TARGET_MBM) $(TARGET_PPMBM)
TESTS = $(TARGET_TEST_ASAN) $(TARGET_TEST_TSAN)

# Headers every target depends on; editing them must trigger a rebuild. Recipes
# use $< (the source file), not $^, so the headers are prerequisites only and
# are not passed to the compiler. The machine configuration is on the list for
# the same reason: a changed compiler or -march must rebuild everything.
HEADERS = concurrent_queue.h spinlock.h ../config.mk
# Used only by the two latency benchmarks (lmbm, ppmbm).
LAT_HEADERS = latbench_common.h

.PHONY: all benchmarks tests clean run_tests run_benchmarks

# The first rule in a Makefile is the default goal; without this line it would
# be the $(BUILD_DIR) rule below, and a bare `make` would only create the
# directory.
.DEFAULT_GOAL := all

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

all: benchmarks tests

benchmarks: $(BENCHMARKS)

tests: $(TESTS)

# Google Benchmark throughput benchmark
$(TARGET_GMBM): concurrent_queue_gmbm.C $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) -o $@

# Custom latency benchmark (push->pop handoff under contention)
$(TARGET_LMBM): concurrent_queue_lmbm.C $(HEADERS) $(LAT_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) -o $@

# Custom hand-rolled throughput benchmark (gmbm's twin)
$(TARGET_MBM): concurrent_queue_mbm.C $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) -o $@

# Ping-pong latency benchmark (1P/1C round trip)
$(TARGET_PPMBM): concurrent_queue_ppmbm.C $(HEADERS) $(LAT_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) -o $@

# ASan test
$(TARGET_TEST_ASAN): concurrent_queue_test.C $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_TEST_ASAN) $< $(LIBS_TEST) -o $@

# TSan test
$(TARGET_TEST_TSAN): concurrent_queue_test.C $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_TEST_TSAN) $< $(LIBS_TEST) -o $@

# Run tests
run_tests: tests
	./$(TARGET_TEST_ASAN)
	./$(TARGET_TEST_TSAN)

# Run benchmarks
run_benchmarks: benchmarks
	./$(TARGET_GMBM)
	./$(TARGET_LMBM)
	./$(TARGET_MBM)
	./$(TARGET_PPMBM)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) *.bak *~
