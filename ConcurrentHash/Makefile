# Machine-specific configuration -- compiler, -march target, standard, and the
# Google Benchmark / GoogleTest install paths -- comes from ../config.mk, shared
# by every benchmark directory and written once per machine. Everything below
# this include is machine-independent and ships unchanged.
include ../config.mk

# Per-host build directory: several hosts build concurrently in the same
# NFS-shared tree, so binaries must not overwrite each other. Same convention as
# the Spinlock, DCLP, ConcurrentDeque, and ConcurrentQueue directories: per-host
# subdirectories under build/, and clean removes only this host's subdirectory.
BUILD_DIR = build/$(shell hostname)

# Common configuration
COMMON_FLAGS = -march=$(VEC) --std=$(STD) -g -Werror -Wall -Wextra -pedantic
COMMON_INCLUDES = -I.
# `?=` so a machine's config.mk can override: macOS has no librt.
COMMON_LIBS ?= -lpthread -lrt -lm

# Benchmark flags (optimized)
CXXFLAGS_BM = $(COMMON_FLAGS) -O3 -DNDEBUG $(COMMON_INCLUDES) -I$(GBENCH_DIR)/include
LIBS_BM = $(GBENCH_DIR)/lib/libbenchmark.a $(COMMON_LIBS)

# Test flags
CXXFLAGS_TEST_ASAN = $(COMMON_FLAGS) -O0 -Wno-misleading-indentation -fsanitize=address $(COMMON_INCLUDES) -I$(GTEST_DIR)/include
CXXFLAGS_TEST_TSAN = $(COMMON_FLAGS) -O0 -Wno-misleading-indentation -fsanitize=thread $(COMMON_INCLUDES) -I$(GTEST_DIR)/include
# The store-buffering regression test is the one test in this directory that must
# NOT be a sanitizer build. It observes a real store-buffering execution (the
# resizer's plain store of table_size_ still in its store buffer when it reads a
# bucket head); both sanitizer runtimes execute locked instructions all over that
# path and -O0 stretches it by hundreds of instructions, either of which drains
# the buffer before the window opens. Built under ASan/TSan or at -O0 the test can
# only ever pass, which would make it worthless. Same optimization settings as the
# benchmarks, same GoogleTest link as the other tests.
CXXFLAGS_TEST_TSO = $(COMMON_FLAGS) -O3 -DNDEBUG $(COMMON_INCLUDES) -I$(GTEST_DIR)/include
LIBS_TEST = $(GTEST_DIR)/lib/libgtest.a $(GTEST_DIR)/lib/libgtest_main.a $(COMMON_LIBS)

# Target definitions (all under the per-host build directory)
TARGET_BM = $(BUILD_DIR)/concurrent_hash_set_bm
TARGET_TEST_ASAN = $(BUILD_DIR)/concurrent_hash_set_test_asan
TARGET_TEST_TSAN = $(BUILD_DIR)/concurrent_hash_set_test_tsan
TARGET_TEST_TSO = $(BUILD_DIR)/concurrent_hash_set_tso_test

# `make benchmarks` needs only Google Benchmark and builds on machines without
# sanitizer runtimes; `make tests` needs GoogleTest and ASan/TSan. `make all` is
# both.
BENCHMARKS = $(TARGET_BM)
TESTS = $(TARGET_TEST_ASAN) $(TARGET_TEST_TSAN) $(TARGET_TEST_TSO)

# Headers every target depends on; editing them must trigger a rebuild. Recipes
# use $< (the source file), not $^, so the headers are prerequisites only and
# are not passed to the compiler. The machine configuration is on the list for
# the same reason: a changed compiler or -march must rebuild everything, or
# stale binaries get measured.
# concurrent_deque.h and spinlock.h are symlinks into ../ConcurrentDeque and
# ../Spinlock; make follows them, so a change to the shipped header rebuilds here.
HEADERS = concurrent_deque.h concurrent_hash_set.h spinlock.h ../config.mk

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

# Build benchmark
$(TARGET_BM): concurrent_hash_set_bm.C $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) -o $@

# Build ASan test
$(TARGET_TEST_ASAN): concurrent_hash_set_test.C $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_TEST_ASAN) $< $(LIBS_TEST) -o $@

# Build TSan test
$(TARGET_TEST_TSAN): concurrent_hash_set_test.C $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_TEST_TSAN) $< $(LIBS_TEST) -o $@

# Build the -O3, no-sanitizer store-buffering test (see CXXFLAGS_TEST_TSO above)
$(TARGET_TEST_TSO): concurrent_hash_set_tso_test.C $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_TEST_TSO) $< $(LIBS_TEST) -o $@

# Run tests
run_tests: tests
	./$(TARGET_TEST_ASAN)
	./$(TARGET_TEST_TSAN)
	./$(TARGET_TEST_TSO)

# Run benchmarks
run_benchmarks: benchmarks
	./$(TARGET_BM)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) *.bak *~
