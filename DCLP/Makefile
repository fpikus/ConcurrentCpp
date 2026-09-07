# Machine-specific configuration -- compiler, -march target, standard, library
# paths -- comes from ../config.mk, shared by all benchmark directories and
# written once per machine. Everything below this include is machine-
# independent, so this Makefile ships to any machine unchanged.
include ../config.mk

# Per-host build directory: several hosts build concurrently in the same
# NFS-shared tree, so binaries must not overwrite each other. Same convention
# as the Spinlock directory: per-host subdirectories under build/, and clean
# removes only this host's subdirectory.
BUILD_DIR = build/$(shell hostname)

# Common configuration. -I../Spinlock so atomic_max_bm.C can include the shipped
# SpinLock (spinlock.h) instead of a copy.
COMMON_FLAGS = -march=$(VEC) --std=$(STD) -g -Werror -Wall -Wextra -pedantic
COMMON_INCLUDES = -I. -I../Spinlock
# `?=` so a machine's config.mk can override: macOS has no librt, so an M3's
# config.mk sets COMMON_LIBS = -lpthread -lm.
COMMON_LIBS ?= -lpthread -lrt -lm

# Benchmark flags (optimized)
CXXFLAGS_BM = $(COMMON_FLAGS) -O3 -DNDEBUG $(COMMON_INCLUDES) -I$(GBENCH_DIR)/include
LIBS_BM = $(GBENCH_DIR)/lib/libbenchmark.a $(COMMON_LIBS)

# Test flags
CXXFLAGS_TEST_ASAN = $(COMMON_FLAGS) -O0 -Wno-misleading-indentation -fsanitize=address $(COMMON_INCLUDES) -I$(GTEST_DIR)/include
CXXFLAGS_TEST_TSAN = $(COMMON_FLAGS) -O0 -Wno-misleading-indentation -fsanitize=thread $(COMMON_INCLUDES) -I$(GTEST_DIR)/include
LIBS_TEST = $(GTEST_DIR)/lib/libgtest.a $(GTEST_DIR)/lib/libgtest_main.a $(COMMON_LIBS)

# Target definitions (all under the per-host build directory)
TARGET_BM = $(BUILD_DIR)/atomic_max_bm
TARGET_TEST_ASAN = $(BUILD_DIR)/atomic_max_test_asan
TARGET_TEST_TSAN = $(BUILD_DIR)/atomic_max_test_tsan

# The two groups: `make benchmarks` needs only Google Benchmark and works on
# machines without sanitizer runtimes; `make tests` needs GoogleTest and
# ASan/TSan. `make all` is both.
BENCHMARKS = $(TARGET_BM)
TESTS = $(TARGET_TEST_ASAN) $(TARGET_TEST_TSAN)

# Headers every target depends on; editing them must trigger a rebuild. Recipes
# use $< (the source file), not $^, so the headers are prerequisites only and
# are not passed to the compiler command line. The machine configuration and
# the shipped SpinLock are on the list for the same reason: a change to either
# must rebuild what uses it.
HEADERS = atomic_max.h ../Spinlock/spinlock.h ../config.mk

.PHONY: all benchmarks tests clean run_tests run_benchmarks

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

all: benchmarks tests

benchmarks: $(BENCHMARKS)

tests: $(TESTS)

# Build benchmark (lock-free vs spinlocked vs DCLP running maximum)
$(TARGET_BM): atomic_max_bm.C $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) -o $@

# Build ASan test (atomic_max correctness + ordering)
$(TARGET_TEST_ASAN): atomic_max_test.C $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_TEST_ASAN) $< $(LIBS_TEST) -o $@

# Build TSan test (validates the acquire/acq_rel handoff)
$(TARGET_TEST_TSAN): atomic_max_test.C $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_TEST_TSAN) $< $(LIBS_TEST) -o $@

# Run tests
run_tests: tests
	./$(TARGET_TEST_ASAN)
	./$(TARGET_TEST_TSAN)

# Run benchmarks
run_benchmarks: benchmarks
	./$(TARGET_BM)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) *.bak *~
