# Machine-specific configuration -- compiler, -march target, standard, library
# paths -- comes from ../config.mk, shared by all benchmark directories and
# written once per machine. Everything below this include is machine-
# independent, so this Makefile ships to any machine unchanged.
include ../config.mk

# Per-host build directory: several hosts build concurrently in the same
# NFS-shared tree, so binaries must not overwrite each other. Same convention
# as results/<hostname>/ in run_sets.sh: per-host subdirectories under build/.
# clean removes only this host's subdirectory, never the other hosts'.
BUILD_DIR = build/$(shell hostname)

# Common configuration
COMMON_FLAGS = -march=$(VEC) --std=$(STD) -g -Werror -Wall -Wextra -pedantic
COMMON_INCLUDES = -I.
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
TARGET_BM = $(BUILD_DIR)/spinlock_bm
TARGET_TUNE_BM = $(BUILD_DIR)/spinlock_tune_bm
TARGET_RW_TUNE_BM = $(BUILD_DIR)/spinlock_rw_tune_bm
TARGET_OVERHEAD_BM = $(BUILD_DIR)/overhead_bm
TARGET_MEM_BM = $(BUILD_DIR)/spinlock_mem_bm
TARGET_LAYOUT_BM = $(BUILD_DIR)/spinlock_layout_bm
TARGET_SEQLOCK_BM = $(BUILD_DIR)/seqlock_bm
TARGET_TEST_ASAN = $(BUILD_DIR)/spinlock_test_asan
TARGET_TEST_TSAN = $(BUILD_DIR)/spinlock_test_tsan
TARGET_SEQLOCK_TEST_ASAN = $(BUILD_DIR)/seqlock_test_asan
TARGET_SEQLOCK_TEST_TSAN = $(BUILD_DIR)/seqlock_test_tsan

# The two groups: `make benchmarks` needs only Google Benchmark and works on
# machines without sanitizer runtimes; `make tests` needs GoogleTest and
# ASan/TSan. `make all` is both.
BENCHMARKS = $(TARGET_BM) $(TARGET_TUNE_BM) $(TARGET_RW_TUNE_BM) \
             $(TARGET_OVERHEAD_BM) $(TARGET_MEM_BM) $(TARGET_LAYOUT_BM) \
             $(TARGET_SEQLOCK_BM)
TESTS = $(TARGET_TEST_ASAN) $(TARGET_TEST_TSAN) \
        $(TARGET_SEQLOCK_TEST_ASAN) $(TARGET_SEQLOCK_TEST_TSAN)

# Headers every target depends on; editing them must trigger a rebuild.
# Recipes use $< (the source file), not $^, so the headers are prerequisites
# only and are not passed to the compiler command line. The machine
# configuration is on the list for the same reason: a changed compiler or
# -march must rebuild everything, or stale binaries get measured.
HEADERS = spinlock.h spinlock_bm_common.h ../config.mk
# Headers used only by the tuning benchmark; kept out of HEADERS so that
# editing the tunable lock does not rebuild the tests and the main benchmark.
TUNE_HEADERS = spinlock_tune.h
# The shared ladder-configuration list, needed only by the two sweep benchmarks.
TUNE_CONFIGS = spinlock_tune_configs.h
# The sequence lock. It includes spinlock_tune.h (for spin_pause()), so its
# users list $(TUNE_HEADERS) as well.
SEQLOCK_HEADERS = seqlock.h

.PHONY: all benchmarks tests clean run_tests run_benchmarks

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

all: benchmarks tests

benchmarks: $(BENCHMARKS)

tests: $(TESTS)

# Build benchmark (SpinLock vs atomic and mutex across the contention range)
$(TARGET_BM): spinlock_bm.C $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) -o $@

# Build tuning benchmark (back-off ladder parameter exploration)
$(TARGET_TUNE_BM): spinlock_tune_bm.C $(HEADERS) $(TUNE_HEADERS) $(TUNE_CONFIGS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) -o $@

# Build read/write tuning benchmark (separate ladders for readers and writers)
$(TARGET_RW_TUNE_BM): spinlock_rw_tune_bm.C $(HEADERS) $(TUNE_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) -o $@

# Build overhead benchmark (spinlock vs atomic vs CAS loop across the work dial)
$(TARGET_OVERHEAD_BM): overhead_bm.C $(HEADERS) $(TUNE_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) -o $@

# Build memory-work tuning benchmark (the ladder sweep against streaming work)
$(TARGET_MEM_BM): spinlock_mem_bm.C $(HEADERS) $(TUNE_HEADERS) $(TUNE_CONFIGS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) -o $@

# Build lock/payload cache-line layout benchmark
$(TARGET_LAYOUT_BM): spinlock_layout_bm.C $(HEADERS) $(TUNE_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) -o $@

# Build sequence-lock benchmark (seqlock vs spinlock vs atomic, read/write mix)
$(TARGET_SEQLOCK_BM): seqlock_bm.C $(HEADERS) $(TUNE_HEADERS) $(SEQLOCK_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) -o $@

# Build ASan test
$(TARGET_TEST_ASAN): spinlock_test.C $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_TEST_ASAN) $< $(LIBS_TEST) -o $@

# Build TSan test
$(TARGET_TEST_TSAN): spinlock_test.C $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_TEST_TSAN) $< $(LIBS_TEST) -o $@

# Build SeqLock ASan test
$(TARGET_SEQLOCK_TEST_ASAN): seqlock_test.C $(HEADERS) $(TUNE_HEADERS) $(SEQLOCK_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_TEST_ASAN) $< $(LIBS_TEST) -o $@

# Build SeqLock TSan test
$(TARGET_SEQLOCK_TEST_TSAN): seqlock_test.C $(HEADERS) $(TUNE_HEADERS) $(SEQLOCK_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_TEST_TSAN) $< $(LIBS_TEST) -o $@

# Run tests
run_tests: tests
	./$(TARGET_TEST_ASAN)
	./$(TARGET_TEST_TSAN)
	./$(TARGET_SEQLOCK_TEST_ASAN)
	./$(TARGET_SEQLOCK_TEST_TSAN)

# Run benchmarks
run_benchmarks: benchmarks
	./$(TARGET_BM)
	./$(TARGET_TUNE_BM)
	./$(TARGET_RW_TUNE_BM)
	./$(TARGET_OVERHEAD_BM)
	./$(TARGET_MEM_BM)
	./$(TARGET_SEQLOCK_BM)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) *.bak *~
