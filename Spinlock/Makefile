# Machine-specific configuration -- compiler, -march target, standard, library
# paths -- comes from ../config.mk, shared by all benchmark directories and
# written once per machine. Everything below this include is machine-
# independent, so this Makefile ships to any machine unchanged.
include ../config.mk

# Compilers and run-time library paths. The per-machine config may select a
# toolset compiler whose libstdc++ is not on the default library path; a binary
# it links then builds cleanly and fails at run time with "GLIBCXX_x.y.z not
# found" against the system libstdc++. So every link embeds an rpath to the
# directory of the libstdc++ that the linking compiler itself uses, derived from
# the compiler rather than from config.mk so it is right for any compiler on any
# machine: `-print-file-name=libstdc++.so` prints that library's full path, or
# the bare name when the compiler knows none, in which case RPATH_OF adds no
# flag. For a system compiler the directory may hold only the link-time symlink;
# the loader finds no libstdc++.so.6 there and falls through to the default path.
# The sanitizer tests are built with SAN_CXX, which config.mk may point at a
# separate toolchain and which defaults to CXX; a config.mk that sets SAN_RPATH
# overrides the rpath derived for SAN_CXX.
# The commas of the flag go through $(comma): a literal one would split the
# arguments of $(if).
SAN_CXX ?= $(CXX)
comma := ,
RPATH_OF = $(strip $(if $(findstring /,$(1)),-Wl$(comma)-rpath$(comma)$(dir $(1))))
CXX_RPATH := $(call RPATH_OF,$(shell $(CXX) -print-file-name=libstdc++.so))
SAN_CXX_RPATH := $(if $(SAN_RPATH),$(SAN_RPATH),$(call RPATH_OF,$(shell $(SAN_CXX) -print-file-name=libstdc++.so)))

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
TARGET_SPINLOCK_SCOPE_BM = $(BUILD_DIR)/spinlock_scope_bm
TARGET_SPINLOCK_SCOPE_MBM = $(BUILD_DIR)/spinlock_scope_mbm

# The two groups: `make benchmarks` needs only Google Benchmark and works on
# machines without sanitizer runtimes; `make tests` needs GoogleTest and
# ASan/TSan. `make all` is both.
BENCHMARKS = $(TARGET_BM) $(TARGET_TUNE_BM) $(TARGET_RW_TUNE_BM) \
             $(TARGET_OVERHEAD_BM) $(TARGET_MEM_BM) $(TARGET_LAYOUT_BM) \
             $(TARGET_SEQLOCK_BM) $(TARGET_SPINLOCK_SCOPE_BM) \
             $(TARGET_SPINLOCK_SCOPE_MBM)
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
# The lock scope benchmark's shared part: slot types, layout, the measured
# operation and the variant list, used by its Google Benchmark harness and by
# the hand-rolled twin.
SCOPE_HEADERS = spinlock_scope_common.h

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

# Build benchmark (SpinLock vs atomic and mutex across the contention range)
$(TARGET_BM): spinlock_bm.C $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) $(CXX_RPATH) -o $@

# Build tuning benchmark (back-off ladder parameter exploration)
$(TARGET_TUNE_BM): spinlock_tune_bm.C $(HEADERS) $(TUNE_HEADERS) $(TUNE_CONFIGS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) $(CXX_RPATH) -o $@

# Build read/write tuning benchmark (separate ladders for readers and writers)
$(TARGET_RW_TUNE_BM): spinlock_rw_tune_bm.C $(HEADERS) $(TUNE_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) $(CXX_RPATH) -o $@

# Build overhead benchmark (spinlock vs atomic vs CAS loop across the work dial)
$(TARGET_OVERHEAD_BM): overhead_bm.C $(HEADERS) $(TUNE_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) $(CXX_RPATH) -o $@

# Build memory-work tuning benchmark (the ladder sweep against streaming work)
$(TARGET_MEM_BM): spinlock_mem_bm.C $(HEADERS) $(TUNE_HEADERS) $(TUNE_CONFIGS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) $(CXX_RPATH) -o $@

# Build lock/payload cache-line layout benchmark
$(TARGET_LAYOUT_BM): spinlock_layout_bm.C $(HEADERS) $(TUNE_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) $(CXX_RPATH) -o $@

# Build sequence-lock benchmark (seqlock vs spinlock vs atomic, read/write mix)
$(TARGET_SEQLOCK_BM): seqlock_bm.C $(HEADERS) $(TUNE_HEADERS) $(SEQLOCK_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) $(CXX_RPATH) -o $@

# Build lock scope benchmark (slot store before vs after unlock)
$(TARGET_SPINLOCK_SCOPE_BM): spinlock_scope_bm.C $(HEADERS) $(SCOPE_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) $(CXX_RPATH) -o $@

# Build the hand-rolled twin of the lock scope benchmark (every thread runs
# until a common stop). It does not use Google Benchmark, so it links only the
# common libraries.
$(TARGET_SPINLOCK_SCOPE_MBM): spinlock_scope_mbm.C $(HEADERS) $(SCOPE_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(COMMON_LIBS) $(CXX_RPATH) -o $@

# Build ASan test
$(TARGET_TEST_ASAN): spinlock_test.C $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_ASAN) $< $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Build TSan test
$(TARGET_TEST_TSAN): spinlock_test.C $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_TSAN) $< $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Build SeqLock ASan test
$(TARGET_SEQLOCK_TEST_ASAN): seqlock_test.C $(HEADERS) $(TUNE_HEADERS) $(SEQLOCK_HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_ASAN) $< $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Build SeqLock TSan test
$(TARGET_SEQLOCK_TEST_TSAN): seqlock_test.C $(HEADERS) $(TUNE_HEADERS) $(SEQLOCK_HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_TSAN) $< $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Run tests
run_tests: tests
	./$(TARGET_TEST_ASAN)
	./$(TARGET_TEST_TSAN)
	./$(TARGET_SEQLOCK_TEST_ASAN)
	./$(TARGET_SEQLOCK_TEST_TSAN)

# Run benchmarks. spinlock_scope_bm and spinlock_scope_mbm are left out: one
# pass over all their cells takes minutes on the large servers, and
# runs with repetitions take proportionally longer.
run_benchmarks: benchmarks
	./$(TARGET_BM)
	./$(TARGET_TUNE_BM)
	./$(TARGET_RW_TUNE_BM)
	./$(TARGET_OVERHEAD_BM)
	./$(TARGET_MEM_BM)
	./$(TARGET_LAYOUT_BM)
	./$(TARGET_SEQLOCK_BM)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) *.bak *~
