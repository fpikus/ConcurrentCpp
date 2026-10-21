# Machine-specific configuration -- compiler, -march target, standard, and the
# Google Benchmark / GoogleTest install paths -- comes from ../config.mk, shared
# by every benchmark directory and written once per machine. Everything below
# this include is machine-independent and ships unchanged.
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
# NFS-shared tree, so binaries must not overwrite each other. Same convention as
# the Spinlock, DCLP, ConcurrentDeque, and ConcurrentQueue directories: per-host
# subdirectories under build/, and clean removes only this host's subdirectory.
BUILD_DIR = build/$(shell hostname)

# Common configuration
COMMON_FLAGS = -march=$(VEC) --std=$(STD) -g -Werror -Wall -Wextra -pedantic
COMMON_INCLUDES = -I.
# `?=` so a machine's config.mk can override: macOS has no librt.
COMMON_LIBS ?= -lpthread -lrt -lm

# Benchmark flags (optimized). -I../Spinlock reaches the wall-clock accounting
# header (gb_wall_clock.h) of the benchmark; on the benchmark flags only, since
# the tests do not use it.
CXXFLAGS_BM = $(COMMON_FLAGS) -O3 -DNDEBUG $(COMMON_INCLUDES) -I../Spinlock -I$(GBENCH_DIR)/include
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
TARGET_BM = $(BUILD_DIR)/concurrent_hash_set_rcu_bm
TARGET_TEST_ASAN = $(BUILD_DIR)/concurrent_hash_set_rcu_test_asan
TARGET_TEST_TSAN = $(BUILD_DIR)/concurrent_hash_set_rcu_test_tsan
TARGET_TEST_TSO = $(BUILD_DIR)/concurrent_hash_set_rcu_tso_test

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
HEADERS = concurrent_deque.h concurrent_hash_set_rcu.h spinlock.h ../config.mk
# The wall-clock accounting header of the benchmark. A variable of its own
# rather than an entry in HEADERS: HEADERS is a prerequisite of the tests, which
# do not use Google Benchmark and must not depend on this header.
GB_HEADERS = ../Spinlock/gb_wall_clock.h

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

# Build the benchmark: one object per translation unit, linked into one binary.
# The benchmark is several units over its shared header (BM_HEADER), one per
# family of rows, because a compiler's inlining decisions depend on the whole
# unit (GCC's inliner has a unit-wide growth budget): with every row in one
# unit, which calls are inlined into a row's timed loop depends on the code of
# every other row, and a change in one row can change the code of rows that do
# not contain it. As separate units, the code generated for a row depends on
# its own unit alone. Every unit depends on the headers of the other targets
# plus BM_HEADER and is compiled with the benchmark flags (CXXFLAGS_BM).
# BM_SOURCES is in the order the rows are listed and run: Google Benchmark
# runs the rows in registration order, which is the order of the static
# initializers, across units the link order; $(BM_OBJECTS) keeps this order on
# the link line. The first unit, the index and main(), registers no row.
BM_HEADER = concurrent_hash_set_rcu_bm.h
BM_SOURCES = concurrent_hash_set_rcu_bm.C \
             concurrent_hash_set_rcu_bm_vs_baseline.C \
             concurrent_hash_set_rcu_bm_states_insert.C \
             concurrent_hash_set_rcu_bm_states_lookup.C \
             concurrent_hash_set_rcu_bm_erase.C \
             concurrent_hash_set_rcu_bm_churn.C \
             concurrent_hash_set_rcu_bm_hot_growth.C \
             concurrent_hash_set_rcu_bm_handles.C \
             concurrent_hash_set_rcu_bm_advance.C
BM_OBJECTS = $(BM_SOURCES:%.C=$(BUILD_DIR)/%.o)

$(BUILD_DIR)/%.o: %.C $(HEADERS) $(GB_HEADERS) $(BM_HEADER) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) -c $< -o $@

$(TARGET_BM): $(BM_OBJECTS)
	$(CXX) $(CXXFLAGS_BM) $(BM_OBJECTS) $(LIBS_BM) $(CXX_RPATH) -o $@

# Build ASan test
$(TARGET_TEST_ASAN): concurrent_hash_set_rcu_test.C $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_ASAN) $< $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Build TSan test
$(TARGET_TEST_TSAN): concurrent_hash_set_rcu_test.C $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_TSAN) $< $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Build the -O3, no-sanitizer store-buffering test (see CXXFLAGS_TEST_TSO above)
$(TARGET_TEST_TSO): concurrent_hash_set_rcu_tso_test.C $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_TEST_TSO) $< $(LIBS_TEST) $(CXX_RPATH) -o $@

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
