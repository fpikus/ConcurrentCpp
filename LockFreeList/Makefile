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

# Benchmark flags (optimized)
CXXFLAGS_BM = $(COMMON_FLAGS) -O3 -DNDEBUG $(COMMON_INCLUDES) -I$(GBENCH_DIR)/include
LIBS_BM = $(GBENCH_DIR)/lib/libbenchmark.a $(COMMON_LIBS)

# Test flags
CXXFLAGS_TEST_ASAN = $(COMMON_FLAGS) -O0 -Wno-misleading-indentation -fsanitize=address $(COMMON_INCLUDES) -I$(GTEST_DIR)/include
CXXFLAGS_TEST_TSAN = $(COMMON_FLAGS) -O0 -Wno-misleading-indentation -fsanitize=thread $(COMMON_INCLUDES) -I$(GTEST_DIR)/include
LIBS_TEST = $(GTEST_DIR)/lib/libgtest.a $(GTEST_DIR)/lib/libgtest_main.a $(COMMON_LIBS)

# Target definitions (all under the per-host build directory)
TARGET_BM = $(BUILD_DIR)/lock_free_list_bm
TARGET_TEST_ASAN = $(BUILD_DIR)/lock_free_list_test_asan
TARGET_TEST_TSAN = $(BUILD_DIR)/lock_free_list_test_tsan

# `make benchmarks` needs only Google Benchmark and builds on machines without
# sanitizer runtimes; `make tests` needs GoogleTest and ASan/TSan. `make all` is
# both.
BENCHMARKS = $(TARGET_BM)
TESTS = $(TARGET_TEST_ASAN) $(TARGET_TEST_TSAN)

# Headers every target depends on; editing them must trigger a rebuild. Recipes
# use $< (the source file), not $^, so the headers are prerequisites only and
# are not passed to the compiler. The machine configuration is on the list for
# the same reason: a changed compiler or -march must rebuild everything, or
# stale binaries get measured.
# intr_shared_ptr.h and lock_free_shared_ptr/ are symlinks into ../SharedPtr;
# make follows them, so a change to the shipped header rebuilds here.
HEADERS = lock_free_list.h intr_shared_ptr.h lock_free_shared_ptr/atomic_shared_ptr.hpp ../config.mk

# Maged Michael's hazard pointers (vendored in ../SharedPtr/mm_hp), used by the
# intr_shared_ptr_hp policy and the drain helper. Every name below is a symlink
# into ../SharedPtr (mm_hp/ is a directory symlink); make follows them, as for
# HEADERS. mm_hp is a header plus one source file. Every target that links it
# lists HP_SOURCES twice: as a prerequisite, so editing mm_hp.cpp rebuilds the
# target, and explicitly in the recipe after $<, so mm_hp.cpp is compiled in the
# same command with the same flags (optimization, sanitizer) as the TU it links
# with; a separately built object would need one build per flavour and
# per-flavour rules. Compiling it again per target is cheap (~240 lines). Linux
# only: mm_hp issues membarrier(2).
HP_SOURCES = mm_hp/mm_hp.cpp
HP_HEADERS = intr_shared_ptr_hp.h mm_hp/mm_hp.hpp mm_hp/asymmetric_fence.hpp hp_drain.h hp_drain_gtest.h
HEADERS += $(HP_HEADERS)

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

# Build benchmark (links mm_hp for the intr_shared_ptr_hp policy)
$(TARGET_BM): lock_free_list_bm.C $(HP_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(HP_SOURCES) $(LIBS_BM) $(CXX_RPATH) -o $@

# Build ASan test (links mm_hp for the intr_shared_ptr_hp policy)
$(TARGET_TEST_ASAN): lock_free_list_test.C $(HP_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_ASAN) $< $(HP_SOURCES) $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Build TSan test (links mm_hp for the intr_shared_ptr_hp policy)
$(TARGET_TEST_TSAN): lock_free_list_test.C $(HP_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_TSAN) $< $(HP_SOURCES) $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

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
