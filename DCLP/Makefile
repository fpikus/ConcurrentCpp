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
# as the Spinlock directory: per-host subdirectories under build/, and clean
# removes only this host's subdirectory.
BUILD_DIR = build/$(shell hostname)

# Common configuration. -I../Spinlock so atomic_max_bm.C can include the shipped
# SpinLock (spinlock.h) instead of a copy, and the wall-clock accounting header
# (gb_wall_clock.h).
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
TARGET_COUNT = $(BUILD_DIR)/atomic_max_count
TARGET_TEST_ASAN = $(BUILD_DIR)/atomic_max_test_asan
TARGET_TEST_TSAN = $(BUILD_DIR)/atomic_max_test_tsan

# The two groups: `make benchmarks` needs only Google Benchmark and works on
# machines without sanitizer runtimes; `make tests` needs GoogleTest and
# ASan/TSan. `make all` is both.
BENCHMARKS = $(TARGET_BM) $(TARGET_COUNT)
TESTS = $(TARGET_TEST_ASAN) $(TARGET_TEST_TSAN)

# Headers every target depends on; editing them must trigger a rebuild. Recipes
# use $< (the source file), not $^, so the headers are prerequisites only and
# are not passed to the compiler command line. The machine configuration and
# the shipped SpinLock are on the list for the same reason: a change to either
# must rebuild what uses it.
HEADERS = atomic_max.h ../Spinlock/spinlock.h ../config.mk
# The wall-clock accounting header of the Google Benchmark binary, reached
# through the -I../Spinlock above. A variable of its own rather than an entry
# in HEADERS: HEADERS is a prerequisite of the tests, which do not use Google
# Benchmark and must not depend on this header.
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

# Build benchmark (lock-free vs spinlocked vs DCLP running maximum)
$(TARGET_BM): atomic_max_bm.C $(HEADERS) $(GB_HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) $(CXX_RPATH) -o $@

# Build the outcome counter (per-offer outcomes of CAS vs DCLP on the grow
# feed; not a timing run, so no Google Benchmark library)
$(TARGET_COUNT): atomic_max_count.C $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(COMMON_LIBS) $(CXX_RPATH) -o $@

# Build ASan test (atomic_max correctness + ordering)
$(TARGET_TEST_ASAN): atomic_max_test.C $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_ASAN) $< $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Build TSan test (validates the acquire/acq_rel handoff)
$(TARGET_TEST_TSAN): atomic_max_test.C $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_TSAN) $< $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Run tests
run_tests: tests
	./$(TARGET_TEST_ASAN)
	./$(TARGET_TEST_TSAN)

# Run benchmarks
run_benchmarks: benchmarks
	./$(TARGET_BM)
	./$(TARGET_COUNT)

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) *.bak *~
