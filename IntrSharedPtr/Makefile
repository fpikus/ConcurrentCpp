# Machine-specific configuration -- compiler, -march target, standard, and the
# Google Benchmark / GoogleTest install paths -- comes from ../config.mk, shared
# by every benchmark directory and written once per machine. Everything below
# this include is machine-independent and ships unchanged.
include ../config.mk

# The rules that make mm_hp/ from its upstream clone (`make imports`, below).
include ../third_party.mk

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
# the other project directories: per-host subdirectories under build/, and clean
# removes only this host's subdirectory.
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
# The microbenchmark of the two pointers: one program, typed over both, the
# same four operations for each.
TARGET_INTR_MBM = $(BUILD_DIR)/intr_shared_ptr_mbm
# The standalone TSan stress program of the two pointers (no GoogleTest): one
# program, typed over both.
TARGET_INTR_TSAN = $(BUILD_DIR)/intr_shared_ptr_tsan
# Self-test of the mm_hp drain helper (hp_drain.h), one build per sanitizer.
TARGET_HP_DRAIN_SELFTEST_ASAN = $(BUILD_DIR)/hp_drain_selftest_asan
TARGET_HP_DRAIN_SELFTEST_TSAN = $(BUILD_DIR)/hp_drain_selftest_tsan
# White-box seam tests of intr_shared_ptr_hp (deterministic, through the
# pointee's hooks), one build per sanitizer.
TARGET_HP_SEAM_TEST_ASAN = $(BUILD_DIR)/intr_shared_ptr_hp_seam_test_asan
TARGET_HP_SEAM_TEST_TSAN = $(BUILD_DIR)/intr_shared_ptr_hp_seam_test_tsan
# Black-box contract tests of the pointee side (the hooks, the two concepts, the
# count bases), one build per sanitizer.
TARGET_POINTEE_TEST_ASAN = $(BUILD_DIR)/intr_pointee_test_asan
TARGET_POINTEE_TEST_TSAN = $(BUILD_DIR)/intr_pointee_test_tsan
# A user's own structure on both pointers (a Treiber stack written from
# README.md, "Building your own data structure"), one build per sanitizer.
TARGET_EXAMPLE_TEST_ASAN = $(BUILD_DIR)/intr_shared_ptr_example_test_asan
TARGET_EXAMPLE_TEST_TSAN = $(BUILD_DIR)/intr_shared_ptr_example_test_tsan

# `make benchmarks` needs only Google Benchmark and builds on machines without
# sanitizer runtimes; `make tests` needs GoogleTest and ASan/TSan (the
# standalone TSan program needs only the latter). `make all` is both. A new
# test binary takes four edits: its TARGET_ variable above, this list, its
# build rule below, and its line in run_tests.
BENCHMARKS = $(TARGET_INTR_MBM)
TESTS = $(TARGET_INTR_TSAN) \
        $(TARGET_HP_DRAIN_SELFTEST_ASAN) $(TARGET_HP_DRAIN_SELFTEST_TSAN) \
        $(TARGET_HP_SEAM_TEST_ASAN) $(TARGET_HP_SEAM_TEST_TSAN) \
        $(TARGET_POINTEE_TEST_ASAN) $(TARGET_POINTEE_TEST_TSAN) \
        $(TARGET_EXAMPLE_TEST_ASAN) $(TARGET_EXAMPLE_TEST_TSAN)

# Headers every target depends on; editing them must trigger a rebuild. Recipes
# use $< (the source file), not $^, so the headers are prerequisites only and
# are not passed to the compiler. The machine configuration is on the list for
# the same reason: a changed compiler or -march must rebuild everything, or
# stale binaries get measured.
HEADERS = intr_shared_ptr.h intr_pointee.h intr_shared_ptr_common.h ../config.mk

# Maged Michael's hazard pointers (mm_hp/, made by `make imports` below), used by
# intr_shared_ptr_hp and the drain helper: HP_SOURCES, MM_HP_STAMP and
# HP_HEADERS come from mm_hp.mk, shared with the consumers' Makefiles (it says
# why each target lists HP_SOURCES twice). mm_hp/ sits beside
# intr_shared_ptr_hp.h, so the header's quoted #include "mm_hp/mm_hp.hpp"
# resolves next to the header itself, for the TUs here and for a consumer that
# reaches the header through -I../IntrSharedPtr alike.
include mm_hp.mk
HEADERS += $(HP_HEADERS)

.PHONY: all benchmarks tests clean run_tests run_benchmarks imports

# The first rule in a Makefile is the default goal; without this line it would
# be the $(BUILD_DIR) rule below, and a bare `make` would only create the
# directory.
.DEFAULT_GOAL := all

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

all: benchmarks tests

benchmarks: $(BENCHMARKS)

tests: $(TESTS)

# Make mm_hp/ from the HazardPtr clone and mm_hp.patch (../third_party.mk says
# how, when it is remade, and what happens on a host without the clone). Every
# target that uses mm_hp depends on its stamp, so a fresh tree makes it at the
# first build; SharedPtr and LockFreeList get it by `make -C ../IntrSharedPtr
# imports`.
imports: $(MM_HP_STAMP)

$(MM_HP_STAMP): mm_hp.patch $(THIRD_PARTY_MK)
	$(call third_party_import,HazardPtr)

# Build the microbenchmark of both pointers (Google Benchmark; links mm_hp for
# intr_shared_ptr_hp)
$(TARGET_INTR_MBM): intr_shared_ptr_mbm.C $(HP_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $< $(HP_SOURCES) $(LIBS_BM) $(CXX_RPATH) -o $@

# Build the TSan stress test of both pointers (standalone, no GoogleTest; -O1
# and no -DNDEBUG: its checks are assert()s; links mm_hp for intr_shared_ptr_hp)
$(TARGET_INTR_TSAN): intr_shared_ptr_tsan.C $(HP_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(COMMON_FLAGS) -O1 -fsanitize=thread $(COMMON_INCLUDES) $< $(HP_SOURCES) $(COMMON_LIBS) $(SAN_CXX_RPATH) -o $@

# Build the drain helper's self-test, ASan (includes the LeakSanitizer positive control)
$(TARGET_HP_DRAIN_SELFTEST_ASAN): hp_drain_selftest.C $(HP_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_ASAN) $< $(HP_SOURCES) $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Build the drain helper's self-test, TSan
$(TARGET_HP_DRAIN_SELFTEST_TSAN): hp_drain_selftest.C $(HP_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_TSAN) $< $(HP_SOURCES) $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Build the hazard-pointer intrusive pointer's seam tests, ASan
$(TARGET_HP_SEAM_TEST_ASAN): intr_shared_ptr_hp_seam_test.C $(HP_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_ASAN) $< $(HP_SOURCES) $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Build the hazard-pointer intrusive pointer's seam tests, TSan
$(TARGET_HP_SEAM_TEST_TSAN): intr_shared_ptr_hp_seam_test.C $(HP_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_TSAN) $< $(HP_SOURCES) $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Build the pointee contract tests, ASan
$(TARGET_POINTEE_TEST_ASAN): intr_pointee_test.C $(HP_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_ASAN) $< $(HP_SOURCES) $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Build the pointee contract tests, TSan
$(TARGET_POINTEE_TEST_TSAN): intr_pointee_test.C $(HP_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_TSAN) $< $(HP_SOURCES) $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Build the example structure's tests, ASan
$(TARGET_EXAMPLE_TEST_ASAN): intr_shared_ptr_example_test.C $(HP_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_ASAN) $< $(HP_SOURCES) $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Build the example structure's tests, TSan
$(TARGET_EXAMPLE_TEST_TSAN): intr_shared_ptr_example_test.C $(HP_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_TSAN) $< $(HP_SOURCES) $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Run tests
run_tests: tests
	./$(TARGET_INTR_TSAN)
	./$(TARGET_HP_DRAIN_SELFTEST_ASAN)
	./$(TARGET_HP_DRAIN_SELFTEST_TSAN)
	./$(TARGET_HP_SEAM_TEST_ASAN)
	./$(TARGET_HP_SEAM_TEST_TSAN)
	./$(TARGET_POINTEE_TEST_ASAN)
	./$(TARGET_POINTEE_TEST_TSAN)
	./$(TARGET_EXAMPLE_TEST_ASAN)
	./$(TARGET_EXAMPLE_TEST_TSAN)

# Run benchmarks
run_benchmarks: benchmarks
	./$(TARGET_INTR_MBM)

# Clean build artifacts: this host's build directory only. mm_hp/ stays: it is
# not a build output of this host but the shared input of every host building
# in this tree (and of SharedPtr and LockFreeList), and a host without the
# HazardPtr clone (a fleet host) could not make it again. `make -B imports`
# remakes it from the clone.
clean:
	rm -rf $(BUILD_DIR) *.bak *~
