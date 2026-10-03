CXX = clang++-22
VEC = native
STD = c++23

# Compilers and run-time library paths. CXX (set above, or on the command line)
# may be a toolset compiler whose libstdc++ is not on the default library path;
# a binary it links then builds cleanly and fails at run time with
# "GLIBCXX_x.y.z not found" against the system libstdc++. So every link embeds
# an rpath to the directory of the libstdc++ that the linking compiler itself
# uses, derived from the compiler so it is right for any compiler on any
# machine: `-print-file-name=libstdc++.so` prints that library's full path, or
# the bare name when the compiler knows none, in which case RPATH_OF adds no
# flag. For a system compiler the directory may hold only the link-time symlink;
# the loader finds no libstdc++.so.6 there and falls through to the default path.
# The sanitizer tests are built with SAN_CXX, which may name a separate
# toolchain and defaults to CXX; a non-empty SAN_RPATH overrides the rpath
# derived for SAN_CXX. (This directory does not include ../config.mk.)
# The commas of the flag go through $(comma): a literal one would split the
# arguments of $(if).
SAN_CXX ?= $(CXX)
comma := ,
RPATH_OF = $(strip $(if $(findstring /,$(1)),-Wl$(comma)-rpath$(comma)$(dir $(1))))
CXX_RPATH := $(call RPATH_OF,$(shell $(CXX) -print-file-name=libstdc++.so))
SAN_CXX_RPATH := $(if $(SAN_RPATH),$(SAN_RPATH),$(call RPATH_OF,$(shell $(SAN_CXX) -print-file-name=libstdc++.so)))

# Common configuration
COMMON_FLAGS = -march=$(VEC) --std=$(STD) -g -Werror -Wall -Wextra -pedantic
COMMON_INCLUDES = -I.
COMMON_LIBS = -lpthread -lrt -lm

# Benchmark flags (optimized)
GBENCH_DIR ?= $(HOME)/GoogleBench
CXXFLAGS_BM = $(COMMON_FLAGS) -O3 -DNDEBUG $(COMMON_INCLUDES) -I$(GBENCH_DIR)/include
LIBS_BM = $(GBENCH_DIR)/lib/libbenchmark.a $(COMMON_LIBS)

# Test flags
GTEST_DIR ?= $(HOME)/GoogleTest
CXXFLAGS_TEST_ASAN = $(COMMON_FLAGS) -O0 -Wno-misleading-indentation -fsanitize=address $(COMMON_INCLUDES) -I$(GTEST_DIR)/include
CXXFLAGS_TEST_TSAN = $(COMMON_FLAGS) -O0 -Wno-misleading-indentation -fsanitize=thread $(COMMON_INCLUDES) -I$(GTEST_DIR)/include
LIBS_TEST = $(GTEST_DIR)/lib/libgtest.a $(GTEST_DIR)/lib/libgtest_main.a $(COMMON_LIBS)

# Target definitions
TARGET_BM = atomic_shared_ptr_bm
TARGET_TEST_ASAN = atomic_shared_ptr_test_asan
TARGET_TEST_TSAN = atomic_shared_ptr_test_tsan
TARGET_INTR_MBM = intr_shared_ptr/intr_shared_ptr_mbm
TARGET_INTR_TSAN = intr_shared_ptr/intr_shared_ptr_tsan
# The hazard-pointer intrusive pointer's microbenchmark and TSan program,
# mirroring the two above.
TARGET_INTR_HP_MBM = intr_shared_ptr_hp/intr_shared_ptr_hp_mbm
TARGET_INTR_HP_TSAN = intr_shared_ptr_hp/intr_shared_ptr_hp_tsan
# Self-test of the mm_hp drain helper (hp_drain.h), one build per sanitizer.
TARGET_HP_DRAIN_SELFTEST_ASAN = hp_drain_selftest_asan
TARGET_HP_DRAIN_SELFTEST_TSAN = hp_drain_selftest_tsan

# Headers every target depends on; editing them must trigger a rebuild.
# Recipes use $< (the source file), not $^, so the headers are prerequisites
# only and are not passed to the compiler command line.
# Note: lock_free_shared_ptr/atomic_shared_ptr.hpp requires Daniel Anderson's
# parlay/ headers, which are not distributed with this repository (see
# lock_free_shared_ptr/README.md for where to get them).
HEADERS = atomic_shared_ptr_concept.h intr_shared_ptr.h intr_shared_ptr/intr_shared_ptr.h lock_free_shared_ptr/atomic_shared_ptr.hpp

# Maged Michael's hazard pointers (mm_hp/, vendored), used by intr_shared_ptr_hp
# and the drain helper. mm_hp is a header plus one source file. Every target that
# links it lists HP_SOURCES twice: as a prerequisite, so editing mm_hp.cpp
# rebuilds the target, and explicitly in the recipe after $<, so mm_hp.cpp is
# compiled in the same command with the same flags (optimization, sanitizer) as
# the TU it links with; a separately built object would need one build per
# flavour and per-flavour rules. Compiling it again per target is cheap (~240
# lines). Linux only: mm_hp issues membarrier(2).
HP_SOURCES = mm_hp/mm_hp.cpp
HP_HEADERS = intr_shared_ptr_hp.h intr_shared_ptr_hp/intr_shared_ptr_hp.h mm_hp/mm_hp.hpp mm_hp/asymmetric_fence.hpp hp_drain.h hp_drain_gtest.h
HEADERS += $(HP_HEADERS)

.PHONY: all clean run_tests run_benchmarks

all: $(TARGET_BM) $(TARGET_TEST_ASAN) $(TARGET_TEST_TSAN) $(TARGET_INTR_MBM) $(TARGET_INTR_TSAN) \
     $(TARGET_INTR_HP_MBM) $(TARGET_INTR_HP_TSAN) $(TARGET_HP_DRAIN_SELFTEST_ASAN) $(TARGET_HP_DRAIN_SELFTEST_TSAN)

# Build benchmark (every pointer policy; links mm_hp for intr_shared_ptr_hp)
$(TARGET_BM): atomic_shared_ptr_bm.C $(HP_SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS_BM) $< $(HP_SOURCES) $(LIBS_BM) $(CXX_RPATH) -o $@

# Build ASan test (links mm_hp for intr_shared_ptr_hp)
$(TARGET_TEST_ASAN): atomic_shared_ptr_test.C $(HP_SOURCES) $(HEADERS)
	$(SAN_CXX) $(CXXFLAGS_TEST_ASAN) $< $(HP_SOURCES) $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Build TSan test (links mm_hp for intr_shared_ptr_hp)
$(TARGET_TEST_TSAN): atomic_shared_ptr_test.C $(HP_SOURCES) $(HEADERS)
	$(SAN_CXX) $(CXXFLAGS_TEST_TSAN) $< $(HP_SOURCES) $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Build intrusive pointer microbenchmark
$(TARGET_INTR_MBM): intr_shared_ptr/intr_shared_ptr_mbm.C $(HEADERS)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) $(CXX_RPATH) -o $@

# Build intrusive pointer TSan stress test (standalone, no GoogleTest)
$(TARGET_INTR_TSAN): intr_shared_ptr/intr_shared_ptr_tsan.C $(HEADERS)
	$(SAN_CXX) $(COMMON_FLAGS) -O1 -fsanitize=thread $(COMMON_INCLUDES) $< $(COMMON_LIBS) $(SAN_CXX_RPATH) -o $@

# Build hazard-pointer intrusive pointer microbenchmark (as the intrusive one, plus mm_hp)
$(TARGET_INTR_HP_MBM): intr_shared_ptr_hp/intr_shared_ptr_hp_mbm.C $(HP_SOURCES) $(HEADERS)
	$(CXX) $(CXXFLAGS_BM) $< $(HP_SOURCES) $(LIBS_BM) $(CXX_RPATH) -o $@

# Build hazard-pointer intrusive pointer TSan stress test (standalone, no GoogleTest; plus mm_hp)
$(TARGET_INTR_HP_TSAN): intr_shared_ptr_hp/intr_shared_ptr_hp_tsan.C $(HP_SOURCES) $(HEADERS)
	$(SAN_CXX) $(COMMON_FLAGS) -O1 -fsanitize=thread $(COMMON_INCLUDES) $< $(HP_SOURCES) $(COMMON_LIBS) $(SAN_CXX_RPATH) -o $@

# Build the drain helper's self-test, ASan (includes the LeakSanitizer positive control)
$(TARGET_HP_DRAIN_SELFTEST_ASAN): hp_drain_selftest.C $(HP_SOURCES) $(HEADERS)
	$(SAN_CXX) $(CXXFLAGS_TEST_ASAN) $< $(HP_SOURCES) $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Build the drain helper's self-test, TSan
$(TARGET_HP_DRAIN_SELFTEST_TSAN): hp_drain_selftest.C $(HP_SOURCES) $(HEADERS)
	$(SAN_CXX) $(CXXFLAGS_TEST_TSAN) $< $(HP_SOURCES) $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Run tests
run_tests: $(TARGET_TEST_ASAN) $(TARGET_TEST_TSAN) $(TARGET_INTR_TSAN) $(TARGET_INTR_HP_TSAN) \
           $(TARGET_HP_DRAIN_SELFTEST_ASAN) $(TARGET_HP_DRAIN_SELFTEST_TSAN)
	./$(TARGET_TEST_ASAN)
	./$(TARGET_TEST_TSAN)
	./$(TARGET_INTR_TSAN)
	./$(TARGET_INTR_HP_TSAN)
	./$(TARGET_HP_DRAIN_SELFTEST_ASAN)
	./$(TARGET_HP_DRAIN_SELFTEST_TSAN)

# Run benchmarks
run_benchmarks: $(TARGET_BM) $(TARGET_INTR_MBM) $(TARGET_INTR_HP_MBM)
	./$(TARGET_BM)
	./$(TARGET_INTR_MBM)
	./$(TARGET_INTR_HP_MBM)

# Clean build artifacts
clean:
	rm -f $(TARGET_BM) $(TARGET_TEST_ASAN) $(TARGET_TEST_TSAN) $(TARGET_INTR_MBM) $(TARGET_INTR_TSAN) \
	      $(TARGET_INTR_HP_MBM) $(TARGET_INTR_HP_TSAN) $(TARGET_HP_DRAIN_SELFTEST_ASAN) $(TARGET_HP_DRAIN_SELFTEST_TSAN) *.bak *~
