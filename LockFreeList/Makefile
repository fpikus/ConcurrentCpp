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

# Common configuration. -I../IntrSharedPtr -I../SharedPtr so the list's tests
# and benchmark include the pointer policies from their projects instead of
# copies: the intrusive pointers, the drain helper and mm_hp from
# ../IntrSharedPtr, the concept header (with the std adapter) from ../SharedPtr.
COMMON_FLAGS = -march=$(VEC) --std=$(STD) -g -Werror -Wall -Wextra -pedantic
COMMON_INCLUDES = -I. -I../IntrSharedPtr -I../SharedPtr
# `?=` so a machine's config.mk can override: macOS has no librt.
COMMON_LIBS ?= -lpthread -lrt -lm

# Benchmark flags (optimized). -I../Spinlock reaches gb_wall_clock.h, the
# wall-clock accounting of the threaded Google Benchmark rows; on the benchmark
# flags only, since the tests do not use Google Benchmark.
CXXFLAGS_BM = $(COMMON_FLAGS) -O3 -DNDEBUG $(COMMON_INCLUDES) -I../Spinlock -I$(GBENCH_DIR)/include
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
# stale binaries get measured. The pointer headers from ../IntrSharedPtr and
# ../SharedPtr are on the list for the same reason, by their real paths.
HEADERS = lock_free_list.h ../SharedPtr/atomic_shared_ptr_concept.h ../IntrSharedPtr/intr_shared_ptr.h \
          ../IntrSharedPtr/intr_pointee.h ../IntrSharedPtr/intr_shared_ptr_common.h ../config.mk

# Maged Michael's hazard pointers (../IntrSharedPtr/mm_hp/, made by `make
# imports` there), used by the intr_shared_ptr_hp policy and the drain helper:
# HP_SOURCES, MM_HP_STAMP and HP_HEADERS come from ../IntrSharedPtr/mm_hp.mk,
# written once for every Makefile that links mm_hp (it says why each target
# lists HP_SOURCES twice, and makes the generated files come with the stamp).
include ../IntrSharedPtr/mm_hp.mk
HEADERS += $(HP_HEADERS)

# The wall-clock accounting header, a prerequisite of the benchmark alone. A
# variable of its own rather than an entry in HEADERS: HEADERS is a
# prerequisite of the tests, which do not use Google Benchmark and must not
# depend on this header.
GB_HEADERS = ../Spinlock/gb_wall_clock.h

.PHONY: all benchmarks tests clean run_tests run_benchmarks imports FORCE

# The first rule in a Makefile is the default goal; without this line it would
# be the $(BUILD_DIR) rule below, and a bare `make` would only create the
# directory.
.DEFAULT_GOAL := all

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

all: benchmarks tests

benchmarks: $(BENCHMARKS)

tests: $(TESTS)

# The generated third-party directory ../IntrSharedPtr/mm_hp/ is IntrSharedPtr's
# to make (../third_party.mk), by its `make imports`. The owner decides whether
# its directory is out of date, so this rule always asks it (FORCE) and does not
# repeat its prerequisites here. The stamp's date after the sub-make decides
# what is rebuilt here: nothing, unless the directory was just remade.
$(MM_HP_STAMP): FORCE
	$(MAKE) --no-print-directory -C ../IntrSharedPtr imports

# The generated directory, made by asking its owner (the rule above): `make
# imports` here makes nothing of its own.
imports: $(MM_HP_STAMP)

FORCE:

# Build benchmark (links mm_hp for the intr_shared_ptr_hp policy)
$(TARGET_BM): lock_free_list_bm.C $(HP_SOURCES) $(HEADERS) $(GB_HEADERS) | $(BUILD_DIR)
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

# Clean build artifacts: this host's build directory only. The generated
# third-party directory of ../IntrSharedPtr is its project's input, not this
# host's build output, and is never removed here.
clean:
	rm -rf $(BUILD_DIR) *.bak *~
