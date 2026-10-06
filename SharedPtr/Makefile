# Machine-specific configuration -- compiler, -march target, standard, and the
# Google Benchmark / GoogleTest install paths -- comes from ../config.mk, shared
# by every benchmark directory and written once per machine. Everything below
# this include is machine-independent and ships unchanged.
include ../config.mk

# The rules that make lock_free_shared_ptr/parlay/ from its upstream clones
# (`make imports`, below).
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

# Common configuration. -I../IntrSharedPtr so the tests and the benchmark
# include the intrusive pointers, the drain helper and mm_hp from their project
# (../IntrSharedPtr) instead of copies.
COMMON_FLAGS = -march=$(VEC) --std=$(STD) -g -Werror -Wall -Wextra -pedantic
COMMON_INCLUDES = -I. -I../IntrSharedPtr
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
TARGET_BM = $(BUILD_DIR)/atomic_shared_ptr_bm
# The same benchmark with destruction-burst timing (see BM_COUNTER_FLAGS); built
# on demand (`make build/$(hostname)/atomic_shared_ptr_bursts_bm`), not by `all`.
TARGET_BM_BURSTS = $(BUILD_DIR)/atomic_shared_ptr_bursts_bm
TARGET_TEST_ASAN = $(BUILD_DIR)/atomic_shared_ptr_test_asan
TARGET_TEST_TSAN = $(BUILD_DIR)/atomic_shared_ptr_test_tsan

# `make benchmarks` needs only Google Benchmark and builds on machines without
# sanitizer runtimes; `make tests` needs GoogleTest and ASan/TSan. `make all` is
# both.
BENCHMARKS = $(TARGET_BM)
TESTS = $(TARGET_TEST_ASAN) $(TARGET_TEST_TSAN)

# Headers every target depends on; editing them must trigger a rebuild. Recipes
# use $< (the source file), not $^, so the headers are prerequisites only and
# are not passed to the compiler. The machine configuration and the intrusive
# pointer from ../IntrSharedPtr are on the list for the same reason: a change to
# either must rebuild what uses it.
# lock_free_shared_ptr/atomic_shared_ptr.hpp needs lock_free_shared_ptr/parlay/,
# which is not in the repository: `make imports` (below) makes it. PARLAY_STAMP
# (its SOURCE.txt) is on the list so that remaking parlay/ rebuilds everything
# that uses it: the generated files keep upstream's old dates.
PARLAY_STAMP = lock_free_shared_ptr/parlay/SOURCE.txt
HEADERS = atomic_shared_ptr_concept.h ../IntrSharedPtr/intr_shared_ptr.h ../IntrSharedPtr/intr_pointee.h ../IntrSharedPtr/intr_shared_ptr_common.h lock_free_shared_ptr/atomic_shared_ptr.hpp $(PARLAY_STAMP) ../config.mk

# Maged Michael's hazard pointers (../IntrSharedPtr/mm_hp/, made by `make
# imports` there), used by intr_shared_ptr_hp and the drain helper: HP_SOURCES,
# MM_HP_STAMP and HP_HEADERS come from ../IntrSharedPtr/mm_hp.mk, written once
# for every Makefile that links mm_hp (it says why each target lists HP_SOURCES
# twice, and makes the generated files come with the stamp).
include ../IntrSharedPtr/mm_hp.mk
HEADERS += $(HP_HEADERS)

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

# Make lock_free_shared_ptr/parlay/ from the ParlayLib and AtomicSharedPtr
# clones and parlay.patch (../third_party.mk says how, when it is remade, and
# what happens on a host without the clones). Every target depends on its stamp,
# so a fresh tree makes it at the first build; LockFreeList gets it by
# `make -C ../SharedPtr imports`.
imports: $(PARLAY_STAMP)

$(PARLAY_STAMP): lock_free_shared_ptr/parlay.patch $(THIRD_PARTY_MK)
	$(call third_party_import,ParlayLib AtomicSharedPtr)

# ../IntrSharedPtr/mm_hp/ is IntrSharedPtr's to make: its own `imports` decides
# whether it is out of date, so this rule always asks it (FORCE) and does not
# repeat its prerequisites here. The stamp's date after the sub-make decides
# what is rebuilt here: nothing, unless mm_hp/ was just remade.
$(MM_HP_STAMP): FORCE
	$(MAKE) --no-print-directory -C ../IntrSharedPtr imports

FORCE:

# Benchmark instrumentation (atomic_shared_ptr_bm.C, "Instrumentation counters").
# INTR_HP_BM_COUNTERS (event counters: CAS results, destructions, and
# intr_shared_ptr_hp's load-retry counters) is always on, because its cost is
# within the run-to-run spread; INTR_HP_BM_BURSTS (destruction-burst timing, one
# rdtsc per destroyed pointee; x86-64 only) is not free, so it has its own
# binary, used only for its burst counters.
BM_COUNTER_FLAGS = -DINTR_HP_BM_COUNTERS
BM_BURST_FLAGS = $(BM_COUNTER_FLAGS) -DINTR_HP_BM_BURSTS

# Build benchmark (every pointer policy; links mm_hp for intr_shared_ptr_hp)
$(TARGET_BM): atomic_shared_ptr_bm.C $(HP_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $(BM_COUNTER_FLAGS) $< $(HP_SOURCES) $(LIBS_BM) $(CXX_RPATH) -o $@

# Build the benchmark with destruction-burst timing (`make build/$(hostname)/atomic_shared_ptr_bursts_bm`)
$(TARGET_BM_BURSTS): atomic_shared_ptr_bm.C $(HP_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_BM) $(BM_BURST_FLAGS) $< $(HP_SOURCES) $(LIBS_BM) $(CXX_RPATH) -o $@

# Build ASan test (links mm_hp for intr_shared_ptr_hp)
$(TARGET_TEST_ASAN): atomic_shared_ptr_test.C $(HP_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_ASAN) $< $(HP_SOURCES) $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Build TSan test (links mm_hp for intr_shared_ptr_hp)
$(TARGET_TEST_TSAN): atomic_shared_ptr_test.C $(HP_SOURCES) $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_TSAN) $< $(HP_SOURCES) $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

# Run tests
run_tests: tests
	./$(TARGET_TEST_ASAN)
	./$(TARGET_TEST_TSAN)

# Run benchmarks
run_benchmarks: benchmarks
	./$(TARGET_BM)

# Clean build artifacts: this host's build directory only. The generated
# third-party directories stay (lock_free_shared_ptr/parlay/ here, and
# ../IntrSharedPtr/mm_hp/): they are not build outputs of this host but the
# shared inputs of every host building in this tree, and a host without the
# clones (a fleet host) could not make them again. `make -B imports` remakes
# parlay/ from the clones.
clean:
	rm -rf $(BUILD_DIR) *.bak *~
