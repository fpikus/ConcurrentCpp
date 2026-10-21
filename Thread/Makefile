# Machine-specific configuration -- compiler, -march target, standard, and the
# GoogleTest install path -- comes from ../config.mk, shared by every project
# directory and written once per machine. Everything below this include is
# machine-independent and ships unchanged.
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
# NFS-shared tree, so binaries must not overwrite each other: per-host
# subdirectories under build/, and clean removes only this host's subdirectory.
# Two builds on one host (two compilers, or two sets of targets built at the
# same time) each take a directory of their own from the command line, as in
# `make BUILD_DIR=build/<name> tests`; clean with the same setting removes only
# that directory. The directory is relative to this one: run_tests starts the
# binaries as ./$(BUILD_DIR)/...
BUILD_DIR = build/$(shell hostname)

# Common configuration
COMMON_FLAGS = -march=$(VEC) --std=$(STD) -g -Werror -Wall -Wextra -pedantic
COMMON_INCLUDES = -I.
# `?=` so a machine's config.mk can override the list.
COMMON_LIBS ?= -lpthread -lrt -lm

# Test flags. Every test source is built three ways: under ASan and under TSan
# (at -O0), and optimized without a sanitizer. The optimized build is the only
# one that runs the header's code as it ships: the sanitizer runtimes intercept
# pthread_create and change what the new thread starts with (ASan sets the
# child's signal mask back to the creator's, TSan enlarges small stack
# requests), and -O0 keeps code that the optimizer may remove or inline (a
# stack reservation that nothing reads, a call to a small function), so only an
# optimized build shows that the header keeps them. -O3 -DNDEBUG and the same
# GoogleTest link as the sanitizer builds. -Wno-misleading-indentation is in
# all three, so that a source that builds in one flavor builds in the others.
CXXFLAGS_TEST_ASAN = $(COMMON_FLAGS) -O0 -Wno-misleading-indentation -fsanitize=address $(COMMON_INCLUDES) -I$(GTEST_DIR)/include
CXXFLAGS_TEST_TSAN = $(COMMON_FLAGS) -O0 -Wno-misleading-indentation -fsanitize=thread $(COMMON_INCLUDES) -I$(GTEST_DIR)/include
CXXFLAGS_TEST_OPT = $(COMMON_FLAGS) -O3 -DNDEBUG -Wno-misleading-indentation $(COMMON_INCLUDES) -I$(GTEST_DIR)/include
LIBS_TEST = $(GTEST_DIR)/lib/libgtest.a $(GTEST_DIR)/lib/libgtest_main.a $(COMMON_LIBS)

# Target definitions (all under the per-host build directory).
# thread_test.C tests the documented contract of thread.h;
# thread_internals_test.C tests its implementation details.
TARGET_TEST_ASAN = $(BUILD_DIR)/thread_test_asan
TARGET_TEST_TSAN = $(BUILD_DIR)/thread_test_tsan
TARGET_TEST_OPT = $(BUILD_DIR)/thread_test_opt
TARGET_INTERNALS_TEST_ASAN = $(BUILD_DIR)/thread_internals_test_asan
TARGET_INTERNALS_TEST_TSAN = $(BUILD_DIR)/thread_internals_test_tsan
TARGET_INTERNALS_TEST_OPT = $(BUILD_DIR)/thread_internals_test_opt

# The two groups: `make benchmarks` and `make tests`; `make all` is both. This
# directory has no benchmark yet, so BENCHMARKS is empty and the benchmarks
# and run_benchmarks targets do nothing; they exist so that every project
# directory answers the same make targets.
BENCHMARKS =
TESTS = $(TARGET_TEST_ASAN) $(TARGET_TEST_TSAN) $(TARGET_TEST_OPT) \
        $(TARGET_INTERNALS_TEST_ASAN) $(TARGET_INTERNALS_TEST_TSAN) \
        $(TARGET_INTERNALS_TEST_OPT)

# Headers every target depends on; editing them must trigger a rebuild. Recipes
# use $< (the source file), not $^, so the headers are prerequisites only and
# are not passed to the compiler. The machine configuration is on the list for
# the same reason: a changed -march or standard must rebuild everything. A
# compiler chosen on the command line (`make CXX=...`) is not a prerequisite:
# switching compilers needs a `make clean` or a BUILD_DIR of its own.
HEADERS = thread.h ../config.mk

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

# Build the contract tests: ASan, TSan, optimized without a sanitizer
$(TARGET_TEST_ASAN): thread_test.C $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_ASAN) $< $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

$(TARGET_TEST_TSAN): thread_test.C $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_TSAN) $< $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

$(TARGET_TEST_OPT): thread_test.C $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_TEST_OPT) $< $(LIBS_TEST) $(CXX_RPATH) -o $@

# Build the implementation tests: ASan, TSan, optimized without a sanitizer
$(TARGET_INTERNALS_TEST_ASAN): thread_internals_test.C $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_ASAN) $< $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

$(TARGET_INTERNALS_TEST_TSAN): thread_internals_test.C $(HEADERS) | $(BUILD_DIR)
	$(SAN_CXX) $(CXXFLAGS_TEST_TSAN) $< $(LIBS_TEST) $(SAN_CXX_RPATH) -o $@

$(TARGET_INTERNALS_TEST_OPT): thread_internals_test.C $(HEADERS) | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS_TEST_OPT) $< $(LIBS_TEST) $(CXX_RPATH) -o $@

# Run tests. The sanitizer options (ASAN_OPTIONS, TSAN_OPTIONS) come from the
# caller's environment; the binaries see it unchanged.
run_tests: tests
	./$(TARGET_TEST_ASAN)
	./$(TARGET_TEST_TSAN)
	./$(TARGET_TEST_OPT)
	./$(TARGET_INTERNALS_TEST_ASAN)
	./$(TARGET_INTERNALS_TEST_TSAN)
	./$(TARGET_INTERNALS_TEST_OPT)

# Run benchmarks: none yet (see BENCHMARKS above)
run_benchmarks: benchmarks

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) *.bak *~
