CXX = clang++-22
VEC = native
STD = c++23

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

# Headers every target depends on; editing them must trigger a rebuild.
# Recipes use $< (the source file), not $^, so the headers are prerequisites
# only and are not passed to the compiler command line.
# Note: lock_free_shared_ptr/atomic_shared_ptr.hpp requires Daniel Anderson's
# parlay/ headers, which are not distributed with this repository (see
# lock_free_shared_ptr/README.md for where to get them).
HEADERS = atomic_shared_ptr_concept.h intr_shared_ptr.h intr_shared_ptr/intr_shared_ptr.h lock_free_shared_ptr/atomic_shared_ptr.hpp

.PHONY: all clean run_tests run_benchmarks

all: $(TARGET_BM) $(TARGET_TEST_ASAN) $(TARGET_TEST_TSAN) $(TARGET_INTR_MBM) $(TARGET_INTR_TSAN)

# Build benchmark (all three pointer policies)
$(TARGET_BM): atomic_shared_ptr_bm.C $(HEADERS)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) -o $@

# Build ASan test
$(TARGET_TEST_ASAN): atomic_shared_ptr_test.C $(HEADERS)
	$(CXX) $(CXXFLAGS_TEST_ASAN) $< $(LIBS_TEST) -o $@

# Build TSan test
$(TARGET_TEST_TSAN): atomic_shared_ptr_test.C $(HEADERS)
	$(CXX) $(CXXFLAGS_TEST_TSAN) $< $(LIBS_TEST) -o $@

# Build intrusive pointer microbenchmark
$(TARGET_INTR_MBM): intr_shared_ptr/intr_shared_ptr_mbm.C $(HEADERS)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) -o $@

# Build intrusive pointer TSan stress test (standalone, no GoogleTest)
$(TARGET_INTR_TSAN): intr_shared_ptr/intr_shared_ptr_tsan.C $(HEADERS)
	$(CXX) $(COMMON_FLAGS) -O1 -fsanitize=thread $(COMMON_INCLUDES) $< $(COMMON_LIBS) -o $@

# Run tests
run_tests: $(TARGET_TEST_ASAN) $(TARGET_TEST_TSAN) $(TARGET_INTR_TSAN)
	./$(TARGET_TEST_ASAN)
	./$(TARGET_TEST_TSAN)
	./$(TARGET_INTR_TSAN)

# Run benchmarks
run_benchmarks: $(TARGET_BM) $(TARGET_INTR_MBM)
	./$(TARGET_BM)
	./$(TARGET_INTR_MBM)

# Clean build artifacts
clean:
	rm -f $(TARGET_BM) $(TARGET_TEST_ASAN) $(TARGET_TEST_TSAN) $(TARGET_INTR_MBM) $(TARGET_INTR_TSAN) *.bak *~
