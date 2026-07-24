CXX = clang++-22
VEC = native
STD = c++23

# Common configuration
COMMON_FLAGS = -march=$(VEC) --std=$(STD) -g -Werror -Wall -Wextra -pedantic
COMMON_INCLUDES = -I.
COMMON_LIBS = -lpthread -lrt -lm

# Benchmark flags (optimized). NDEBUG keeps the queue's debug-only asserts
# out of the measured hot path; the tests (-O0, no NDEBUG) still run them.
GBENCH_DIR ?= $(HOME)/GoogleBench
CXXFLAGS_BM = $(COMMON_FLAGS) -O3 -DNDEBUG $(COMMON_INCLUDES) -I$(GBENCH_DIR)/include
LIBS_BM = $(GBENCH_DIR)/lib/libbenchmark.a $(COMMON_LIBS)

# Test flags
GTEST_DIR ?= $(HOME)/GoogleTest
CXXFLAGS_TEST_ASAN = $(COMMON_FLAGS) -O0 -Wno-misleading-indentation -fsanitize=address $(COMMON_INCLUDES) -I$(GTEST_DIR)/include
CXXFLAGS_TEST_TSAN = $(COMMON_FLAGS) -O0 -Wno-misleading-indentation -fsanitize=thread $(COMMON_INCLUDES) -I$(GTEST_DIR)/include
LIBS_TEST = $(GTEST_DIR)/lib/libgtest.a $(GTEST_DIR)/lib/libgtest_main.a $(COMMON_LIBS)

# Target definitions
TARGET_GMBM = concurrent_queue_gmbm
TARGET_LMBM = concurrent_queue_lmbm
TARGET_MBM = concurrent_queue_mbm
TARGET_PPMBM = concurrent_queue_ppmbm
TARGET_TEST_ASAN = concurrent_queue_test_asan
TARGET_TEST_TSAN = concurrent_queue_test_tsan

# Headers every target depends on; editing them must trigger a rebuild.
# Recipes use $< (the source file), not $^, so the headers are prerequisites
# only and are not passed to the compiler command line.
HEADERS = concurrent_queue.h spinlock.h

.PHONY: all clean run_tests run_benchmarks

all: $(TARGET_GMBM) $(TARGET_LMBM) $(TARGET_MBM) $(TARGET_PPMBM) $(TARGET_TEST_ASAN) $(TARGET_TEST_TSAN)

# Build benchmarks
$(TARGET_GMBM): concurrent_queue_gmbm.C $(HEADERS)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) -o $@

$(TARGET_LMBM): concurrent_queue_lmbm.C $(HEADERS)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) -o $@

$(TARGET_MBM): concurrent_queue_mbm.C $(HEADERS)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) -o $@

$(TARGET_PPMBM): concurrent_queue_ppmbm.C $(HEADERS)
	$(CXX) $(CXXFLAGS_BM) $< $(LIBS_BM) -o $@

# Build ASan test
$(TARGET_TEST_ASAN): concurrent_queue_test.C $(HEADERS)
	$(CXX) $(CXXFLAGS_TEST_ASAN) $< $(LIBS_TEST) -o $@

# Build TSan test
$(TARGET_TEST_TSAN): concurrent_queue_test.C $(HEADERS)
	$(CXX) $(CXXFLAGS_TEST_TSAN) $< $(LIBS_TEST) -o $@

# Run tests
run_tests: $(TARGET_TEST_ASAN) $(TARGET_TEST_TSAN)
	./$(TARGET_TEST_ASAN)
	./$(TARGET_TEST_TSAN)

# Run benchmarks
run_benchmarks: $(TARGET_GMBM) $(TARGET_LMBM) $(TARGET_MBM) $(TARGET_PPMBM)
	./$(TARGET_GMBM)
	./$(TARGET_LMBM)
	./$(TARGET_MBM)
	./$(TARGET_PPMBM)

# Clean build artifacts
clean:
	rm -f $(TARGET_GMBM) $(TARGET_LMBM) $(TARGET_MBM) $(TARGET_PPMBM) $(TARGET_TEST_ASAN) $(TARGET_TEST_TSAN) f *.bak *~
