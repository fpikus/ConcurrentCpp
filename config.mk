# Per-machine build configuration, shared by every benchmark directory in this
# repository. Write this file once per machine; the Makefiles that include it
# are machine-independent and travel with the sources unchanged.
#
#   CXX        C++ compiler to use (clang++-NN or g++-NN)
#   VEC        -march target: native unless cross-compiling for another machine
#   STD        C++ language standard
#   GBENCH_DIR Google Benchmark installation (include/ and lib/ under it)
#   GTEST_DIR  GoogleTest installation (include/ and lib/ under it)
#
# CXX is assigned with plain `=`: make predefines CXX (to g++), so `?=` would
# never take effect. The directories keep `?=` so an environment variable can
# still point a single run elsewhere; a command-line `make CXX=... VAR=...`
# overrides any of these regardless.
CXX = clang++-22
VEC = native
STD = c++23
GBENCH_DIR ?= $(HOME)/GoogleBench
GTEST_DIR ?= $(HOME)/GoogleTest
