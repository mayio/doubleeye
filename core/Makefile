# Plain make, no cmake required.
#
# The desktop has g++ but no cmake, and installing one needs root there. Since
# this component depends on nothing but the standard library, a Makefile is
# genuinely sufficient and keeps the desktop dependency list at zero.
#
# The same Makefile is used on the TX2 (make 4.1, g++ 7.5), so there is no
# second build system to keep in sync.
#
# C++14 rather than 17: g++ 7.5 on the Jetson only ships std::filesystem behind
# <experimental/filesystem>, and nothing here needs it.

CXX      ?= g++
# -pthread is required, not optional. std::thread on the Jetson's glibc 2.27
# needs it explicitly; the desktop's glibc >= 2.34 merged pthread into libc, so
# omitting it links fine there and fails only on the target. A difference that
# only shows up on the machine that matters is worth a comment.
CXXFLAGS ?= -O3 -std=c++14 -Wall -Wextra -pthread
# On aarch64 NEON is baseline in armv8-a, so no flag is needed for the Jetson.
# ARCHFLAGS is left empty by default because -march=native would make the
# desktop binary non-portable, and this code is meant to move between machines.
ARCHFLAGS ?=

INC     := -Iinclude
BUILD   := build
OBJS    := $(BUILD)/preproc.o $(BUILD)/match.o
TOOL    := $(BUILD)/de_preprocess
MATCHER := $(BUILD)/de_match
BENCH   := $(BUILD)/de_bench
PROF    := $(BUILD)/de_profile
PIPE    := $(BUILD)/de_pipe
DENSE   := $(BUILD)/de_dense
TESTBIN := $(BUILD)/test_preproc
TESTMAT := $(BUILD)/test_match
# Header-only kernel, so no object to link; reports SKIPPED off AArch64.
TESTSIMD := $(BUILD)/test_simd_score
TESTTOP2 := $(BUILD)/test_top2_merge
# CUDA tool: built only where nvcc exists (the TX2). sm_62 is the TX2's Pascal;
# the desktop has no toolkit, so its build is untouched. If a desktop toolkit ever
# appears, sm_61 is the GTX 1050 -- adjust or use -arch there.
NVCC := $(wildcard /usr/local/cuda/bin/nvcc)
DENSECU := $(BUILD)/de_dense_cuda

.PHONY: all clean test
all: $(TOOL) $(MATCHER) $(BENCH) $(PROF) $(PIPE) $(DENSE) $(if $(NVCC),$(DENSECU))

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/%.o: src/%.cpp include/doubleeye/preproc.hpp include/doubleeye/match.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $(INC) -c $< -o $@

$(TOOL): tools/de_preprocess.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $(INC) $^ -o $@

$(MATCHER): tools/de_match.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $(INC) $^ -o $@

$(BENCH): tools/de_bench.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $(INC) $^ -o $@

$(PROF): tools/de_profile.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $(INC) $^ -o $@

$(PIPE): tools/de_pipe.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $(INC) $^ -o $@

# EVERY header this tool includes is a prerequisite, AND the recipe uses $< rather
# than $^, because $^ would hand a .hpp to g++ as an input file. Without the
# prerequisite, editing a header relinks a stale de_dense and the change appears
# simply not to work -- obstacle 10.
#
# It happened a second time on 2026-08-10: dense_solve.hpp was added to the CUDA
# rule below when the solver moved into it, and not to this one. A header-only
# change (the sigma_s default) then produced a benchmark run that silently measured
# the OLD value, and it was caught only because the number came back exactly equal
# to a figure that should have changed. Add the header here when you add an include.
$(DENSE): tools/de_dense.cpp include/doubleeye/simd_score.hpp \
          include/doubleeye/dense_solve.hpp $(OBJS)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $(INC) $< $(OBJS) -o $@

$(TESTBIN): tests/test_preproc.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $(INC) $^ -o $@

$(TESTMAT): tests/test_match.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $(INC) $^ -o $@

$(DENSECU): tools/de_dense_cuda.cu include/doubleeye/dense_solve.hpp $(OBJS)
	$(NVCC) -O3 -std=c++14 -arch=sm_62 $(INC) $< $(OBJS) -o $@ -Xcompiler -pthread

$(TESTSIMD): tests/test_simd_score.cpp include/doubleeye/simd_score.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $(INC) $< -o $@

$(TESTTOP2): tests/test_top2_merge.cpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $(INC) $< -o $@

test: $(TESTBIN) $(TESTMAT) $(TESTSIMD) $(TESTTOP2)
	@$(TESTBIN)
	@echo
	@$(TESTMAT)
	@echo
	@$(TESTSIMD)
	@echo
	@$(TESTTOP2)

clean:
	rm -rf $(BUILD)
