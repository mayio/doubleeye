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

.PHONY: all clean test
all: $(TOOL) $(MATCHER) $(BENCH) $(PROF) $(PIPE) $(DENSE)

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

$(DENSE): tools/de_dense.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $(INC) $^ -o $@

$(TESTBIN): tests/test_preproc.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $(INC) $^ -o $@

$(TESTMAT): tests/test_match.cpp $(OBJS)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $(INC) $^ -o $@

test: $(TESTBIN) $(TESTMAT)
	@$(TESTBIN)
	@echo
	@$(TESTMAT)

clean:
	rm -rf $(BUILD)
