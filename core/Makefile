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
SRC     := src/preproc.cpp
OBJ     := $(BUILD)/preproc.o
TOOL    := $(BUILD)/de_preprocess
TESTBIN := $(BUILD)/test_preproc

.PHONY: all clean test
all: $(TOOL)

$(BUILD):
	@mkdir -p $(BUILD)

$(OBJ): $(SRC) include/doubleeye/preproc.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $(INC) -c $< -o $@

$(TOOL): tools/de_preprocess.cpp $(OBJ)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $(INC) $^ -o $@

$(TESTBIN): tests/test_preproc.cpp $(OBJ)
	$(CXX) $(CXXFLAGS) $(ARCHFLAGS) $(INC) $^ -o $@

test: $(TESTBIN)
	@$(TESTBIN)

clean:
	rm -rf $(BUILD)
