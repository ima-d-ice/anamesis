CXX ?= g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pthread -MMD -MP -Isrc -Isrc/third_party -Isrc/third_party/hnswlib
LDFLAGS := -pthread

SRC_DIR := src
OBJ_DIR := .objs

SRCS := $(shell find $(SRC_DIR) -name '*.cpp' ! -name 'test_*' ! -path '*/router/*')
OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SRCS))

DEPS := $(OBJS:.o=.d)

TARGET := anamesis

TEST_SRCS := $(shell find tests -name 'test_*.cpp')
TESTS := $(TEST_SRCS:.cpp=)

.PHONY: all test clean run

all: $(TARGET)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(OBJ_DIR):
	mkdir -p $(OBJ_DIR)
	mkdir -p $(OBJ_DIR)/eviction
	mkdir -p $(OBJ_DIR)/index

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET) $(LDFLAGS)

# Test binaries compile only their own translation units (no main.cpp link)
TEST_OBJ_DIR := .objs_test

tests/%: tests/%.cpp $(filter-out $(OBJ_DIR)/main.o,$(OBJS)) | $(TEST_OBJ_DIR)
	$(CXX) $(CXXFLAGS) -Isrc $< $(filter-out $(OBJ_DIR)/main.o,$(OBJS)) -o $@ $(LDFLAGS)

$(TEST_OBJ_DIR):
	mkdir -p $(TEST_OBJ_DIR)

test: $(TESTS)
	@for t in $(TESTS); do \
		echo "Running $$t..."; \
		./$$t || exit 1; \
	done

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(OBJ_DIR) $(TEST_OBJ_DIR) $(TARGET) $(TESTS)
	rm -f src/**/*.d src/*.d tests/*.d

-include $(DEPS)
