CXX      := hipcc
CXXFLAGS := -Wall -std=c++17 --offload-arch=gfx908 -fopenmp -O2 -march=znver2 # -Wextra -Werror -pedantic-errors
LDFLAGS  := -L/usr/lib -lstdc++ -lm -fopenmp -O2 -march=znver2
BUILD    := ./build
OBJ_DIR  := $(BUILD)/objects
APP_DIR  := $(BUILD)/apps
TARGET   := llama
INCLUDE  := -Iinclude/ -I/opt/rocm/include/hipblas -I/opt/rocm/include/rccl
SRC      :=                      \
   $(wildcard src/*.cpp)         \
	 $(wildcard src/thaDNN/*.cpp)  \

OBJECTS  := $(SRC:%.cpp=$(OBJ_DIR)/%.o)

DEPENDENCIES \
         := $(OBJECTS:.o=.d)

all: build $(APP_DIR)/$(TARGET)

$(OBJ_DIR)/%.o: %.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -MMD -o $@

$(APP_DIR)/$(TARGET): $(OBJECTS)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -o $(APP_DIR)/$(TARGET) $^ $(LDFLAGS)

-include $(DEPENDENCIES)

.PHONY: all build clean debug release info 
build:
	@mkdir -p $(APP_DIR)
	@mkdir -p $(OBJ_DIR)

debug: CXXFLAGS += -DDEBUG -g
debug: all

release: CXXFLAGS += -O2
release: all

clean:
	-@rm -rvf $(OBJ_DIR)/*
	-@rm -rvf $(APP_DIR)/*

info:
	@echo "[*] Application dir: ${APP_DIR}     "
	@echo "[*] Object dir:      ${OBJ_DIR}     "
	@echo "[*] Sources:         ${SRC}         "
	@echo "[*] Objects:         ${OBJECTS}     "
	@echo "[*] Dependencies:    ${DEPENDENCIES}"


# .PHONY: perf_comp
perf_comp:
	hipcc $(CXXFLAGS) -o $(APP_DIR)/llama_bench llama_benchmark.cpp -DAUTO_KERNEL $(LDFLAGS) 

.PHONY: clean_perf
clean_perf:
	rm -f $(APP_DIR)/llama_bench

# the most basic way of building that is most likely to work on most systems
.PHONY: runcc
runcc: run.cc
	hipcc -o $(APP_DIR)/runcc run.cc -O2 # --offload-arch=gfx908 

.PHONY: runcc_clean
runcc_clean:
	rm -f $(APP_DIR)/runcc
