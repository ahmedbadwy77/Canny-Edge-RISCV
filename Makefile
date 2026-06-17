HOST_CXX ?= g++
RV_CXX ?= riscv64-linux-gnu-g++

OPT ?= -O3
COMMON_FLAGS = $(OPT) -std=c++17 -Wall -Wextra -DNDEBUG
HOST_FLAGS = $(COMMON_FLAGS) -march=native
RV_FLAGS = $(COMMON_FLAGS) -static -mabi=lp64d
RV_SCALAR_FLAGS = $(RV_FLAGS) -march=rv64gc
RVV_FLAGS = $(RV_FLAGS) -march=rv64gcv

SRC = src/main.cpp src/image_io.cpp
HOST_OUT = host_canny
RV_SCALAR_OUT = rv_scalar
RVV_OUT = rv_canny_linux

QEMU = qemu-riscv64
VLEN ?= 128
QEMU_RVV_CPU = rv64,v=true,vlen=$(VLEN),vext_spec=v1.0
ARGS ?= 256 256 100
VERIFY_ARGS ?= 256 256 1

.PHONY: all host scalar_rv canny_rv run_scalar run_rvv verify benchmark test clean

all: host scalar_rv canny_rv

host:
	$(HOST_CXX) $(HOST_FLAGS) $(SRC) -o $(HOST_OUT)

scalar_rv:
	$(RV_CXX) $(RV_SCALAR_FLAGS) $(SRC) -o $(RV_SCALAR_OUT)

canny_rv:
	$(RV_CXX) $(RVV_FLAGS) $(SRC) -o $(RVV_OUT)

run_scalar: scalar_rv
	$(QEMU) -cpu $(QEMU_RVV_CPU) ./$(RV_SCALAR_OUT) $(ARGS)

run_rvv: canny_rv
	$(QEMU) -cpu $(QEMU_RVV_CPU) ./$(RVV_OUT) $(ARGS)

verify: scalar_rv canny_rv
	$(QEMU) -cpu $(QEMU_RVV_CPU) ./$(RV_SCALAR_OUT) $(VERIFY_ARGS)
	cp output.raw scalar_output.raw
	$(QEMU) -cpu $(QEMU_RVV_CPU) ./$(RVV_OUT) $(VERIFY_ARGS)
	cp output.raw rvv_output.raw
	cmp scalar_output.raw rvv_output.raw
	sha256sum scalar_output.raw rvv_output.raw

benchmark: scalar_rv canny_rv
	$(QEMU) -cpu $(QEMU_RVV_CPU) ./$(RV_SCALAR_OUT) $(ARGS)
	$(QEMU) -cpu $(QEMU_RVV_CPU) ./$(RVV_OUT) $(ARGS)

test:
	$(HOST_CXX) $(COMMON_FLAGS) tests/test_pipeline.cpp src/image_io.cpp -o run_tests -lgtest -lgtest_main -pthread
	./run_tests

clean:
	rm -f $(HOST_OUT) $(RV_SCALAR_OUT) $(RVV_OUT) rv_canny run_tests
	rm -f output.raw scalar_output.raw rvv_output.raw
