HOST_CXX = g++
RV_CXX   = riscv64-linux-gnu-g++

SRC      = src/main.cpp src/image_io.cpp
TEST_SRC = tests/test_pipeline.cpp src/image_io.cpp

HOST_OUT = host_canny
RV_OUT   = rv_canny
TEST_OUT = run_tests

CXXFLAGS   = -std=c++17 -Wall -Wextra -fopt-info-vec-all
ARCH_FLAGS = -march=rv64gcv -mabi=lp64d -static
TEST_FLAGS = -std=c++17 -Wall -Iinclude
TEST_LIBS  = -lgtest -lgtest_main -lpthread

all: host canny_rv

host:
	$(HOST_CXX) $(CXXFLAGS) -Iinclude $(SRC) -o $(HOST_OUT)

canny_rv:
	$(RV_CXX) $(CXXFLAGS) $(ARCH_FLAGS) -Iinclude $(SRC) -o $(RV_OUT)

test:
	$(HOST_CXX) $(TEST_FLAGS) $(TEST_SRC) -o $(TEST_OUT) $(TEST_LIBS)
	./$(TEST_OUT)

run: canny_rv
	qemu-riscv64 $(RV_OUT)

clean:
	rm -f $(HOST_OUT) $(RV_OUT) $(TEST_OUT) images/test_image.raw

OPT_FLAGS = -O0 -O2 -O3 -Os -Ofast

sweep: $(OPT_FLAGS)

$(OPT_FLAGS):
	$(RV_CXX) $(CXXFLAGS) $@ $(ARCH_FLAGS) -Iinclude $(SRC) -o rv_canny$@
	ls -lh rv_canny$@
	qemu-riscv64 -cpu rv64,v=true,vlen=128 ./rv_canny$@
