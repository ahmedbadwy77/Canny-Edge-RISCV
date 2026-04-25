
HOST_CXX = g++
RV_CXX = riscv64-unknown-elf-g++


SRC = src/main.cpp src/image_io.cpp


HOST_OUT = host_canny
RV_OUT = rv_canny


all: host canny_rv

host:
	$(HOST_CXX) $(SRC) -o $(HOST_OUT)

canny_rv:
	$(RV_CXX) $(SRC) -o $(RV_OUT)

run: canny_rv
	qemu-riscv64 $(RV_OUT)

clean:
	rm -f $(HOST_OUT) $(RV_OUT) images/test_image.raw