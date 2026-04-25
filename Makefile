HOST_CXX = g++
RV_CXX = riscv64-unknown-elf-g++


test:
	@echo "Running tests on Host..."

canny_rv:
	@echo "Compiling for RISC-V..."

run:
	@echo "Running on QEMU..."

clean:
	@echo "Cleaning up..."
	rm -rf *.o