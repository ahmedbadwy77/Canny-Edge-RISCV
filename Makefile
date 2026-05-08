# المترجمات (Compilers)
HOST_CXX = g++
# المتغير ده بيسمحلك تغير الـ Optimization من بره (الافتراضي هو -O0) [cite: 1003-1005]
OPT ?= -O0
RV_CXX = riscv64-unknown-elf-g++ $(OPT)

# ملفات الكود الأساسية
SRC = src/main.cpp src/image_io.cpp

# أسماء الملفات الناتجة
HOST_OUT = host_canny
RV_OUT = rv_canny

# الأهداف (Targets)
all: host canny_rv

host:
	$(HOST_CXX) $(SRC) -o $(HOST_OUT)

canny_rv:
	$(RV_CXX) $(SRC) -o $(RV_OUT)

run: canny_rv
	qemu-riscv64 $(RV_OUT)

# هدف تشغيل الاختبارات باستخدام GoogleTest [cite: 798-810]
test:
	$(HOST_CXX) tests/test_pipeline.cpp src/image_io.cpp -o run_tests -lgtest -lgtest_main -pthread
	./run_tests

clean:
	rm -f $(HOST_OUT) $(RV_OUT) run_tests images/*.raw