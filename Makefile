NVCC      = nvcc
NVCCFLAGS = -O3 -arch=sm_86 --std=c++17 -lineinfo -I.
LDFLAGS   = -lcublas

SRCS   = bench.cu $(wildcard kernels/*.cu)
OBJS   = $(SRCS:.cu=.o)
TARGET = sgemm

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

run: $(TARGET)
	./$(TARGET)

sanitize: $(TARGET)
	compute-sanitizer --tool memcheck --destroy-on-device-error kernel ./$(TARGET) 256

clean:
	rm -f $(TARGET) $(OBJS)
