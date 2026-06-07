NVCC      = nvcc
NVCCFLAGS = -O3 -arch=sm_86 --std=c++17 -I.
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

clean:
	rm -f $(TARGET) $(OBJS)
