GCC = /usr/bin/gcc
NVCC = /usr/local/cuda-5.0/bin/nvcc

GCC_FLAGS = -Wall
NVCC_FLAGS = -use_fast_math

CFLAGS += -O3

CFLAGS +=
LDFLAGS +=

all: bin/computeFTLE bin/computeFTLE_CUDA bin/graph_data

clean:
	/bin/rm bin/*
	
bin/computeFTLE: src/computeFTLE.c
	${GCC} ${GCC_FLAGS} ${CFLAGS} ${LDFLAGS} -o bin/computeFTLE -funroll-loops src/computeFTLE.c -lm

bin/computeFTLE_CUDA: src/computeFTLE_CUDA.cu src/computeFTLE_CUDA.h src/vectorfield.cu src/vectorfield.h src/RK4.cu src/RK4.h src/settings.h
	${NVCC} ${NVCC_FLAGS} ${CFLAGS} ${LDFLAGS} -o bin/computeFTLE_CUDA -lpthread -use_fast_math src/computeFTLE_CUDA.cu -lm

bin/graph_data: src/graph_data.c src/uthash.h
	${GCC} ${GCC_FLAGS} ${CFLAGS} ${LDFLAGS} -o bin/graph_data -O3 src/graph_data.c -lgd -lm
