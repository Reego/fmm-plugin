BLAS_DIR := $(HOME)/blis
STRASSEN_DIR := ../

CC = gcc-13
CXX = g++

ARCH = gcc-ar-13
ARCHFLAGS = cr
RANLIB = gcc-ranlib-13

COMPILER_OPT_LEVEL=O0

CFLAGS = -$(COMPILER_OPT_LEVEL) -g -fopenmp -m64 -mavx2 -fPIC -march=native
LDFLAGS = -lpthread -lm -fopenmp

INC_DIR =include/

