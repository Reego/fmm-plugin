
# Define the name of the config makefile.
CONFIG_MK_FILE := plugin/config.mk

# Include the configuration file.
-include $(CONFIG_MK_FILE) 

#CC = gcc-13
#CXX = g++

#ARCH = gcc-ar-13
#ARCHFLAGS = cr
#RANLIB = gcc-ranlib-13

BLIS_INCDIR := $(includedir)/blis
STRASSEN_DIR := ../

# Define the name of the common makefile.
COMMON_MK_FILE := $(sharedir)/blis/common.mk

# Include the configuration file.
# include $(COMMON_MK_FILE)

COMPILER_OPT_LEVEL=O0

CFLAGS = -$(COMPILER_OPT_LEVEL) -g -fopenmp -m64 -mavx2 -fPIC -march=native
LDFLAGS = -lpthread -lm -fopenmp

$(info * Using CFLAGS=$(CFLAGS))
$(info * Using LDFLAGS=$(LDFLAGS))
$(info * Using LDLIBS=$(LDLIBS))

INC_DIR = -Iframe/include -Iplugin

FMM_LIB = lib/libfmm.a
PLUGIN_LIB = plugin/lib/haswell/libblis_fmm_blis.a

LIBBLIS = $(libdir)/libblis.a

FRAME_CC_SRC= 	frame/util/bli_fmm_util.c \
				frame/base/bli_strassen_ab.c

# KERNEL_SRC=     plugin/ref_kernels/bli_packm_fmm_ref.c \
# 				plugin/ref_kernels/bli_gemm_fmm_ref.c

TEST_SRC = test/test_strassen_oapi.c

OTHER_DEP = 	frame/include/bli_fmm.h 

CFLAGS += $(INC_DIR) -I$(BLIS_INCDIR)
                             
FMM_LIB_OBJ=$(FRAME_CC_SRC:.c=.o)
TEST_OBJ=$(TEST_SRC:.c=.o) 
TEST_EXE= test_strassen.x

all: $(FMM_LIB) test

lib:  $(FMM_LIB)

test: $(TEST_EXE)

$(TEST_EXE): $(TEST_OBJ) $(FMM_LIB)
	$(CC) $(CFLAGS) $(TEST_OBJ) -o $(TEST_EXE) $(LDFLAGS) $(LIBBLIS) $(PLUGIN_LIB) $(FMM_LIB)

$(FMM_LIB): $(FMM_LIB_OBJ)
	-mkdir -p lib
	$(AR) $(ARFLAGS) $@ $(FMM_LIB_OBJ)
	$(RANLIB) $@

# ---------------------------------------------------------------------------
# Object files compiling rules
# ---------------------------------------------------------------------------
%.o: %.c $(OTHER_DEP)
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

# ---------------------------------------------------------------------------

clean:
	-rm -f $(FMM_LIB_OBJ) $(FMM_LIB) test/*.o *.x
	-rm -r lib
	
#$(MAKE) clean -f Makefile -C test
