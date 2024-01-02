include make.inc.files/make.gnu.inc

$(info * Using CFLAGS=$(CFLAGS))
$(info * Using LDFLAGS=$(LDFLAGS))
$(info * Using LDLIBS=$(LDLIBS))


STRASSENLIB = lib/lib222-1_ab.a

FRAME_CC_SRC= 	dgemm/bl_dgemm_ref.c \
				dgemm/my_dgemm.c \
				dgemm/bl_dgemm_util.c \
                dgemm/my_dgemm_222-1_ab.c \
				dgemm/bli_strassen_ab.c

KERNEL_SRC=     kernels/bl_dgemm_asm_8x6.c \
				kernels/bl_dgemm_asm_8x6_mulstrassen.c

OTHER_DEP = include/bl_dgemm.h 



CFLAGS += -I$(INC_DIR) -I$(BLAS_DIR)/include/blis/
                             

STRASSENLIB_OBJ=$(FRAME_CC_SRC:.c=.o) $(KERNEL_SRC:.c=.o) 

all: $(STRASSENLIB) TESTBLISLAB

TESTBLISLAB: $(STRASSENLIB)
	cd ./test && $(MAKE) -f Makefile && cd $(STRASSEN_DIR)

$(STRASSENLIB): $(STRASSENLIB_OBJ)
	$(ARCH) $(ARCHFLAGS) $@ $(STRASSENLIB_OBJ)
	$(RANLIB) $@

# ---------------------------------------------------------------------------
# Object files compiling rules
# ---------------------------------------------------------------------------
%.o: %.c $(OTHER_DEP)
	$(CC) $(CFLAGS) -c $< -o $@ $(LDFLAGS)

# ---------------------------------------------------------------------------

clean:
	-rm $(STRASSENLIB_OBJ) $(STRASSENLIB) kernels/*.o
	$(MAKE) clean -f Makefile -C test
