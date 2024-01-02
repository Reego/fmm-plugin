include ../make.inc.files/make.gnu.inc

LDFLAGS  += ${BLAS_DIR}/lib/libblis.a

CFLAGS += -I$(STRASSEN_DIR)$(INC_DIR) -I$(BLAS_DIR)/include/blis/

STRASSENLIB = $(STRASSEN_DIR)/lib/lib222-1_ab.a

BLISLAB_TEST_CC_SRC= test_222-1_ab.c test_strassen_oapi.c
BLISLAB_TEST_CC_OBJS= test_222-1_ab.o test_strassen_oapi.o

BLISLAB_TEST_EXE= test_strassen.x

all: $(BLISLAB_TEST_EXE)

clean:
	rm -f $(BLISLAB_TEST_EXE)

# ---------------------------------------------------------------------------
# Executable files compiling rules
# ---------------------------------------------------------------------------
%.x: $(BLISLAB_TEST_CC_OBJS) $(OTHER_DEP)
	$(CC) $(CFLAGS) $(BLISLAB_TEST_CC_OBJS) -o $@ $(LDFLAGS) $(STRASSENLIB)

%.x: %.cpp $(OTHER_DEP)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS) $(STRASSENLIB)
