include Makefile.in

SRC = ./SRC/utils/memory.o\
	./SRC/utils/protos.o\
	./SRC/utils/utils.o\
	./SRC/linearalg/vector.o\
	./SRC/linearalg/vecops.o\
	./SRC/linearalg/matops.o\
	./SRC/linearalg/kernels.o\
	./SRC/linearalg/ordering.o\
	./SRC/linearalg/rankest.o\
	./SRC/solvers/solvers.o\
	./SRC/solvers/pcg.o\
	./SRC/solvers/fgmres.o\
	./SRC/solvers/lanczos.o\
	./SRC/preconds/precond.o\
	./SRC/preconds/chol.o\
	./SRC/preconds/fsai.o\
	./SRC/preconds/nys.o\
	./SRC/optimizer/transform.o\
	./SRC/optimizer/optimizer.o\
	./SRC/optimizer/gp_loss.o\
	./SRC/optimizer/gp_predict.o\
	./SRC/optimizer/gp_problem.o\
	./SRC/optimizer/adam.o\
	./SRC/external/nfft_interface.o

RELATIVE_PATH = ./

INC = -I./INC $(INCLAPACKBLAS)
LIB_BLASLAPACK = $(LIBLAPACKBLAS)

LIB = -L$(RELATIVE_PATH)$(NFFT4GP_PATH)/lib -lnfft4gp\
	$(LIB_BLASLAPACK) $(LIB_ARPACK) -lm

ifneq ($(USING_MKL),0)
LIB += -mkl
endif

ifneq ($(USING_NFFT),0)
INC += -I$(NFFT_INSTALL_PATH)/include -I$(NFFT_PATH)/include -I$(NFFT_PATH)/applications/fastsum 
INC += -I$(FFTW_PATH)/../include
LIB += -L$(FFTW_PATH) -lfftw3
LIB += -L$(NFFT_INSTALL_PATH)/lib -lnfft3 -lnfft3_threads
LIB += $(NFFT_PATH)/applications/fastsum/fastsum.o $(NFFT_PATH)/applications/fastsum/kernels.o
endif

default: libnfft4gp.a
all: libnfft4gp.a
lib: libnfft4gp.a

%.o : %.c
	$(CC) $(FLAGS) $(INC) -o $@ -c $<

libnfft4gp.a: $(SRC)
	$(AR) $@ $(SRC)
	$(RANLIB) $@
	rm -rf build;mkdir build;mkdir build/lib;mkdir build/include;
	cp libnfft4gp.a build/lib;cp INC/*.h build/include;
	$(CC) -shared $(FLAGS) $(INC) -o libnfft4gp.so $(SRC) $(LIB)
	cp libnfft4gp.so build/lib;
clean:
	rm -rf ./SRC/*.o;rm -rf ./SRC/optimizer/*.o;rm -rf ./SRC/utils/*.o;rm -rf ./SRC/linearalg/*.o;rm -rf ./SRC/solvers/*.o;rm -rf ./SRC/preconds/*.o;rm -rf ./SRC/external/*.o;rm -rf ./build;rm -rf *.a;rm -rf *.so;
