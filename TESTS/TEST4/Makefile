include ../../Makefile.in

RELATIVE_PATH = ../../

INC = -I$(RELATIVE_PATH)$(NFFT4GP_PATH)/include $(INCLAPACKBLAS)
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

ALLTESTS = foo.ex

default: $(ALLTESTS)
all: $(ALLTESTS)

%.o : %.cpp
	$(CXX) $(FLAGSCPP) $(INC) -o $@ -c $<

%.o : %.c
	$(CXX) $(FLAGS) $(INC) -o $@ -c $<

foo.ex: foo.o $(SRC)
	$(LINKXX) $(FLAGSCPP) $(INC) -o foo.ex foo.o $(SRC) $(LIB)

clean:
	rm -rf *.ex;rm -rf *.o;rm -rf *.a;rm -rf ./TempData;
