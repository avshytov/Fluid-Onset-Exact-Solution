PYTH_INC = -I/usr/include/python3.8   
NUMPY_INC = -I /usr/lib/python3.8/site-packages/numpy/core/include/
NUMPY_INC += -I /usr/lib/python3.8/site-packages/numpy/core/include/numpy 
INC = $(NUMPY_INC) $(PYTH_INC)
LIB = -lm -lpython3.8

CC = gcc -g -O2 -fPIC
CXX = $(CC)
#gcc -g -O2
#-DNPY_1_11_API_VERSION
LD = gcc -shared -g
LDFLAGS = $(LIB)
CFLAGS = -fPIC   $(INC) -Wall
CXXFLAGS = $(CFLAGS)

FOURIERTARGET = _makequad.so
FOURIERSRC = makequad.cc
FOURIEROBJ = $(FOURIERSRC:.cc=.o)
TARGETS = $(FOURIERTARGET)

all: $(TARGETS)

#%.o: %.c
#	$(CC) $(CFLAGS) -o $@ -c $<

%.o: %.cc
	$(CXX) $(CXXFLAGS) -o $@ -c $<


$(FOURIERTARGET): $(FOURIEROBJ) $(FOURIERSRC)
	$(LD) -o $@ $(FOURIEROBJ)  $(LDFLAGS)
