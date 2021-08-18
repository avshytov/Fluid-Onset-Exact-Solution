PYTH_INC = -I/usr/include/python2.7   
NUMPY_INC = -I /usr/lib64/python2.7/site-packages/numpy/core/include/
NUMPY_INC += -I /usr/lib64/python2.7/site-packages/numpy/core/include/numpy 
INC = $(NUMPY_INC) $(PYTH_INC)
LIB = -lm -lpython2.7

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
