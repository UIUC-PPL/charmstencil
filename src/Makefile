CHARMDIR=/u/bhosale/charm/
CHARMC=$(CHARMDIR)/mpi-linux-x86_64-cuda/bin/charmc
CHARM_INC=-I$(CHARMDIR)/include
NVCC=nvcc
NVCC_FLAGS=-c -std=c++17  -use_fast_math -O3 -I/u/bhosale/fmt/include 
OPTS=-c++-option -std=c++17 -O3 -g -ldl -I/u/bhosale/fmt/include -DNDEBUG
LD_LIBS=-lcuda

all: server.out

.PHONY: clean server.out

server.decl.h: server.ci server.hpp
	$(CHARMC) -E server.ci

server.def.h: server.ci server.hpp

stencil.decl.h: stencil.ci stencil.hpp
	$(CHARMC) -E stencil.ci

stencil.def.h: stencil.ci stencil.hpp

server.o: server.cpp server.decl.h server.def.h server.hpp stencil.decl.h stencil.def.h stencil.hpp codegen.hpp
	$(CHARMC) -c $< $(OPTS)

stencilCUDA.o: stencil.cu stencil.hpp
	$(NVCC) -o $@ $(NVCC_FLAGS) $(CHARM_INC) $<

server.out: server.o stencilCUDA.o
	$(CHARMC) -language charm++ -o $@ server.o stencilCUDA.o $(LD_LIBS)

run-server: server.out
	./charmrun +p4 ./server.out ++server ++server-port 10000 ++local

clean:
	rm *.decl.h *.def.h *.out *.o charmrun
