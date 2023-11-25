CHARMDIR=/home/sb56/charm/
CHARMC=$(CHARMDIR)/mpi-linux-x86_64-cuda/bin/charmc
CHARM_INC=-I$(CHARMDIR)/include
NVCC=nvcc
NVCC_FLAGS=-c -std=c++17  -use_fast_math -O3 -I/home/sb56/fmt/include -L/home/sb56/anaconda3/envs/charm/lib/
OPTS=-c++-option -std=c++17 -O3 -g -ldl -I/home/sb56/fmt/include -DNDEBUG
LD_LIBS=-lcuda -L/home/sb56/anaconda3/envs/charm/lib/

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
	LD_LIBRARY_PATH=/home/sb56/anaconda3/envs/charm/lib ./charmrun +p4 ./server.out ++server ++server-port 10000

clean:
	rm *.decl.h *.def.h *.out *.o charmrun
