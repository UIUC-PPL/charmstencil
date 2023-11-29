CHARMDIR=/home/adityapb/charm/charm
CHARMC=$(CHARMDIR)/mpi-linux-x86_64/bin/charmc
CHARM_INC=-I$(CHARMDIR)/include
OPTS=-c++-option -std=c++17 -O3 -g -ldl -I/u/bhosale/fmt/include -DNDEBUG

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

server.out: server.o
	$(CHARMC) -language charm++ -o $@ server.o

run-server: server.out
	LD_LIBRARY_PATH=/home/sb56/anaconda3/envs/charm/lib ./charmrun +p4 ./server.out ++server ++server-port 10000

clean:
	rm *.decl.h *.def.h *.out *.o charmrun
