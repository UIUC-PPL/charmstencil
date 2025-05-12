CHARMDIR=/home/aditya/charm/
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

server.o: server.cpp server.decl.h server.def.h server.hpp stencil.decl.h stencil.def.h stencil.hpp codegen.hpp array.hpp utils.hpp
	$(CHARMC) -c $< $(OPTS) $(CHARM_INC)

utils.o: utils.cpp utils.hpp
	$(CHARMC) -c $< $(OPTS) $(CHARM_INC)

ast.o: ast.cpp ast.hpp array.hpp utils.hpp
	$(CHARMC) -c $< $(OPTS) $(CHARM_INC)

dag.o: dag.cpp dag.hpp array.hpp utils.hpp ast.hpp
	$(CHARMC) -c $< $(OPTS) $(CHARM_INC)

codegen.o: codegen.cpp codegen.hpp array.hpp utils.hpp ast.hpp
	$(CHARMC) -c $< $(OPTS) $(CHARM_INC)

stencil.o: stencil.cpp stencil.hpp array.hpp utils.hpp ast.hpp dag.hpp
	$(CHARMC) -c $< $(OPTS) $(CHARM_INC)

stencilCUDA.o: stencil.cu stencil.hpp
	$(NVCC) -o $@ $(NVCC_FLAGS) $(CHARM_INC) $<

server.out: server.o stencilCUDA.o stencil.o ast.o utils.o codegen.o dag.o
	$(CHARMC) -language charm++ -o $@ server.o stencilCUDA.o ast.o dag.o codegen.o utils.o stencil.o $(LD_LIBS)

run-server: server.out
	LD_LIBRARY_PATH=/home/sb56/anaconda3/envs/charm/lib ./charmrun +p4 ./server.out ++server ++server-port 10000

clean: clean-gen
	rm *.decl.h *.def.h *.out *.o charmrun

clean-gen:
	rm generated/*
