CHARMC=/home/adityapb/charm/charm/netlrts-linux-x86_64/bin/charmc
OPTS=-c++-option -std=c++17 -O3 -g -ldl -DNDEBUG

all: server

.PHONY: clean server.out

server_ci: server.ci
	$(CHARMC) -E server.ci

stencil_ci: stencil.ci
	$(CHARMC) -E stencil.ci

server: server.cpp server_ci stencil_ci
	$(CHARMC) $< -o $@.out $(OPTS)

run-server: server.out
	./charmrun +p4 ./server.out ++server ++server-port 10000 ++local

clean:
	rm *.decl.h *.def.h *.out charmrun
