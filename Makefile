# choose your compiler, e.g. gcc/clang
# example override to clang: make run CC=clang
CC = gcc

# the most basic way of building that is most likely to work on most systems
.PHONY: runcc
runcc: run.cc
	hipcc -o runcc run.cc -O2 --offload-arch=gfx908 

.PHONY: clean
clean:
	rm -f runcc
