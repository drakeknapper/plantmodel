
CC=gcc
CPP=gcc
CFLAGS=-O2 -march=x86-64 -fPIC -pipe
LDFLAGS=-lm

plant: plant.o lib
	gcc -shared plant.o -o lib/_plant.so -lm

.c.o:
	$(CC) -c $< $(CFLAGS)
