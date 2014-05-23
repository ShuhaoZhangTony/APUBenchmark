#!/bin/bash

make clean
make

for i in 0 1 2 4 8 16 32 50 64 75 100
do
	echo "gpu_offset:$i%"
	./GEMM --KiB 16 --gpu_offset "$i"
done
