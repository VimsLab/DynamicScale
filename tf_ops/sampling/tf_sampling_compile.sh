#!/bin/sh
/usr/local/cuda/bin/nvcc --gpu-architecture=sm_35 --compiler-options -Wall -Xcompiler -fopenmp -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart tf_sampling_g.cu -o tf_sampling_g.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -lcusolver -lcurand -lcublas -lcusparse -g

TF_CFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') 
echo $TF_CFLAGS
echo $TF_LFLAGS

g++ -std=c++11 -shared tf_sampling.cpp tf_sampling_g.cu.o -o tf_sampling_so.so -fPIC ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -I /usr/local/cuda/include -L /usr/local/cuda/lib64 -O2 -D_GLIBCXX_USE_CXX11_ABI=1 -g -lgomp -lcusolver

