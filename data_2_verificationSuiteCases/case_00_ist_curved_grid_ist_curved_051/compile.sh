#!/bin/bash

rm -f ../src/*.o

lrefresco -set_source=../src/set_source.F90 -set_phi=../src/set_phi.F90 -post=../src/post.F90 \
    -extras="../src/extra_2.F90 ../src/extra_3.F90 ../src/extra_4.F90 ../src/extra_5.F90 ../src/extra_6.F90" \
    -exec=refresco

rm -f ../src/*.o
