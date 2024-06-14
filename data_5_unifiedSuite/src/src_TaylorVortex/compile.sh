#!/bin/bash

rm -f *.o

lrefresco -set_phi=set_phi.F90 -post=post.F90 -common_user_code=common_user_code.F90 \
    -set_bodyforce=set_bodyforce.F90 -set_source=set_source.F90 \
    -exec=refresco

rm -f *.o