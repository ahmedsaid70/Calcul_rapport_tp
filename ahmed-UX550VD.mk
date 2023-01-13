#######################################
# ambre.mk
# Default options for ambre computer
#######################################
CC=gcc
LIBSLOCAL=-L/usr/lib -llapack -llapacke -lcblas -lblas -lm
INCLUDEBLASLOCAL=-I//usr/include/x86_64-linux-gnu
OPTCLOCAL=-fPIC -march=native