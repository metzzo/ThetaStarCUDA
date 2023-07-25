# How to run
* Download boost https://www.boost.org/users/download/ (1.82)
* Extract to ./includes folder
* nvcc sequential.cu cuda.cu -o cuda -lm -I'./includes/' && ./cuda

Run test suite:
./compile_and_run.sh

Controlling the code is done through defines (see compile_and_run.sh for more details).