#!/bin/bash

echo "name;type;duration_in_ms;mean_absolute_difference;obstacle_ratio;distance;grid_size;priority_queues;map_seed;memory_config;num_repetition" > result.txt

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D EXPERIMENT_NAME="\"memory\"" \
    -D USE_ZC_MEMORY \
    -D RUN_PARALLEL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D EXPERIMENT_NAME="\"memory\"" \
    -D USE_PINNED_MEMORY \
    -D RUN_PARALLEL && \
     ./cuda

#: '
nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D MAP_SPARSITY="(0.001)" \
    -D EXPERIMENT_NAME="\"map_sparse\"" \
    -D RUN_SEQUENTIAL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D MAP_SPARSITY="(0.015)" \
    -D EXPERIMENT_NAME="\"map_sparse\"" \
    -D RUN_SEQUENTIAL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D MAP_SPARSITY="(0.001)" \
    -D EXPERIMENT_NAME="\"map_sparse\"" \
    -D RUN_PARALLEL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D MAP_SPARSITY="(0.015)" \
    -D EXPERIMENT_NAME="\"map_sparse\"" \
    -D RUN_PARALLEL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D EXPERIMENT_NAME="\"prio_queue\"" \
    -D THREADS_PER_BLOCK="(256)" \
    -D QUEUE_BLOCKS="(4)" \
    -D RUN_PARALLEL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D EXPERIMENT_NAME="\"prio_queue\"" \
    -D THREADS_PER_BLOCK="(256)" \
    -D QUEUE_BLOCKS="(3)" \
    -D RUN_PARALLEL && \
     ./cuda


nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D EXPERIMENT_NAME="\"prio_queue\"" \
    -D QUEUE_BLOCKS="(1)" \
    -D RUN_PARALLEL && \
     ./cuda


nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D EXPERIMENT_NAME="\"prio_queue\"" \
    -D THREADS_PER_BLOCK="(256)" \
    -D RUN_PARALLEL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D EXPERIMENT_NAME="\"prio_queue\"" \
    -D THREADS_PER_BLOCK="(128)" \
    -D RUN_PARALLEL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024 + 512)" \
    -D EXPERIMENT_NAME="\"grid_size\"" \
    -D RUN_PARALLEL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024 + 512)" \
    -D EXPERIMENT_NAME="\"grid_size\"" \
    -D RUN_SEQUENTIAL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D EXPERIMENT_NAME="\"grid_size\"" \
    -D RUN_PARALLEL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D EXPERIMENT_NAME="\"grid_size\"" \
    -D RUN_SEQUENTIAL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE=512 \
    -D EXPERIMENT_NAME="\"grid_size\"" \
    -D RUN_PARALLEL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE=512 \
    -D EXPERIMENT_NAME="\"grid_size\"" \
    -D RUN_SEQUENTIAL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE=256 \
    -D EXPERIMENT_NAME="\"grid_size\"" \
    -D RUN_PARALLEL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE=256 \
    -D EXPERIMENT_NAME="\"grid_size\"" \
    -D RUN_SEQUENTIAL && \
     ./cuda


nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE=64 \
    -D EXPERIMENT_NAME="\"grid_size\"" \
    -D RUN_PARALLEL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE=64 \
    -D EXPERIMENT_NAME="\"grid_size\"" \
    -D RUN_SEQUENTIAL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D MAP_SEED="(1)" \
    -D EXPERIMENT_NAME="\"map\"" \
    -D RUN_PARALLEL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D MAP_SEED="(1)" \
    -D EXPERIMENT_NAME="\"map\"" \
    -D RUN_SEQUENTIAL && \
     ./cuda
    
nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D MAP_SEED="(2)" \
    -D EXPERIMENT_NAME="\"map\"" \
    -D RUN_PARALLEL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D MAP_SEED="(2)" \
    -D EXPERIMENT_NAME="\"map\"" \
    -D RUN_SEQUENTIAL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D MAP_SEED="(3)" \
    -D EXPERIMENT_NAME="\"map\"" \
    -D RUN_PARALLEL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D MAP_SEED="(3)" \
    -D EXPERIMENT_NAME="\"map\"" \
    -D RUN_SEQUENTIAL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D MAP_SEED="(4)" \
    -D EXPERIMENT_NAME="\"map\"" \
    -D RUN_PARALLEL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D MAP_SEED="(4)" \
    -D EXPERIMENT_NAME="\"map\"" \
    -D RUN_SEQUENTIAL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D MAP_SEED="(5)" \
    -D EXPERIMENT_NAME="\"map\"" \
    -D RUN_PARALLEL && \
     ./cuda

nvcc sequential.cu cuda.cu cuda_prefix.cu -o cuda -lm -I'./includes/' \
    -D GRID_SIZE="(1024)" \
    -D MAP_SEED="(5)" \
    -D EXPERIMENT_NAME="\"map\"" \
    -D RUN_SEQUENTIAL && \
     ./cuda
#'