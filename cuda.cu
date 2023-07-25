#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <vector>
#include <assert.h>
#include <chrono>
#include <fstream>
#include <algorithm>

#include "sequential.h"

// #define USE_PINNED_MEMORY
//  #define USE_ZC_MEMORY

#if defined(USE_PINNED_MEMORY) && defined(USE_ZC_MEMORY)
#error "USE_PINNED_MEMORY and USE_ZC_MEMORY cannot be defined at the same time"
#endif

#ifndef EXPERIMENT_NAME
#define EXPERIMENT_NAME "DEFAULT"
#endif

#ifndef MAP_SEED
#define MAP_SEED (1688035615)
#endif

#ifndef EXPERIMENT_REPETITION
#define EXPERIMENT_REPETITION (5)
#endif

#ifndef GRID_SIZE
#define GRID_SIZE (32)
#endif

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK (512)
#endif

#ifndef QUEUE_BLOCKS
#define QUEUE_BLOCKS (1)
#endif

#ifndef MAP_SPARSITY
#define MAP_SPARSITY (0.01)
#endif

#if !defined(RUN_SEQUENTIAL) && !defined(RUN_PARALLEL)
// 0 ... both, 1 ... only sequential, 2 ... only parallel
#define RUN_SEQUENTIAL
#define RUN_PARALLEL
#endif

#define NUM_QUEUES (THREADS_PER_BLOCK * QUEUE_BLOCKS)
#define LIST_CAPACITY (NUM_QUEUES * 10)
#define STATE_POOL_SIZE (10000000)
#define OPEN_LIST_SIZE (GRID_SIZE * GRID_SIZE)

// #define CUDA_DEBUG
#define CUDA_TIME_DEBUG
#define CHECK_CUDA

#ifdef CHECK_CUDA
#define gpuErrchk()                    \
    {                                  \
        gpuAssert(__FILE__, __LINE__); \
    }
#else
#define gpuErrchk() (0)
#endif
inline void gpuAssert(const char *file, int line, bool abort = true)
{
    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
        {
            getchar();
            exit(code);
        }
    }
}

void prefixSum(int *d_out, int *d_in, int length);

int line_of_sight(int x0, int y0, int x1, int y1, int *map, int size)
{
    int dx = abs(x1 - x0);
    int dy = -abs(y1 - y0);
    int sx = (x0 < x1) ? 1 : -1;
    int sy = (y0 < y1) ? 1 : -1;
    int error = dx + dy;

    while (1)
    {
        // use inside CUDA => no map available
        if (x0 >= 0 && x0 < size && y0 >= 0 && y0 < size)
        {
            // so we do not "slip" through tight edges, we expand the collision check to a 2x2 area (defined by sx and sy)
            if (map[x0 + y0 * size] == OBSTACLE || map[(x0 + sx) + y0 * size] == OBSTACLE || map[(x0) + (y0 + sy) * size] == OBSTACLE || map[(x0 + sx) + (y0 + sy) * size] == OBSTACLE)
            {
                return 0;
            }
        }
        else
        {
            return 0;
        }

        if (x0 == x1 && y0 == y1)
        {
            break;
        }

        int e2 = 2 * error;
        if (e2 >= dy)
        {
            if (x0 == x1)
            {
                break;
            }
            error += dy;
            x0 += sx;
        }
        if (e2 <= dx)
        {
            if (y0 == y1)
            {
                break;
            }
            error += dx;
            y0 += sy;
        }
    }
    return 1;
}

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

typedef struct MapData
{
    int size;
    int *tiles;
} MapData;

MapData create_map_data(Map *map)
{
    MapData map_data;
    map_data.size = map->size;
    map_data.tiles = (int *)malloc(sizeof(int) * map->size * map->size);
    memcpy(map_data.tiles, map->data, sizeof(int) * map->size * map->size);
    return map_data;
}

void map_data_to_device(MapData &map, MapData &d_map)
{
    cudaMemcpy(d_map.tiles, map.tiles, sizeof(int) * map.size * map.size, cudaMemcpyDefault);
    gpuErrchk();
    d_map.size = map.size;
}

MapData create_map_data_device(MapData map)
{
    MapData d_map;
    d_map.size = map.size;
    cudaMalloc(&d_map.tiles, sizeof(int) * map.size * map.size);
    gpuErrchk();
    return d_map;
}

typedef struct State
{
    int x;
    int y;
    double g;
    double h;
    double f;
    // int parent_x, parent_y;
    State *parent;
} State;

typedef struct StatePool
{
    State *states;
    int *num_states;
} StatePool;

StatePool create_state_pool()
{
    StatePool state_pool;
    state_pool.states = (State *)malloc(sizeof(State) * STATE_POOL_SIZE);
    // no special memory considerations, because there are no significant mem copies of state pool data
    state_pool.num_states = (int *)malloc(sizeof(int));

    state_pool.num_states = (int *)malloc(sizeof(int));
    state_pool.num_states[0] = 0;
    return state_pool;
}

StatePool create_state_pool_device(StatePool &state_pool)
{
    StatePool d_state_pool;
    d_state_pool.num_states = (int *)malloc(sizeof(int));
    cudaMalloc(&d_state_pool.states, sizeof(State) * STATE_POOL_SIZE);
    gpuErrchk();
    cudaMalloc(&d_state_pool.num_states, sizeof(int));
    gpuErrchk();
    return d_state_pool;
}

void state_pool_to_device(StatePool &state_pool, StatePool &d_state_pool)
{
    cudaMemcpy(d_state_pool.states, state_pool.states, sizeof(State) * state_pool.num_states[0], cudaMemcpyDefault);
    gpuErrchk();
    cudaMemcpy(d_state_pool.num_states, state_pool.num_states, sizeof(int), cudaMemcpyDefault);
    gpuErrchk();
}

void state_pool_to_host(StatePool &state_pool, StatePool &d_state_pool)
{
    cudaMemcpy(state_pool.states, d_state_pool.states, sizeof(State) * state_pool.num_states[0], cudaMemcpyDefault);
    gpuErrchk();
    cudaMemcpy(state_pool.num_states, d_state_pool.num_states, sizeof(int), cudaMemcpyDefault);
    gpuErrchk();
}

typedef struct StateList
{
    State **states;
    int size;
    int *num_states;
} StateList;

StateList create_state_list(int size)
{
    StateList state_list;
    state_list.size = size;
#ifdef USE_PINNED_MEMORY
    cudaMallocHost(&state_list.num_states, sizeof(int));
#elif defined(USE_ZC_MEMORY)
    cudaHostAlloc(&state_list.num_states, sizeof(int), cudaHostAllocMapped);
#else
    state_list.num_states = (int *)malloc(sizeof(int));
#endif
    gpuErrchk();
    state_list.states = (State **)malloc(sizeof(State *) * size);
    state_list.num_states[0] = 0;
    return state_list;
}

void state_list_to_device(StateList &state_list, StateList &d_state_list)
{
#ifndef USE_ZC_MEMORY
    cudaMemcpy(d_state_list.num_states, state_list.num_states, sizeof(int), cudaMemcpyDefault);
    gpuErrchk();
#endif
    cudaMemcpy(d_state_list.states, state_list.states, sizeof(State *) * state_list.num_states[0], cudaMemcpyDefault);
    gpuErrchk();
    d_state_list.size = state_list.size;
}

void state_list_to_host(StateList &state_list, StateList &d_state_list)
{
#ifndef USE_ZC_MEMORY
    cudaMemcpy(state_list.num_states, d_state_list.num_states, sizeof(int), cudaMemcpyDefault);
    gpuErrchk();
#endif
    printf("num_state (state_list_to_host): %d\n", state_list.num_states[0]);
    cudaMemcpy(state_list.states, d_state_list.states, sizeof(State *) * state_list.num_states[0], cudaMemcpyDefault);
    gpuErrchk();
    state_list.size = d_state_list.size;
}

StateList create_state_list_device(StateList &state_list)
{
    StateList d_state_list;
    d_state_list.size = state_list.size;
    cudaMalloc(&d_state_list.states, sizeof(State *) * state_list.size);
    gpuErrchk();
#ifdef USE_ZC_MEMORY
    cudaHostGetDevicePointer(&d_state_list.num_states, state_list.num_states, 0);
#else
    cudaMalloc(&d_state_list.num_states, sizeof(int));
#endif
    gpuErrchk();
    return d_state_list;
}

typedef struct ClosedGrid
{
    State **data;
    char *assigned;
    int size;
} ClosedGrid;

void reset_closed_grid(ClosedGrid &closed_grid)
{
    memset(closed_grid.assigned, 0, sizeof(char) * closed_grid.size * closed_grid.size);
}

ClosedGrid create_closed_grid(int size)
{
    ClosedGrid closed_grid;
    closed_grid.size = size;
    closed_grid.data = (State **)malloc(sizeof(State *) * size * size);
    closed_grid.assigned = (char *)malloc(sizeof(char) * size * size);
    reset_closed_grid(closed_grid);
    return closed_grid;
}

void closed_grid_to_device(ClosedGrid &closed_grid, ClosedGrid &d_closed_grid)
{
    cudaMemcpy(d_closed_grid.data, closed_grid.data, sizeof(State *) * closed_grid.size * closed_grid.size, cudaMemcpyDefault);
    gpuErrchk();
    cudaMemcpy(d_closed_grid.assigned, closed_grid.assigned, sizeof(char) * closed_grid.size * closed_grid.size, cudaMemcpyDefault);
    gpuErrchk();
    d_closed_grid.size = closed_grid.size;
}

void closed_grid_to_host(ClosedGrid &closed_grid, ClosedGrid &d_closed_grid)
{
    cudaMemcpy(closed_grid.data, d_closed_grid.data, sizeof(State *) * closed_grid.size * closed_grid.size, cudaMemcpyDefault);
    gpuErrchk();
    cudaMemcpy(closed_grid.assigned, d_closed_grid.assigned, sizeof(char) * closed_grid.size * closed_grid.size, cudaMemcpyDefault);
    gpuErrchk();
    closed_grid.size = d_closed_grid.size;
}

ClosedGrid create_closed_grid_device(ClosedGrid &closed_grid)
{
    ClosedGrid d_closed_grid;
    d_closed_grid.size = closed_grid.size;
    cudaMalloc(&d_closed_grid.data, sizeof(State *) * closed_grid.size * closed_grid.size);
    gpuErrchk();
    cudaMalloc(&d_closed_grid.assigned, sizeof(char) * closed_grid.size * closed_grid.size);
    gpuErrchk();
    return d_closed_grid;
}

typedef struct DuplicateGrid
{
    float *min_f; // due to atomic => float and not double
    int size;
} DuplicateGrid;

void reset_duplicate_grid(DuplicateGrid &duplicate_grid)
{
    memset(duplicate_grid.min_f, 0, sizeof(float) * duplicate_grid.size * duplicate_grid.size);
}

DuplicateGrid create_duplicate_grid(int size)
{
    DuplicateGrid duplicate_grid;
    duplicate_grid.size = size;
    duplicate_grid.min_f = (float *)malloc(sizeof(float) * size * size);
    memset(duplicate_grid.min_f, 0, sizeof(float) * size * size);
    return duplicate_grid;
}

void duplicate_grid_to_device(DuplicateGrid &duplicate_grid, DuplicateGrid &d_duplicate_grid)
{
    cudaMemcpy(d_duplicate_grid.min_f, duplicate_grid.min_f, sizeof(float) * duplicate_grid.size * duplicate_grid.size, cudaMemcpyDefault);
    gpuErrchk();
    d_duplicate_grid.size = duplicate_grid.size;
}

void duplicate_grid_to_host(DuplicateGrid &duplicate_grid, DuplicateGrid &d_duplicate_grid)
{
    cudaMemcpy(duplicate_grid.min_f, d_duplicate_grid.min_f, sizeof(float) * duplicate_grid.size * duplicate_grid.size, cudaMemcpyDefault);
    gpuErrchk();
    duplicate_grid.size = d_duplicate_grid.size;
}

DuplicateGrid create_duplicate_grid_device(DuplicateGrid &duplicate_grid)
{
    DuplicateGrid d_duplicate_grid;
    d_duplicate_grid.size = duplicate_grid.size;
    cudaMalloc(&d_duplicate_grid.min_f, sizeof(float) * duplicate_grid.size * duplicate_grid.size);
    gpuErrchk();
    return d_duplicate_grid;
}

typedef struct PriorityQueue
{
    State **entries;
    int size;
    int *num_entries;
} PriorityQueue;

void queue_to_device(PriorityQueue &q, PriorityQueue &d_q)
{
    cudaMemcpy(d_q.num_entries, q.num_entries, sizeof(int), cudaMemcpyDefault);
    gpuErrchk();
    cudaMemcpy(d_q.entries, q.entries, sizeof(State *) * q.num_entries[0], cudaMemcpyDefault);
    gpuErrchk();

    d_q.size = q.size;
}

void queue_to_host(PriorityQueue &q, PriorityQueue &d_q)
{
    cudaMemcpy(q.num_entries, d_q.num_entries, sizeof(int), cudaMemcpyDefault);
    gpuErrchk();

    cudaMemcpy(q.entries, d_q.entries, sizeof(State *) * q.num_entries[0], cudaMemcpyDefault);
    gpuErrchk();
    q.size = d_q.size;
}

PriorityQueue create_queue_device(PriorityQueue &q)
{
    PriorityQueue d_q;
    d_q.size = q.size;

    cudaMalloc(&d_q.entries, sizeof(State *) * q.size);
    gpuErrchk();
    // no special memory considerations, because there are no significant mem copies of queue data
    cudaMalloc(&d_q.num_entries, sizeof(int));
    gpuErrchk();

    return d_q;
}

void reset_queue(PriorityQueue &q)
{
    q.num_entries[0] = 0;
}

PriorityQueue create_queue()
{
    PriorityQueue q;
    q.size = OPEN_LIST_SIZE;

    q.num_entries = (int *)malloc(sizeof(int));
    q.entries = (State **)malloc(sizeof(State *) * q.size);

    reset_queue(q);
    return q;
}

typedef struct PriorityQueueList
{
    PriorityQueue queues[NUM_QUEUES];
} PriorityQueueList;

typedef struct ThetaStarCudaDeviceData
{
    MapData map_data;
    StatePool state_pool;
    PriorityQueueList open_lists;
    StateList expanded_list;
    StateList new_expanded_list;
    ClosedGrid closed_grid;
    DuplicateGrid duplicate_grid;

    int *duplicate_criteria;
    int *duplicate_prefix_sum;

    int *is_at_goal;
    int *total_open_list_size;
    int *duplicate_size;

    int *path;
    int *path_length;

    int start_x;
    int start_y;
    int goal_x;
    int goal_y;
} ThetaStarCudaDeviceData;

__device__ State *create_state(ThetaStarCudaDeviceData *data, int x, int y, double g, double h, State *parent_state)
{
    int state_idx = atomicAdd(data->state_pool.num_states, 1);
    if (state_idx >= STATE_POOL_SIZE)
    {
        printf("State pool overflow %d < %d\n", state_idx, STATE_POOL_SIZE);
    }
    assert(state_idx < STATE_POOL_SIZE);

    State *state = &data->state_pool.states[state_idx];
    state->x = x;
    state->y = y;
    state->g = g;
    state->h = h;
    state->f = g + h;
    state->parent = parent_state;
#ifdef CUDA_DEBUG
    printf("x: %d, y: %d, g: %f, h: %f, f: %f\n", state->x, state->y, state->g, state->h, state->f);
#endif
    return state;
}

__device__ float fatomicMin(float *addr, float value)

{
    float old = *addr, assumed;
    if (old <= value)
    {
        return old;
    }

    do
    {
        assumed = old;
        old = atomicCAS((unsigned int *)addr, __float_as_int(assumed), __float_as_int(value));
    } while (old != assumed);

    return old;
}

__device__ inline double heuristic_cuda(int x0, int y0, int x1, int y1)
{
    return sqrt((double)((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1)));
}

__device__ inline int parent(int i)
{
    return (i - 1) / 2;
}

__device__ inline int left_child(int i)
{
    return (2 * i + 1);
}

__device__ inline int right_child(int i)
{
    return (2 * i + 2);
}

__device__ void check_if_min_heap(PriorityQueueList queues, int queue_idx)
{
    // check if it is valid min-heap
    PriorityQueue &queue = queues.queues[queue_idx];
    int is_min_queue = 1;
    for (int i = 0; i < queue.num_entries[0] && is_min_queue; i++)
    {
        int left = left_child(i);
        int right = right_child(i);
        if (left < queue.num_entries[0] && queue.entries[left]->f < queue.entries[i]->f)
        {
            is_min_queue = 0;
        }
        if (right < queue.num_entries[0] && queue.entries[right]->f < queue.entries[i]->f)
        {
            is_min_queue = 0;
        }
    }
    for (int i = 0; i < queue.num_entries[0]; i++)
    {
        printf("%f \n", queue.entries[i]->f);
    }
    if (is_min_queue == 0)
    {
        printf("Queue %d is not valid min-heap\n", queue_idx);
    }
    else
    {
        printf("Queue %d is valid min-heap\n", queue_idx);
    }
    assert(is_min_queue);
}

__device__ inline static void swap_state(State **s1, State **s2)
{
    State *tmp = *s1;
    *s1 = *s2;
    *s2 = tmp;
}

__device__ void expand_open_list(ThetaStarCudaDeviceData *data, State *parent, int offset_x, int offset_y)
{
    int neigh_x = parent->x + offset_x;
    int neigh_y = parent->y + offset_y;

    int neigh_idx = neigh_x + neigh_y * data->map_data.size;
    if (neigh_x < 0 || neigh_x >= data->map_data.size || neigh_y < 0 || neigh_y >= data->map_data.size)
    {
        return;
    }
    if (data->map_data.tiles[neigh_idx] == OBSTACLE)
    {
        return;
    }
    double g;
    if (parent->parent != nullptr && line_of_sight(parent->parent->x, parent->parent->y, neigh_x, neigh_y, data->map_data.tiles, data->map_data.size))
    {
        g = parent->parent->g + heuristic_cuda(parent->parent->x, parent->parent->y, neigh_x, neigh_y);
        if (data->closed_grid.assigned[neigh_idx] > 0 && data->closed_grid.data[neigh_idx]->g <= g)
        {
            return;
        }
        parent = parent->parent;
    }
    else
    {
        g = parent->g + heuristic_cuda(parent->x, parent->y, neigh_x, neigh_y);
        if (data->closed_grid.assigned[neigh_idx] > 0 && data->closed_grid.data[neigh_idx]->g <= g)
        {
            return;
        }
    }
    double h = heuristic_cuda(neigh_x, neigh_y, data->goal_x, data->goal_y);

    int new_idx = atomicAdd(data->expanded_list.num_states, 1);
    assert(new_idx < data->expanded_list.size);

    State *s = create_state(data, neigh_x, neigh_y, g, h, parent);
    data->expanded_list.states[new_idx] = s;
}

__device__ State *best_state = nullptr;

__device__ void heapify(PriorityQueue &open_queue, int current)
{
#ifdef CUDA_DEBUG
    printf("Heapify: %d\n", current);
#endif
    if (open_queue.num_entries[0] <= 1 || current < 0)
    {
        return;
    }

    while (current < open_queue.num_entries[0] && current >= 0)
    {
        int left = left_child(current);
        int right = right_child(current);
#ifdef CUDA_DEBUG
        printf("left: %d, right: %d, current: %d, num_entries: %d\n", left, right, current, open_queue.num_entries[0]);
#endif
        int smallest = current;
        if (left < open_queue.num_entries[0] && open_queue.entries[left]->f < open_queue.entries[current]->f)
        {
            smallest = left;
        }
        if (right < open_queue.num_entries[0] && open_queue.entries[right]->f < open_queue.entries[smallest]->f)
        {
            smallest = right;
        }
        if (smallest == current)
        {
            return;
        }
#ifdef CUDA_DEBUG
        printf("swap: %d, %d\n", current, smallest);
#endif
        swap_state(&(open_queue.entries[current]), &(open_queue.entries[smallest]));

        current = smallest;
    }
}

__global__ void extract_expand_open_list(ThetaStarCudaDeviceData *data)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int exec_id = tid; exec_id < NUM_QUEUES; exec_id += blockDim.x * gridDim.x)
    {
        PriorityQueue &open_queue = data->open_lists.queues[exec_id];
        if (open_queue.num_entries[0] == 0)
        {
#ifdef CUDA_DEBUG
            printf("Queue %d is empty\n", exec_id);
#endif
            continue;
        }
        State *min_state = open_queue.entries[0];
#ifdef CUDA_DEBUG
        for (int i = 0; i < open_queue.num_entries[0]; i++)
        {
            printf("Queue entries: %d: %f (%d %d) \n", exec_id, open_queue.entries[i]->f, open_queue.entries[i]->x, open_queue.entries[i]->y);
        }
        printf("extract x: %d, y: %d min_val: %f\n", min_state->x, min_state->y, min_state->f);
#endif
        // delete min
        atomicSub(data->total_open_list_size, 1);
        int last_idx = open_queue.num_entries[0] - 1;
        open_queue.entries[0] = open_queue.entries[last_idx];

        open_queue.num_entries[0]--;
        heapify(open_queue, 0);

        if (best_state != nullptr && min_state->f > best_state->f)
        {
            // it cannot get better => skip
            continue;
        }

#ifdef CUDA_DEBUG
        check_if_min_heap(data->open_lists, exec_id);
#endif

        if (min_state->x == data->goal_x && min_state->y == data->goal_y)
        {
            if (best_state == nullptr || min_state->f < best_state->f)
            {
                best_state = min_state;
            }
            atomicAdd(data->is_at_goal, 1);
            continue;
        }

        expand_open_list(data, min_state, -1, -1);
        expand_open_list(data, min_state, 0, -1);
        expand_open_list(data, min_state, 1, -1);
        expand_open_list(data, min_state, -1, 0);
        expand_open_list(data, min_state, 1, 0);
        expand_open_list(data, min_state, -1, 1);
        expand_open_list(data, min_state, 0, 1);
        expand_open_list(data, min_state, 1, 1);
    }
}

__global__ void reset_duplicate_grid(ThetaStarCudaDeviceData *data)
{
    auto x = (blockIdx.y * blockDim.y + threadIdx.y);
    auto y = (blockIdx.x * blockDim.x + threadIdx.x);
    int idx = x + y * data->duplicate_grid.size;

    if (idx >= data->duplicate_grid.size * data->duplicate_grid.size)
    {
        return;
    }
    data->duplicate_grid.min_f[idx] = 999999;
}

__global__ void filter_duplicates(ThetaStarCudaDeviceData *data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    State *state = data->expanded_list.states[idx];
    int state_idx = state->x + state->y * data->closed_grid.size;
#ifdef CUDA_DEBUG
    printf("Filter duplicate: (%d, %d), Assigned: %d, %f < %f\n", state->x, state->y, data->closed_grid.assigned[state_idx], data->closed_grid.assigned[state_idx] > 0 ? data->closed_grid.data[state_idx]->g : -42, state->g);
#endif
    if (data->closed_grid.assigned[state_idx] > 0 && data->closed_grid.data[state_idx]->g <= state->g)
    {
#ifdef CUDA_DEBUG
        printf("Set 0 (%d, %d)\n", state->x, state->y);
#endif
        data->duplicate_criteria[idx] = 0;
    }
    else
    {
#ifdef CUDA_DEBUG
        printf("Preliminary 1 (%d, %d)\n", state->x, state->y);
#endif
        fatomicMin(&data->duplicate_grid.min_f[state_idx], state->f);
        data->duplicate_criteria[idx] = 1;
    }
}

__global__ void filter_duplicates_grid(ThetaStarCudaDeviceData *data)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    State *state = data->expanded_list.states[idx];
    int state_idx = state->x + state->y * data->duplicate_grid.size;
    if (data->duplicate_criteria[idx] == 1)
    {
        if ((float)data->duplicate_grid.min_f[state_idx] < (float)state->f)
        {
#ifdef CUDA_DEBUG
            printf("Revert 1 => 0 (%d, %d)\n", state->x, state->y);
#endif
            data->duplicate_criteria[idx] = 0;
        }
        else
        {

#ifdef CUDA_DEBUG
            printf("Set 1 (%d, %d)\n", state->x, state->y);
#endif
            atomicAdd(data->duplicate_size, 1);
        }
    }
}

__global__ void scan_state_list(StateList new_expanded_list, StateList old_expanded_list, int *duplicate_prefix_sum, int *duplicate_criteria)
{
    assert(new_expanded_list.size == old_expanded_list.size);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= old_expanded_list.num_states[0])
    {
        return;
    }
    if (duplicate_criteria[idx] == 1)
    {
        new_expanded_list.states[duplicate_prefix_sum[idx]] = old_expanded_list.states[idx];
    }
}

__global__ void distribute_state_list(ThetaStarCudaDeviceData *data, int offset)
{
    int num_states = data->new_expanded_list.num_states[0];
    for (int i = threadIdx.x; i < num_states; i += blockDim.x)
    {
#ifdef CUDA_DEBUG
        printf("Distribute state: %d threadIdx: %d, blockDim: %d\n", i, threadIdx.x, blockDim.x);
#endif
        // add to closed grid
        if (i >= num_states)
        {
            continue;
        }

        State *s = data->new_expanded_list.states[i];

        int queue_idx = (i + offset) % NUM_QUEUES;
#ifdef CUDA_DEBUG
        printf("Add to closed grid x: %d, y: %d %f to queue %d\n", s->x, s->y, s->f, queue_idx);

        check_if_min_heap(data->open_lists, queue_idx);
#endif
        PriorityQueue &queue = data->open_lists.queues[queue_idx];

        int is_duplicate = 0;
        for (int entry_idx = 0; entry_idx < queue.num_entries[0] && !is_duplicate; entry_idx++)
        {
            State *other_state = queue.entries[entry_idx];
            if (other_state->x == s->x && other_state->y == s->y)
            {
                is_duplicate = 1;
            }
        }
        if (!is_duplicate)
        {
            queue.num_entries[0]++;
            assert(queue.num_entries[0] < queue.size);
            int entry_idx = queue.num_entries[0] - 1;

            queue.entries[entry_idx] = s;
            data->closed_grid.data[s->x + s->y * data->closed_grid.size] = s;
            data->closed_grid.assigned[s->x + s->y * data->closed_grid.size] = 1;

            for (int k = queue.num_entries[0] / 2 - 1; k >= 0; k--)
            {
                heapify(queue, k);
            }

#ifdef CUDA_DEBUG
            check_if_min_heap(data->open_lists, queue_idx);
#endif

            atomicAdd(data->total_open_list_size, 1);
        }
        __syncthreads();
    }

#ifdef CUDA_DEBUG
    if (threadIdx.x == 0)
    {
        int total_count = 0;
        for (int i = 0; i < NUM_QUEUES; i++)
        {
            total_count += data->open_lists.queues[i].num_entries[0];
        }
        printf("Total count: %d\n", total_count);
    }
#endif
}

typedef struct ThetaStarCudaData
{
    Map *map;
    MapData map_data;
    MapData d_map_data;

    StatePool state_pool;
    StatePool d_state_pool;

    PriorityQueueList open_lists;
    PriorityQueueList d_open_lists;

    StateList expanded_list;
    StateList new_expanded_list;
    StateList d_expanded_list;
    StateList d_new_expanded_list;

    ClosedGrid closed_grid;
    ClosedGrid d_closed_grid;

    DuplicateGrid duplicate_grid;
    DuplicateGrid d_duplicate_grid;

    int *d_duplicate_criteria;
    int *d_duplicate_prefix_sum;

    int *d_is_at_goal;
    int *is_at_goal;

    int *total_open_list_size;
    int *d_total_open_list_size;

    int *duplicate_size;
    int *d_duplicate_size;

    int *d_path;
    int *path;

    int *d_path_length;
    int *path_length;

    int *num_states;
} ThetaStarCudaData;

void reset_theta_star_cuda(ThetaStarCudaData *data)
{
    for (int idx = 0; idx < NUM_QUEUES; idx++)
    {
        data->open_lists.queues[idx].num_entries[0] = 0;
        queue_to_device(data->open_lists.queues[idx], data->d_open_lists.queues[idx]);
    }
    data->expanded_list.num_states[0] = 0;
    data->new_expanded_list.num_states[0] = 0;
    state_list_to_device(data->expanded_list, data->d_expanded_list);
    state_list_to_device(data->new_expanded_list, data->d_new_expanded_list);

    reset_closed_grid(data->closed_grid);
    closed_grid_to_device(data->closed_grid, data->d_closed_grid);

    reset_duplicate_grid(data->duplicate_grid);
    duplicate_grid_to_device(data->duplicate_grid, data->d_duplicate_grid);

    data->duplicate_grid = create_duplicate_grid(data->map->size);

    cudaMemset(data->d_duplicate_prefix_sum, 0, sizeof(int) * OPEN_LIST_SIZE);
    gpuErrchk();

    cudaMemset(data->d_duplicate_criteria, 0, sizeof(int) * OPEN_LIST_SIZE);
    gpuErrchk();

#ifndef USE_ZC_MEMORY
    cudaMemset(data->d_is_at_goal, 0, sizeof(int));
    gpuErrchk();

    cudaMemset(data->d_total_open_list_size, 0, sizeof(int));
    gpuErrchk();

    cudaMemset(data->d_duplicate_size, 0, sizeof(int));
    gpuErrchk();

    cudaMemset(data->d_path, 0, sizeof(int) * data->map->size * data->map->size * 2);
    gpuErrchk();

    cudaMemset(data->d_path_length, 0, sizeof(int));
    gpuErrchk();
#endif

    data->is_at_goal[0] = 0;
    data->total_open_list_size[0] = 0;
    data->duplicate_size[0] = 0;
    memset(data->path, 0, sizeof(int) * data->map->size * data->map->size * 2);
    data->path_length[0] = 0;
    data->num_states[0] = 0;

    data->state_pool.num_states[0] = 0;
    state_pool_to_device(data->state_pool, data->d_state_pool);
}

void init_theta_star_cuda(ThetaStarCudaData *data, Map *map)
{
    data->map = map;

    for (int idx = 0; idx < NUM_QUEUES; idx++)
    {
        data->open_lists.queues[idx] = create_queue();
        data->d_open_lists.queues[idx] = create_queue_device(data->open_lists.queues[idx]);
        queue_to_device(data->open_lists.queues[idx], data->d_open_lists.queues[idx]);
    }
    data->expanded_list = create_state_list(LIST_CAPACITY);
    data->new_expanded_list = create_state_list(LIST_CAPACITY);
    data->d_expanded_list = create_state_list_device(data->expanded_list);
    data->d_new_expanded_list = create_state_list_device(data->new_expanded_list);

    data->map_data = create_map_data(map);
    data->d_map_data = create_map_data_device(data->map_data);
    map_data_to_device(data->map_data, data->d_map_data);

    data->closed_grid = create_closed_grid(data->map->size);
    data->d_closed_grid = create_closed_grid_device(data->closed_grid);

    data->duplicate_grid = create_duplicate_grid(data->map->size);
    data->d_duplicate_grid = create_duplicate_grid_device(data->duplicate_grid);

    cudaMalloc(&data->d_duplicate_criteria, sizeof(int) * OPEN_LIST_SIZE);
    gpuErrchk();

    cudaMalloc(&data->d_duplicate_prefix_sum, sizeof(int) * OPEN_LIST_SIZE);
    gpuErrchk();

#ifdef USE_PINNED_MEMORY
    cudaMallocHost(&data->is_at_goal, sizeof(int));
#elif defined(USE_ZC_MEMORY)
    cudaHostAlloc(&data->is_at_goal, sizeof(int), cudaHostAllocMapped);
#else
    data->is_at_goal = (int *)malloc(sizeof(int));
#endif
    gpuErrchk();

#ifdef USE_ZC_MEMORY
    cudaHostGetDevicePointer(&data->d_is_at_goal, data->is_at_goal, 0);
#else
    cudaMalloc(&data->d_is_at_goal, sizeof(int));
#endif
    gpuErrchk();

#ifdef USE_PINNED_MEMORY
    cudaMallocHost(&data->total_open_list_size, sizeof(int));
#elif defined(USE_ZC_MEMORY)
    cudaHostAlloc(&data->total_open_list_size, sizeof(int), cudaHostAllocMapped);
#else
    data->total_open_list_size = (int *)malloc(sizeof(int));
#endif
    gpuErrchk();
#ifndef USE_ZC_MEMORY
    cudaMalloc(&data->d_total_open_list_size, sizeof(int));
#else
    cudaHostGetDevicePointer(&data->d_total_open_list_size, data->total_open_list_size, 0);
#endif
    gpuErrchk();

#ifdef USE_PINNED_MEMORY
    cudaMallocHost(&data->duplicate_size, sizeof(int));
#elif defined(USE_ZC_MEMORY)
    cudaHostAlloc(&data->duplicate_size, sizeof(int), cudaHostAllocMapped);
#else
    data->duplicate_size = (int *)malloc(sizeof(int));
#endif
    gpuErrchk();
#ifndef USE_ZC_MEMORY
    cudaMalloc(&data->d_duplicate_size, sizeof(int));
#else
    cudaHostGetDevicePointer(&data->d_duplicate_size, data->duplicate_size, 0);
#endif
    gpuErrchk();

    data->state_pool = create_state_pool();
    data->d_state_pool = create_state_pool_device(data->state_pool);
    state_pool_to_device(data->state_pool, data->d_state_pool);

#ifdef USE_PINNED_MEMORY
    cudaMallocHost(&data->path, sizeof(int) * data->map->size * data->map->size * 2);
#elif defined(USE_ZC_MEMORY)
    cudaHostAlloc(&data->path, sizeof(int) * data->map->size * data->map->size * 2, cudaHostAllocMapped);
#else
    data->path = (int *)malloc(sizeof(int) * data->map->size * data->map->size * 2);
#endif
    gpuErrchk();

#ifndef USE_ZC_MEMORY
    cudaMalloc(&data->d_path, sizeof(int) * data->map->size * data->map->size * 2);
#else
    cudaHostGetDevicePointer(&data->d_path, data->path, 0);
#endif
    gpuErrchk();

#ifdef USE_PINNED_MEMORY
    cudaMallocHost(&data->path_length, sizeof(int));
#elif defined(USE_ZC_MEMORY)
    cudaHostAlloc(&data->path_length, sizeof(int), cudaHostAllocMapped);
#else
    data->path_length = (int *)malloc(sizeof(int));
#endif
    gpuErrchk();
#ifndef USE_ZC_MEMORY
    cudaMalloc(&data->d_path_length, sizeof(int));
#elif defined(USE_ZC_MEMORY)
    cudaHostGetDevicePointer(&data->d_path_length, data->path_length, 0);
#else
    gpuErrchk();
#endif

#ifdef USE_PINNED_MEMORY
    cudaMallocHost(&data->num_states, sizeof(int));
    gpuErrchk();
#else
    data->num_states = (int *)malloc(sizeof(int));
#endif

    reset_theta_star_cuda(data);
}

__global__ void init_kernel(ThetaStarCudaDeviceData *data)
{
    State *initial_state = create_state(data, data->start_x, data->start_y, 0, heuristic_cuda(data->start_x, data->start_y, data->goal_x, data->goal_y), nullptr);

    data->open_lists.queues[0].entries[0] = initial_state;
    data->open_lists.queues[0].num_entries[0] = 1;

    int start_idx = data->start_x + data->start_y * data->closed_grid.size;
    data->closed_grid.data[start_idx] = data->open_lists.queues[0].entries[0];
    data->closed_grid.assigned[start_idx] = 1;

    best_state = nullptr;
}

#ifdef CUDA_DEBUG
__global__ void debug_duplicate_list(int num_states, int *not_duplicate_list, StateList expanded_list, int *duplicate_prefix_sum)
{
    for (int i = 0; i < num_states; i++)
    {
        const char *duplicate = not_duplicate_list[i] == 1 ? "Not duplicate" : "Duplicate";

        printf("%s x: %d, y: %d: %f (cur val %d) \n", duplicate, expanded_list.states[i]->x, expanded_list.states[i]->y, expanded_list.states[i]->f, duplicate_prefix_sum[i]);
    }
}

__global__ void debug_scanned_list(StateList expanded_list)
{
    for (int i = 0; i < expanded_list.num_states[0]; i++)
    {
        printf("scanned states x: %d, y: %d: %f\n", expanded_list.states[i]->x, expanded_list.states[i]->y, expanded_list.states[i]->f);
    }
}

__global__ void debug_final_open_list(ThetaStarCudaDeviceData *data)
{
    for (int queue_idx = 0; queue_idx < NUM_QUEUES; queue_idx++)
    {
        PriorityQueue &queue = data->open_lists.queues[queue_idx];
        for (int i = 0; i < queue.num_entries[0]; i++)
        {
            printf("distributed x: %d, y: %d: %f\n", queue.entries[i]->x, queue.entries[i]->y, queue.entries[i]->f);
        }
    }
}

__global__ void validate_priority_queue(ThetaStarCudaDeviceData *data)
{
    for (int queue_idx = 0; queue_idx < NUM_QUEUES; queue_idx++)
    {
        printf("Validate priority queue %d\n", queue_idx);
        PriorityQueue &queue = data->open_lists.queues[queue_idx];
        for (int i = 0; i < queue.num_entries[0]; i++)
        {
            printf("validation x: %d, y: %d: %f\n", queue.entries[i]->x, queue.entries[i]->y, queue.entries[i]->f);
        }

        for (int i = 0; i < queue.num_entries[0]; i++)
        {
            for (int j = 0; j < queue.num_entries[0]; j++)
            {
                if (i != j)
                {
                    assert(queue.entries[i]->x != queue.entries[j]->x || queue.entries[i]->y != queue.entries[j]->y);
                }
            }
        }
    }
}
#endif

__global__ void generate_path(ThetaStarCudaDeviceData *data)
{
    if (best_state == nullptr)
    {
        return;
    }
    double min_queue_f = 9999999;
    for (int queue_idx = 0; queue_idx < NUM_QUEUES; queue_idx++)
    {
        PriorityQueue &queue = data->open_lists.queues[queue_idx];
        double queue_f = queue.num_entries[0] > 0 ? queue.entries[0]->f : 9999999;
        min_queue_f = min(min_queue_f, queue_f);
    }
    if (best_state->f > min_queue_f)
    {
        return;
    }
    State *state = best_state;
    data->path_length[0] = 0;
    while (state != nullptr)
    {

        data->path[data->path_length[0] * 2] = state->x;
        data->path[data->path_length[0] * 2 + 1] = state->y;

        state = state->parent;
        data->path_length[0]++;
    }
}

std::vector<std::pair<int, int>> theta_star_cuda(int start_x, int start_y, int goal_x, int goal_y, ThetaStarCudaData &cudaData)
{
    std::vector<std::pair<int, int>> path;
    // avoid Error: Formal parameter space overflowed error
    ThetaStarCudaDeviceData deviceData;
    deviceData.map_data = cudaData.d_map_data;
    deviceData.state_pool = cudaData.d_state_pool;
    deviceData.open_lists = cudaData.d_open_lists;
    deviceData.expanded_list = cudaData.d_expanded_list;
    deviceData.new_expanded_list = cudaData.d_new_expanded_list;
    deviceData.closed_grid = cudaData.d_closed_grid;
    deviceData.duplicate_criteria = cudaData.d_duplicate_criteria;
    deviceData.duplicate_prefix_sum = cudaData.d_duplicate_prefix_sum;
    deviceData.is_at_goal = cudaData.d_is_at_goal;
    deviceData.total_open_list_size = cudaData.d_total_open_list_size;
    deviceData.duplicate_size = cudaData.d_duplicate_size;
    deviceData.path = cudaData.d_path;
    deviceData.path_length = cudaData.d_path_length;
    deviceData.start_x = start_x;
    deviceData.start_y = start_y;
    deviceData.goal_x = goal_x;
    deviceData.goal_y = goal_y;
    deviceData.duplicate_grid = cudaData.d_duplicate_grid;

    ThetaStarCudaDeviceData *d_deviceData;
    cudaMalloc(&d_deviceData, sizeof(ThetaStarCudaDeviceData));
    gpuErrchk();
    cudaMemcpy(d_deviceData, &deviceData, sizeof(ThetaStarCudaDeviceData), cudaMemcpyDefault);
    gpuErrchk();

    // push initial starting node
    init_kernel<<<1, 1>>>(d_deviceData);
    gpuErrchk();

    cudaDeviceSynchronize();
    gpuErrchk();

#ifdef CUDA_DEBUG
    printf("start_x: %d, start_y: %d\n", start_x, start_y);
#endif

    cudaData.total_open_list_size[0] = 1;
#ifndef USE_ZC_MEMORY
    cudaMemcpy(cudaData.d_total_open_list_size, cudaData.total_open_list_size, sizeof(int), cudaMemcpyDefault);
    gpuErrchk();
#endif

    int iteration = 0;
    while (cudaData.total_open_list_size[0] > 0)
    {
        iteration++;
        if (iteration > 1)
        {
            // exit(1);
        }
        reset_duplicate_grid<<<dim3(cudaData.duplicate_grid.size / 8 + 1, cudaData.duplicate_grid.size / 8 + 1), dim3(8, 8)>>>(d_deviceData);
        gpuErrchk();

#ifdef CUDA_DEBUG
        cudaDeviceSynchronize();
        printf("iteration: %d %d\n", iteration, cudaData.total_open_list_size[0]);
#endif
        cudaData.expanded_list.num_states[0] = 0;
#ifndef USE_ZC_MEMORY
        cudaMemset(cudaData.d_expanded_list.num_states, 0, sizeof(int));
        gpuErrchk();
#endif
#ifdef CUDA_DEBUG
        validate_priority_queue<<<1, 1>>>(d_deviceData);
        cudaDeviceSynchronize();
        gpuErrchk();
#endif
        extract_expand_open_list<<<QUEUE_BLOCKS, THREADS_PER_BLOCK>>>(d_deviceData);
        cudaDeviceSynchronize();
        gpuErrchk();
#ifndef USE_ZC_MEMORY
        cudaMemcpy(cudaData.is_at_goal, cudaData.d_is_at_goal, sizeof(int), cudaMemcpyDefault);
        gpuErrchk();
#endif
#ifdef CUDA_DEBUG
        validate_priority_queue<<<1, 1>>>(d_deviceData);
        cudaDeviceSynchronize();
        gpuErrchk();
#endif
        if (cudaData.is_at_goal[0] > 0)
        {
            generate_path<<<1, 1>>>(d_deviceData);
#ifdef CUDA_DEBUG
            cudaDeviceSynchronize();
            gpuErrchk();
#endif
        }
#ifndef USE_ZC_MEMORY
        cudaMemcpy(cudaData.num_states, cudaData.d_expanded_list.num_states, sizeof(int), cudaMemcpyDefault);
        int num_states = cudaData.num_states[0];
#else
        int num_states = cudaData.d_expanded_list.num_states[0];
#endif

#ifdef CUDA_DEBUG
        printf("num states before filter %d\n", num_states);
#endif
        gpuErrchk();
        if (num_states > 0)
        {
#ifdef CUDA_DEBUG
            printf("num_states (before filter_duplicates): %d\n", num_states);
#endif
            cudaData.duplicate_size[0] = 0;
#ifndef USE_ZC_MEMORY
            cudaMemcpy(cudaData.d_duplicate_size, cudaData.duplicate_size, sizeof(int), cudaMemcpyDefault);
            gpuErrchk();
#endif
            filter_duplicates<<<num_states, 1>>>(d_deviceData);
            gpuErrchk();
            cudaDeviceSynchronize();
            gpuErrchk();
            filter_duplicates_grid<<<num_states, 1>>>(d_deviceData);
            gpuErrchk();
            cudaDeviceSynchronize();
            gpuErrchk();
#ifndef USE_ZC_MEMORY
            cudaMemcpy(cudaData.duplicate_size, cudaData.d_duplicate_size, sizeof(int), cudaMemcpyDefault);
            gpuErrchk();
#endif
#ifdef CUDA_DEBUG
            {
                debug_duplicate_list<<<1, 1>>>(num_states, cudaData.d_duplicate_criteria, cudaData.d_expanded_list, cudaData.d_duplicate_prefix_sum);
                cudaDeviceSynchronize();
                gpuErrchk();
            }
#endif
#ifdef CUDA_DEBUG
            printf("num_states (befor prefix sum): %d\n", num_states);
#endif
            prefixSum(cudaData.d_duplicate_prefix_sum, cudaData.d_duplicate_criteria, num_states);
            gpuErrchk();
            cudaDeviceSynchronize();
            gpuErrchk();
#ifdef CUDA_DEBUG
            {
                printf("num_states (after prefix sum): %d\n", num_states);
                int *prefix_summed = (int *)malloc(sizeof(int) * num_states);
                cudaMemcpy(prefix_summed, cudaData.d_duplicate_prefix_sum, sizeof(int) * num_states, cudaMemcpyDefault);
                gpuErrchk();
                for (int i = 0; i < num_states; i++)
                {
                    printf("prefix_summed[%d]: %d\n", i, prefix_summed[i]);
                }
                free(prefix_summed);
            }
#endif
#ifndef USE_ZC_MEMORY
            cudaMemcpy(cudaData.d_new_expanded_list.num_states, cudaData.d_duplicate_size, sizeof(int), cudaMemcpyDefault);
            gpuErrchk();
#else
            cudaData.new_expanded_list.num_states[0] = cudaData.duplicate_size[0];
#endif
            scan_state_list<<<num_states, 1>>>(cudaData.d_new_expanded_list, cudaData.d_expanded_list, cudaData.d_duplicate_prefix_sum, cudaData.d_duplicate_criteria);
            gpuErrchk();
            cudaDeviceSynchronize();
            gpuErrchk();
#ifdef CUDA_DEBUG
            {
                debug_scanned_list<<<1, 1>>>(cudaData.d_new_expanded_list);
                cudaDeviceSynchronize();
                gpuErrchk();
            }
#endif
#ifdef CUDA_DEBUG
            validate_priority_queue<<<1, 1>>>(d_deviceData);
            cudaDeviceSynchronize();
            gpuErrchk();
#endif
            distribute_state_list<<<1, THREADS_PER_BLOCK>>>(d_deviceData, iteration);
            gpuErrchk();
            cudaDeviceSynchronize();
            gpuErrchk();
#ifdef CUDA_DEBUG
            validate_priority_queue<<<1, 1>>>(d_deviceData);
            cudaDeviceSynchronize();
            gpuErrchk();
            {
                debug_final_open_list<<<1, 1>>>(d_deviceData);
                cudaDeviceSynchronize();
                gpuErrchk();
            }
#endif
        }
#ifndef USE_ZC_MEMORY
        cudaMemcpy(cudaData.total_open_list_size, cudaData.d_total_open_list_size, sizeof(int), cudaMemcpyDefault);
        gpuErrchk();
#endif
        if (cudaData.is_at_goal[0] > 0)
        {
#ifndef USE_ZC_MEMORY
            cudaMemcpy(cudaData.path_length, cudaData.d_path_length, sizeof(int), cudaMemcpyDefault);
            gpuErrchk();
#endif
            if (cudaData.path_length[0] > 0)
            {
#ifdef CUDA_DEBUG
                printf("Found path\n");
#endif
                for (int i = 0; i < cudaData.path_length[0]; i++)
                {
                    int x = 0;
                    int y = 0;
#ifndef USE_ZC_MEMORY
                    cudaMemcpy(&x, cudaData.d_path + i * 2, sizeof(int), cudaMemcpyDefault);
                    gpuErrchk();
                    cudaMemcpy(&y, cudaData.d_path + i * 2 + 1, sizeof(int), cudaMemcpyDefault);
                    gpuErrchk();
#else
                    x = cudaData.path[i * 2];
                    y = cudaData.path[i * 2 + 1];
#endif
                    path.push_back({x, y});
#ifdef CUDA_DEBUG
                    printf("x: %d, y: %d\n", x, y);
#endif
                }
                break;
            }
        }
#ifdef CUDA_DEBUG
        printf("total_open_list_size: %d\n", cudaData.total_open_list_size[0]);
#endif
    }
    cudaFree(d_deviceData);

    return path;
}

float get_obstacle_ratio(Map *map)
{
    // count obstacles
    int num_obstacles = 0;
    for (int i = 0; i < map->size; i++)
    {
        for (int j = 0; j < map->size; j++)
        {
            if (get_tile(map, i, j) == OBSTACLE)
            {
                num_obstacles++;
            }
        }
    }
    return num_obstacles / ((float)map->size * map->size);
}

int main(int argc, char *argv[])
{
#ifdef USE_ZC_MEMORY
    cudaSetDeviceFlags(cudaDeviceMapHost);
#endif
    printf("Execute with Grid Size: %d, number of queues %d, number of experiments %d\n", GRID_SIZE, NUM_QUEUES, EXPERIMENT_REPETITION);
    Map *map = generate_map(GRID_SIZE, (int)(GRID_SIZE * GRID_SIZE * MAP_SPARSITY), MAP_SEED);
    ThetaStarCudaData cudaData;

    int start_x = 1, start_y = 1;
    int goal_x = GRID_SIZE - 2, goal_y = GRID_SIZE - 2;

#ifdef RUN_PARALLEL
    init_theta_star_cuda(&cudaData, map);
    std::vector<std::chrono::duration<int64_t, std::nano>> parallel_times(EXPERIMENT_REPETITION);
    std::vector<std::pair<int, int>> parallel_path;

    for (int repeat = 0; repeat < EXPERIMENT_REPETITION; repeat++)
    {
        printf("Parallel Run %d\n", repeat);
        auto start = std::chrono::high_resolution_clock::now();
        parallel_path = theta_star_cuda(start_x, start_y, goal_x, goal_y, cudaData);
        parallel_times[repeat] = std::chrono::high_resolution_clock::now() - start;
        reset_theta_star_cuda(&cudaData);
        if (parallel_path.size() == 0)
        {
            printf("Did not find path to goal\n");
        }
    }

    double parallel_distance = print_map(map, parallel_path);
#endif
#ifdef RUN_SEQUENTIAL
    std::vector<std::chrono::duration<int64_t, std::nano>> sequenial_times(EXPERIMENT_REPETITION);
    std::vector<std::pair<int, int>> sequential_path;

    for (int repeat = 0; repeat < EXPERIMENT_REPETITION; repeat++)
    {
        printf("Sequential Run %d\n", repeat);
        auto start = std::chrono::high_resolution_clock::now();
        sequential_path = theta_star(map, start_x, start_y, goal_x, goal_y);
        sequenial_times[repeat] = std::chrono::high_resolution_clock::now() - start;

        if (sequential_path.size() == 0)
        {
            printf("Did not find path to goal\n");
        }
    }

    double sequential_distance = print_map(map, sequential_path);
#endif

#ifdef RUN_SEQUENTIAL
    std::sort(sequenial_times.begin(), sequenial_times.end());
    double mean_sequential_ms = std::chrono::duration_cast<std::chrono::milliseconds>(sequenial_times[sequenial_times.size() / 2]).count();

    double sequential_standard_dev = 0;
    for (int repeat = 0; repeat < EXPERIMENT_REPETITION; repeat++)
    {
        sequential_standard_dev += abs(std::chrono::duration_cast<std::chrono::milliseconds>(sequenial_times[repeat]).count() - mean_sequential_ms);
    }
    sequential_standard_dev = sequential_standard_dev / EXPERIMENT_REPETITION;
#endif
#ifdef RUN_PARALLEL
    std::sort(parallel_times.begin(), parallel_times.end());

    double mean_parallel_ms = std::chrono::duration_cast<std::chrono::milliseconds>(parallel_times[parallel_times.size() / 2]).count();

    double parallel_standard_dev = 0;
    for (int repeat = 0; repeat < EXPERIMENT_REPETITION; repeat++)
    {
        parallel_standard_dev += abs(std::chrono::duration_cast<std::chrono::milliseconds>(parallel_times[repeat]).count() - mean_parallel_ms);
    }
    parallel_standard_dev = parallel_standard_dev / EXPERIMENT_REPETITION;
#endif
#ifdef RUN_SEQUENTIAL
    printf("Sequential time: %f ms, (+- %f ms)\n", mean_sequential_ms, sequential_standard_dev);
#endif
#ifdef RUN_PARALLEL
    printf("Parallel time: %f ms (+- %f ms)\n", mean_parallel_ms, parallel_standard_dev);
#endif
#if defined(RUN_PARALLEL) && defined(RUN_SEQUENTIAL)
    printf("Distance delta (sequential [%f] - parallel [%f]): %f\n", sequential_distance, parallel_distance, sequential_distance - parallel_distance);
#endif
    printf("Obstacle ratio: %f %%\n", get_obstacle_ratio(map) * 100);
    std::ofstream outfile;

#if !defined(RUN_PARALLEL) != !defined(RUN_SEQUENTIAL)
#ifdef RUN_PARALLEL
    std::string experiment_type = "parallel";
    double duration = mean_parallel_ms;
    double stdev = parallel_standard_dev;
    double distance = parallel_distance;
#else
    std::string experiment_type = "sequential";
    double duration = mean_sequential_ms;
    double stdev = sequential_standard_dev;
    double distance = sequential_distance;
#endif
#ifdef USE_ZC_MEMORY
    std::string memory_config = "zc";
#elif defined(USE_PINNED_MEMORY)
    std::string memory_config = "pinned";
#else
    std::string memory_config = "default";
#endif

    outfile.open("result.txt", std::ios_base::app); // append instead of overwrite
    outfile << EXPERIMENT_NAME << ";"               //
            << experiment_type << ";"               //
            << duration << ";"                      //
            << stdev << ";"                         //
            << get_obstacle_ratio(map) << ";"       //
            << distance << ";"                      //
            << GRID_SIZE << ";"                     //
            << NUM_QUEUES << ";"                    //
            << MAP_SEED << ";"                      //
            << memory_config << ";"                 //
            << EXPERIMENT_REPETITION << ";"         //
            << std::endl;
    return 0;
#endif
}