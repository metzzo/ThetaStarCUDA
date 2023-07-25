#include <vector>

#define OBSTACLE 1
#define FREE 0

typedef struct Node
{
    int x, y;
    double g, h;
    int parent_x, parent_y;
} Node;

typedef struct Map
{
    int size;
    int *data;
} Map;

Map *create_map(int size);
void set_tile(Map *map, int x, int y, int value);
int get_tile(Map *map, int x, int y);
double print_map(Map *map, std::vector<std::pair<int, int>> path = {});
void free_map(Map *map);
Map *generate_map(int size, int num_rectangles, time_t seed);

double heuristic(int x0, int y0, int x1, int y1);
__device__ __host__ int line_of_sight(int x0, int y0, int x1, int y1, int *map, int size);
int update_vertex(Node *current_node, Node *neighbor, Map *map);

std::vector<std::pair<int, int>> theta_star(Map *map, int start_x, int start_y, int end_x, int end_y);