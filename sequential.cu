#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <algorithm>
#include "sequential.h"

#include <boost/container/set.hpp>
#include <boost/container/allocator.hpp>

// #define DEBUG_THETA

double heuristic(int x0, int y0, int x1, int y1)
{
    return sqrt((x0 - x1) * (x0 - x1) + (y0 - y1) * (y0 - y1));
}

Map *create_map(int size)
{
    Map *map = (Map *)malloc(sizeof(Map));
    map->size = size;
    map->data = (int *)malloc(sizeof(int) * size * size);
    for (int i = 0; i < size * size; i++)
    {
        map->data[i] = FREE;
    }
    return map;
}
void set_tile(Map *map, int x, int y, int value)
{
    if (x >= 0 && x < map->size && y >= 0 && y < map->size)
    {
        map->data[x + y * map->size] = value;
    }
}

int get_tile(Map *map, int x, int y)
{
    if (x >= 0 && x < map->size && y >= 0 && y < map->size)
    {
        return map->data[x + y * map->size];
    }
    else
    {
        return OBSTACLE;
    }
}

double print_map(Map *map, std::vector<std::pair<int, int>> path)
{
#ifdef PATH_EXPANSION
    std::vector<std::pair<int, int>> expanded_path;
    for (int i = 1; i < path.size(); i++)
    {
        int x0 = path[i - 1].first, y0 = path[i - 1].second;
        int x1 = path[i].first, y1 = path[i].second;

        int dx = abs(x1 - x0);
        int dy = -abs(y1 - y0);
        int sx = (x0 < x1) ? 1 : -1;
        int sy = (y0 < y1) ? 1 : -1;
        int error = dx + dy;

        while (1)
        {
            // printf("x: %d y: %d\n", x0, y0);
            expanded_path.push_back(std::make_pair(x0, y0));

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
    }
#else
    std::vector<std::pair<int, int>> expanded_path = path;
#endif

    double distance = 0;
    for (int i = 1; i < expanded_path.size(); i++)
    {
        int x0 = expanded_path[i - 1].first, y0 = expanded_path[i - 1].second;
        int x1 = expanded_path[i].first, y1 = expanded_path[i].second;
        distance += heuristic(x0, y0, x1, y1);
    }

    if (map->size < 128)
    {
        for (int y = 0; y < map->size; y++)
        {
            for (int x = 0; x < map->size; x++)
            {
                bool in_path = std::find(expanded_path.begin(), expanded_path.end(), std::make_pair(x, y)) != expanded_path.end();
                bool tile = get_tile(map, x, y);
                if (in_path && tile == OBSTACLE)
                {
                    printf("F ");
                }
                else if (in_path)
                {
                    printf("P ");
                }
                else if (tile == OBSTACLE)
                {
                    printf("X ");
                }
                else
                {
                    printf("  ");
                }
            }
            printf("\n");
        }
    }
    return distance;
}

void free_map(Map *map)
{
    free(map->data);
    free(map);
}

Map *generate_map(int size, int num_rectangles, time_t seed)
{
    // srand(size * size + size * num_rectangles);
    printf("seed: %ld\n", seed);
    srand(seed);

    Map *map = create_map(size);
    for (int i = 0; i < size; i++)
    {
        set_tile(map, i, 0, OBSTACLE);
        set_tile(map, i, size - 1, OBSTACLE);
        set_tile(map, 0, i, OBSTACLE);
        set_tile(map, size - 1, i, OBSTACLE);
    }

    // place random rectangles
    for (int i = 0; i < num_rectangles; i++)
    {
        int x = rand() % (size - 11) + 1;
        int y = rand() % (size - 11) + 1;
        int w = rand() % (10) + 1;
        int h = rand() % (10) + 1;
        for (int j = x; j <= x + w; j++)
        {
            for (int k = y; k <= y + h; k++)
            {
                set_tile(map, j, k, OBSTACLE);
            }
        }
    }

    return map;
}

std::vector<std::pair<int, int>> theta_star(Map *map, int start_x, int start_y, int end_x, int end_y)
{
    // Initialization
    auto CostCmp = [map](Node a, Node b)
    {
        int a_cost = a.g + a.h;
        int b_cost = b.g + b.h;
        if (a_cost == b_cost)
        {
            int a_index = a.x + a.y * map->size;
            int b_index = b.x + b.y * map->size;
            return a_index < b_index;
        }
        else
        {
            return a_cost < b_cost;
        }
    };
    boost::container::set<Node, decltype(CostCmp)> open_list(CostCmp);
    auto PosCmp = [map](Node a, Node b)
    {
        int a_index = a.x + a.y * map->size;
        int b_index = b.x + b.y * map->size;
        return a_index < b_index;
    };
    boost::container::set<Node, decltype(PosCmp)> closed_list(PosCmp);
    boost::container::set<Node, decltype(PosCmp)> open_list_pos(PosCmp);

    Node start_node;
    start_node.x = start_x;
    start_node.y = start_y;
    start_node.g = 0;
    start_node.h = heuristic(start_x, start_y, end_x, end_y);
    start_node.parent_x = -1;
    start_node.parent_y = -1;
    open_list.insert(start_node);
    open_list_pos.insert(start_node);

    while (open_list.size() > 0)
    {
        // find best next node, which minimizes overall cost
        Node current_node = *open_list.begin();
        open_list.erase(open_list.begin());
        open_list_pos.erase(current_node);

        if (current_node.x == end_x && current_node.y == end_y)
        {
#ifdef DEBUG_THETA
            printf("Goal Reached!\n");
#endif
            Node node = current_node;
            std::vector<std::pair<int, int>> path;
            while (node.parent_x >= 0 && node.parent_y >= 0)
            {
                path.push_back(std::make_pair(node.x, node.y));
                node.x = node.parent_x;
                node.y = node.parent_y;
                node = *closed_list.find(node);
            }
            path.push_back(std::make_pair(start_x, start_y));
            return path;
        }

        // Move current node from open list to closed list
        closed_list.insert(current_node);

        for (int dx = -1; dx <= 1; dx++)
        {
            for (int dy = -1; dy <= 1; dy++)
            {
                if (dx == 0 && dy == 0)
                {
                    continue;
                }

                int x = current_node.x + dx;
                int y = current_node.y + dy;
                if (get_tile(map, x, y) == OBSTACLE)
                {
                    continue;
                }

                int in_open_list = 0;
                Node neighbor;
                neighbor.x = x;
                neighbor.y = y;

                if (closed_list.contains(neighbor))
                {
                    continue;
                }
#ifdef DEBUG_THETA
                printf("neighbor: (%d, %d) g: %f h: %f => %f\n", neighbor.x, neighbor.y, neighbor.g, neighbor.h, neighbor.g + neighbor.h);
#endif
                auto neighborIt = open_list_pos.find(neighbor);
                if (neighborIt == open_list_pos.end())
                {
                    neighbor.parent_x = -1;
                    neighbor.parent_y = -1;
                    neighbor.g = 99999999;
                    neighbor.h = heuristic(neighbor.x, neighbor.y, end_x, end_y);
                }
                else
                {
                    neighbor = *neighborIt;
                    in_open_list = 1;
                }

#ifdef DEBUG_THETA
                printf("neighbor: (%d, %d) g: %f h: %f => %f\n", neighbor.x, neighbor.y, neighbor.g, neighbor.h, neighbor.g + neighbor.h);
#endif
                if (current_node.parent_x >= 0 && current_node.parent_x >= 0 && line_of_sight(current_node.parent_x, current_node.parent_y, neighbor.x, neighbor.y, map->data, map->size))
                {
                    // path 2
                    Node parent_node;
                    parent_node.x = current_node.parent_x;
                    parent_node.y = current_node.parent_y;
                    auto it = closed_list.find(parent_node);
                    if (it == closed_list.end())
                    {
                        printf("error: Expected parent node\n");
                        exit(1);
                    }
                    parent_node = *it;
                    double new_g = parent_node.g + heuristic(parent_node.x, parent_node.y, neighbor.x, neighbor.y);
                    if (new_g < neighbor.g)
                    {
                        if (in_open_list)
                        {
                            open_list.erase(neighbor);
                            open_list_pos.erase(neighbor);
                        }
                        neighbor.g = new_g;
                        neighbor.parent_x = current_node.parent_x;
                        neighbor.parent_y = current_node.parent_y;
                        open_list.insert(neighbor);
                        open_list_pos.insert(neighbor);
                    }
                }
                else
                {
                    // path 1
                    double new_g = current_node.g + heuristic(current_node.x, current_node.y, neighbor.x, neighbor.y);
                    if (new_g < neighbor.g)
                    {
                        if (in_open_list)
                        {
                            open_list.erase(neighbor);
                            open_list_pos.erase(neighbor);
                        }
                        neighbor.g = new_g;
                        neighbor.parent_x = current_node.x;
                        neighbor.parent_y = current_node.y;
                        open_list.insert(neighbor);
                        open_list_pos.insert(neighbor);
                    }
                }
            }
        }
#ifdef DEBUG_THETA
        for (auto node : open_list)
        {
            printf("open: (%d, %d) g: %f h: %f => %f\n", node.x, node.y, node.g, node.h, node.g + node.h);
        }
#endif
    }
    return {};
}

/*int main(int argc, char *argv[])
{
    Map *map = generate_map(10, 4);
    print_map(map);

    {
        // test line of sight
        Node a = {1, 1, 0, 0, NULL};
        Node b = {8, 8, 0, 0, NULL};
        printf("line of sight: %d\n", line_of_sight(&a, &b, map));

        // another test
        Node c = {1, 1, 0, 0, NULL};
        Node d = {7, 1, 0, 0, NULL};
        printf("line of sight: %d\n", line_of_sight(&c, &d, map));
    }
    {
        theta_star(map, 1, 1, 8, 8);
    }
    print_map(map);

    free_map(map);
}*/