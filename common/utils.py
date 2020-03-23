import queue
import sys
import numpy as np


def create_map(edge_list, map_size):
    edge_map = [[sys.float_info.max / 3 for x in range(map_size)] for y in range(map_size)]
    for edge in edge_list:
        # maybe multi lines between i and j ,we accpet min of them
        if edge['dis'] < edge_map[edge['i']][edge['j']]:
            edge_map[edge['i']][edge['j']] = edge_map[edge['j']][edge['i']] = edge['dis']
    for x in range(1, map_size):
        edge_map[x][x] = 0
    return edge_map


# get shortest path from fault center by dijkstra
def dijkstra(edge_g, map_size, x):
    q = queue.Queue()
    d = np.ones(map_size) * sys.float_info.max / 3
    used = [False for t in range(map_size)]
    q.put(x)
    d[x] = 0
    used[x] = True
    while not q.empty():
        s = q.get()
        for t in range(1, map_size):
            if d[s] + edge_g[s][t] < d[t]:
                d[t] = d[s] + edge_g[s][t]
                if not used[t]:
                    used[t] = True
                    q.put(t)
        used[s] = False
    return d
