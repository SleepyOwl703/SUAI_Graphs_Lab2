# -*- coding: cp1251 -*-

from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import time
import random
from collections import defaultdict

# ����� ��� �������� �����
@dataclass
class Edge:
    v1: str
    v2: str
    weight: int

# ��������� ���� � ���� ������ �����

r_list = [Edge('A','B',7),
          Edge('A','D',5),
          Edge('B','C',8),
          Edge('B','D',9),
          Edge('B','E',7),
          Edge('C','E',5),
          Edge('D','E',15),
          Edge('D','F',6),
          Edge('E','F',8),
          Edge('E','G',9),
          Edge('F','G',11)]

# ���������� ����������� �������

def create_graph(M):
    grid_edges = []
    for V in range(0,M**2):
        R = int (V / M)
        C = V % M
        if (C < (M-1)):
            grid_edges.append(Edge(V,V+1,1))
        if (R < (M-1)):
            grid_edges.append(Edge(V,V+M,1))
            if ((R % 2 == 0) and (C % 2 == 0)) or ((R % 2 == 1) and (C % 2 == 1)):
                if C > 0:
                    grid_edges.append(Edge(V,V+M-1,1))
                if C < (M-1):
                    grid_edges.append(Edge(V,V+M+1,1))
    return grid_edges


# ���������� �����
def visualization(edges, M, isWeight = False):
   G = nx.Graph()                                                  # ������������� �����
   for edge in edges:
       G.add_edge(edge.v1,edge.v2,weight = edge.weight)            # ���������� �����

   # ��������� �����������
   options = {              
    "font_size": 25,
    "node_size": 1500,
    "node_color": "orange",
    "edgecolors": "black",
    "linewidths": 2,
    "width": 2,
    }

   pos = nx.spring_layout(G,seed=7)                                 # ������������ ������ �� ���������
   if not isWeight:
       pos = {i: (i % M, -(i // M)) for i in range(0,M**2)} 
   # ������������ ������ ��� �������
   else:
       edge_labels = nx.get_edge_attributes(G, "weight")            # ���������� �������� ����� �����
       nx.draw_networkx_edge_labels(G, pos, edge_labels)

   nx.draw_networkx(G, pos, **options)                              # ��������� �����

   # ��������� � ����� ������
   ax = plt.gca()                                                   
   ax.margins(0.08)
   plt.axis("off")
   plt.tight_layout()
   plt.show()

#create_graph(7)                         # �������� �������
#visualization(r_list, 7, True)          # ������������ ���������� �����
#visualization(create_graph(5), 5)       # ������������ �������

# �������� �����

def prim(edges, nodes):
    # ������������ ������ ����� �� ������ ����, ��� ���� �������������� � ����� �������� � ���� �������
    all_list = defaultdict(list)
    for edge in edges:
        all_list[edge.v1].append((edge.v2, edge.weight))
        all_list[edge.v2].append((edge.v1, edge.weight))

    visited = set()
    res = []

    # � �������� ������������ ������� ������� 1-��
    start_vertex = edges[0].v1
    visited.add(start_vertex)

    while len(visited) < nodes:
        min_edge = None
        min_weight = float('inf')

        # ���� �� ��������
        for vertex in visited:
            # Explore edges connected to visited vertices
            for neighbor, weight in all_list[vertex]:
                # Check if the neighbor is not visited and the edge weight is minimum
                if neighbor not in visited and weight < min_weight:
                    min_edge = (vertex, neighbor)
                    min_weight = weight

        if min_edge:
            # ���������� ����������� �����
            res.append(Edge(min_edge[0], min_edge[1], min_weight))
            # ��������� ������� � ����������
            visited.add(min_edge[1])
        else:
            # �� �������� ������ - ������� �� �����
            break
    return res

def create_graphs_dynamically(M, ran=100):
    total_time = 0
    for i in range(ran):
        graph = create_graph(M) # ������������ �������� �����
        for edge in graph:
            edge.weight = random.randint(1,25)      # ��������� ��������� �����
        start_time = time.time()     # ��������� ������� �� ������ ���������
        min_tree = prim(graph,M**2)     # ��������� ���������
        end_time = time.time()       # ��������� ������� ����� ��������� ���������
        # ����� �����
        total_time += end_time - start_time
    # ������� ����� ����������
    return total_time / ran


# ������� �������
sizes = [5, 10, 15, 20, 25]

avg_time = [create_graphs_dynamically(size) for size in sizes]

plt.plot(sizes, avg_time)
plt.title('����������� �������� ������� ���������� �� ������� �������')
plt.xlabel('������ �������')
plt.ylabel('������� ����� ���������� (�)')
plt.grid(True)
plt.show()