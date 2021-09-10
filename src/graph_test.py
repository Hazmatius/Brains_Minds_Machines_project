from ant_control import Graph
from ant_control import Node

graph = Graph()

print('constructing graph...')

graph.add_node(Node([1, 2], 1))
graph.add_node(Node([2, 2], 2))
graph.add_node(Node([3, 2], 3))
graph.add_node(Node([4, 2], 4))
graph.add_node(Node([2, 3], 5))
graph.add_node(Node([3, 3], 6))
graph.add_node(Node([4, 3], 7))
graph.add_node(Node([3, 4], 8))
graph.add_node(Node([4, 4], 9))
graph.add_node(Node([4, 5], 10))

graph.connect_nodes(1, 2)
graph.connect_nodes(2, 3)
graph.connect_nodes(3, 4)
graph.connect_nodes(1, 5)
graph.connect_nodes(1, 6)
graph.connect_nodes(4, 6)
graph.connect_nodes(4, 7)
graph.connect_nodes(5, 8)
graph.connect_nodes(6, 10)
graph.connect_nodes(7, 9)
graph.connect_nodes(8, 10)
graph.connect_nodes(9, 10)

print('traversing...')

path = graph.traverse(1, 10)
for node in path:
    print(node.idx)

