import networkx as nx
import matplotlib.pyplot as plt

def read_txt():
    adjTable = dict()
    f = open("route.txt", mode='r', encoding="utf-8", )
    for line in f:
        line = line.replace("\n", "")
        line = line.split(sep=',')
        if line[0] in adjTable:
            adjTable[line[0]].append([line[1], float(line[2]), float(line[3])])
        else:
            adjTable[line[0]] = [[line[1], float(line[2]), float(line[3])]]  # node1, node2, cost

        if line[1] in adjTable:
            adjTable[line[1]].append([line[0], float(line[2]), float(line[3])])
        else:
            adjTable[line[1]] = [[line[0], float(line[2]), float(line[3])]]  # node1, node2, cost
    f.close()

    return adjTable


if __name__ == '__main__':
    print(read_txt())
    graph = read_txt()
    plt.figure(figsize=(10, 10))
    G = nx.Graph()
    for key, values in graph.items():
        # print(key)
        for value in values:
            if int(key) < int(value[0]):
                print(key, "|", value[0])
                G.add_edge(key, value[0], weight=1000 / value[2], name=str(value[2]))

    pos = nx.spring_layout(G, k=0.5)

    # size

    nx.draw_networkx_nodes(G, pos, node_size=5000)
    nx.draw_networkx_edges(G, pos, )
    nx.draw_networkx_labels(G, pos, font_size=40)
    edge_labels = nx.get_edge_attributes(G, 'name')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=40)
    shortest_way = nx.shortest_path(G, "0", "8")
    print(shortest_way)
    length = len(shortest_way)
    paths = nx.all_simple_paths(G, "0", "8", length)

    # print(paths)
    for p in paths:
        print(p)
    plt.axis("off")
    plt.savefig('figure.png')
    plt.show()
