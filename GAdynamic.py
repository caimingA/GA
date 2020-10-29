from model import Human
import random
from collections import Counter
#import geatpy as ea
import matplotlib.pyplot as plt
import numpy as np
import time
import networkx as nx


random.seed(10)
G = nx.Graph()


def creat_graph(adjTable):
    for key, values in adjTable.items():
        # print(key)
        for value in values:
            if int(key) < int(value[0]):
                print(key, "|", value[0])
                G.add_edge(key, value[0], weight=1000 / value[2], name=str(value[2]))
    return True


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


def is_path(path, adjTable):
    if len(path) == 1:
        return False

    for i in range(len(path) - 1):
        flag = False
        for j in adjTable[path[i]]:
            if j[0] == path[i + 1]:
                flag = True
        if flag:
            pass
        else:
            return False

    return True


def is_loop(path):
    counter = dict(Counter(path))
    temp = [key for key, value in counter.items()if value > 1]
    # print(temp)
    while temp:
        i = temp[0]
        # print("a")
        b = list()
        for index, nums in enumerate(path):
            if nums == i:
                b.append(index)

        # print("index:", b)
        path = path[:b[0]] + path[b[1]:]
        counter = dict(Counter(path))
        temp = [key for key, value in counter.items() if value > 1]
    return path


def is_exist(paths, path):

    if path in paths:
        return True

    return False


# 找到一条从start到end的路径
def findPath(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return path
    for node in graph[start]:
        if node not in path:
            newpath = findPath(graph, node, end, path)
            if newpath:
                return newpath
    return None


# def find_all_path(graph, start, end, path=[]):
#     path = path + [start]
#     if start == end:
#         return [path]
#
#     paths = []  # 存储所有路径
#     for node in graph[start]:
#         if node[0] not in path:
#             newpaths = find_all_path(graph, node[0], end, path)
#             for newpath in newpaths:
#                 paths.append(newpath)
#     return paths


def find_all_path(graph, start, end, path=[]):
    shortest_way = nx.shortest_path(G, start, end)
    # print(shortest_way)
    length = len(shortest_way) + 6
    # print(length)
    path = nx.all_simple_paths(G, start, end, length)
    paths = list()
    for p in path:
        paths.append(p)
    return paths


def compute_cost(path, adjTable, speed):
    cost = 0
    for i in range(1, len(path)):
        for j in adjTable[path[i - 1]]:
            if j[0] == path[i]:
                cost += j[2]
    return speed / cost


def compute_length(path, adjTable):
    cost = 0
    # print(path)
    for i in range(1, len(path)):
        # print(i)
        for j in adjTable[path[i - 1]]:
            if j[0] == path[i]:
                cost += j[2]
    # print(cost)
    return cost


def inition(size, allPaths, adjTable, speed, history):
    costs = list()
    init = list()
    count = 0
    for i in sorted(allPaths, key=lambda i: len(i)):
        if count == size:
            break
        temp = [j for j in i if j not in history]
        if temp == i:
            init.append(i)
            count += 1
    for path in init:
        costs.append(compute_cost(path, adjTable, speed))
    return init, costs


# 轮盘赌选择
def wheel_selection(population, costs, adjTable, speed):
    # print(population)
    length = len(population)
    selectPaths = list()
    newCosts = list()

    # tempCosts = [(sum(costs) - i)/10 for i in costs]
    sumCost = sum(costs)
    # print(tempCosts, "|", costs)
    # 适应度计算 and 累计概率
    fitness_sum = list()
    q = list()
    for i in range(length):
        fitness_sum.append(costs[i] / sumCost)
        if q:
            q.append(q[i - 1] + (costs[i] / sumCost))
        else:
            q.append((costs[i]) / sumCost)

    # selection
    for j in range(length):
        r = np.random.uniform(0, 1)
        for i in range(len(q)):
            if i == 0:
                if 0 <= r <= q[i]:
                    if not is_exist(selectPaths, population[i]):
                        selectPaths.append(population[i])
                        newCosts.append(compute_cost(population[i], adjTable, speed))
            else:
                if q[i - 1] <= r <= q[i]:
                    if not is_exist(selectPaths, population[i]):
                        selectPaths.append(population[i])
                        newCosts.append(compute_cost(population[i], adjTable, speed))

    return selectPaths, newCosts, len(selectPaths)


def crossover(paths, costs, adjTable, speed):
    newPaths = list()
    newCosts = list()
    parents = random.sample(paths, 2)
    for father in range(1, len(parents[0]) - 1):
        for mother in range(1, len(parents[1]) - 1):
            if parents[0][father] == parents[1][mother]:
                child1 = parents[0][: father] + parents[1][mother:]
                # print("c1", child1)
                child1 = is_loop(child1)
                if not is_exist(paths, child1):
                    newPaths.append(child1)
                    newCosts.append(compute_cost(child1, adjTable, speed))
                child2 = parents[1][: mother] + parents[0][father:]
                # print("c2", child2)
                child2 = is_loop(child2)
                if not is_exist(paths, child1):
                    newPaths.append(child2)
                    newCosts.append(compute_cost(child2, adjTable, speed))
                return paths + newPaths, costs + newCosts
    return paths, costs


def mutation(paths, costs, adjTable, length, start, end, speed):
    for i in range(10):
        # print(i, ":", paths)
        numbers = random.randint(2, length - 3)
        change = list()
        node = random.randint(1, length - 2)
        for i in range(numbers):
            change.append(str(random.randint(start, end)))

        x = random.randint(0, len(paths) - 1)
        # y = random.randint(1, len(paths[x]) - 2)
        temp = paths[x][:]
        temp = temp[: node] + change + temp[-1: ]
        # temp[y] = str(random.randint(start, end + 1))
        # print(temp)
        is_loop(temp)
        if not is_path(temp, adjTable):
            pass
        elif is_exist(paths, temp):
            pass
        else:
            paths.append(temp)
            costs.append(compute_cost(temp, adjTable, speed))

    return paths, costs


if __name__ == "__main__":
    time_start = time.time()
    path = list()
    adjTable = read_txt()
    print(adjTable)
    creat_graph(adjTable)
    human = list()
    for i in range(8):
        human.append(Human(str(i), random.uniform(1, 3), str(i)))
    # human = [Human('7', 2, '7')]

    flag = 1
    gen = 100
    while flag:
        flag = 0
        # print("a")
        for h in human:
            if h.end == '8':
                if h.arrive:
                    if h.output == 0:
                        print("human" + h.name, h.time)
                        print(h.history)
                        h.output = 1
                else:
                    flag = 1
                    h.history.append('8')
                    h.time = compute_length(h.history, adjTable) / h.speed
                    h.arrive = 1
            else:
                flag = 1
                # print('a')
                if h.isCompute:
                    h.start = h.end
                    h.allPath = find_all_path(adjTable, h.start, '8', path)
                    # print(h.allPath)
                    for i in h.allPath:
                        h.allCosts.append(compute_cost(i, adjTable, h.speed))
                    h.initPaths, h.costs = inition(4, h.allPath, adjTable, h.speed, h.history)

                    h.Paths, h.Costs, N = wheel_selection(h.initPaths, h.costs, adjTable, h.speed)
                    while len(h.Paths) < 4:
                        newPath = random.choice(h.allPath)
                        if newPath not in h.Paths:
                            h.Paths.append(newPath)
                            h.Costs.append(compute_cost(newPath, adjTable, h.speed))

                    h.Paths, h.Costs = mutation(h.Paths, h.Costs, adjTable, len(adjTable), 1, 7, h.speed)

                    h.Paths, h.Costs = crossover(h.Paths, h.Costs, adjTable, h.speed)

                    result = list()

                    for i in range(gen):
                        h.Paths, h.Costs, N = wheel_selection(h.initPaths, h.costs, adjTable, h.speed)
                        # print(Paths, Costs)
                        index = h.Costs.index(max(h.Costs))
                        # print(index)
                        result.append([i, h.Paths[index], h.Costs[index]])

                        while len(h.Paths) < 4:
                            newPath = random.choice(h.allPath)
                            if newPath not in h.Paths:
                                h.Paths.append(newPath)
                                h.Costs.append(compute_cost(newPath, adjTable, h.speed))

                        h.Paths, h.Costs = mutation(h.Paths, h.Costs, adjTable, len(adjTable), 1, 7, h.speed)
                        # print(Paths, Costs)

                        h.Paths, h.Costs = crossover(h.Paths, h.Costs, adjTable, h.speed)
                        # print(Paths, Costs)
                    if h.start == h.end:
                        h.set_finish(h.speed)
                    else:
                        h.set_finish(h.speed - (h.length - h.finish))
                    h.history.append(result[-1][1][0])
                    h.update_start_end(result[-1][1][0], result[-1][1][1])
                    # print(result[-1])
                    for i in adjTable[h.start]:
                        if i[0] == h.end:
                            h.length = i[2]
                    h.is_compute()
                    # print("human" + h.name, h.start, h.end, h.finish)
                    x = list()
                    y = list()
                    for i in range(len(result)):
                        x.append(result[i][0])
                        y.append(result[i][2])

                    plt.plot(x, y)
                    plt.xlabel("generation (" + h.name + ')')
                    plt.ylabel("chosen route")
                    plt.show()

                else:
                    h.set_finish(h.finish + h.speed)
                    h.is_compute()
    time_end = time.time()
    print('time cost', time_end - time_start, 's')
