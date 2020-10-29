import random
from collections import Counter
#import geatpy as ea
import matplotlib.pyplot as plt
import numpy as np
import time


random.seed(10)
def read_txt():
    adjTable = dict()
    f = open("route.txt", mode='r', encoding="utf-8", )
    for line in f:
        line = line.replace("\n", "")
        line = line.split(sep=',')
        if line[0] in adjTable:
            adjTable[line[0]].append([line[1], float(line[2])])
        else:
            adjTable[line[0]] = [[line[1], float(line[2])]]  # node1, node2, cost

        if line[1] in adjTable:
            adjTable[line[1]].append([line[0], float(line[2])])
        else:
            adjTable[line[1]] = [[line[0], float(line[2])]]  # node1, node2, cost
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
    # print(path)
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
        # print(path)
        # print(temp)

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


def find_all_path(graph, start, end, path=[]):
    path = path + [start]
    if start == end:
        return [path]

    paths = []  # 存储所有路径
    for node in graph[start]:
        if node[0] not in path:
            newpaths = find_all_path(graph, node[0], end, path)
            for newpath in newpaths:
                paths.append(newpath)
    return paths


def compute_cost(path, adjTable):
    cost = 0
    for i in range(1, len(path)):
        for j in adjTable[path[i - 1]]:
            if j[0] == path[i]:
                cost += j[1]
    return cost


def inition(size, allPaths, adjTable):
    costs = list()
    random.seed(10)
    init = random.sample(allPaths, size)
    for path in init:
        costs.append(compute_cost(path, adjTable))
    return init, costs


# def inition(allPaths, adjTable):
#     costs = list()
#     for path in allPaths:
#         costs.append(compute_cost(path, adjTable))
#     return costs


# 轮盘赌选择
def wheel_selection(population, costs, adjTable):
    length = len(population)
    selectPaths = list()
    newCosts = list()
    # 适应度计算 and 累计概率
    fitness_sum = list()
    q = list()
    for i in range(length):
        fitness_sum.append(costs[i] / sum(costs))
        if q:
            q.append(q[i - 1] + costs[i] / sum(costs))
        else:
            q.append(costs[i] / sum(costs))

    print(fitness_sum)
    # print(sum(fitness_sum))
    print(q)

    # selection
    for j in range(length):
        r = np.random.uniform(0, 1)
        for i in range(len(q)):
            if i == 0:
                if 0 <= r <= q[i]:
                    if not is_exist(selectPaths, population[i]):
                        selectPaths.append(population[i])
                        newCosts.append(compute_cost(population[i], adjTable))
            else:
                if q[i - 1] <= r <= q[i]:
                    if not is_exist(selectPaths, population[i]):
                        selectPaths.append(population[i])
                        newCosts.append(compute_cost(population[i], adjTable))

    return selectPaths, newCosts, len(selectPaths)


def crossover(paths, costs, adjTable):
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
                    newCosts.append(compute_cost(child1, adjTable))
                child2 = parents[1][: mother] + parents[0][father:]
                # print("c2", child2)
                child2 = is_loop(child2)
                if not is_exist(paths, child1):
                    newPaths.append(child2)
                    newCosts.append(compute_cost(child2, adjTable))
                return paths + newPaths, costs + newCosts
    return paths, costs


def mutation(paths, costs, adjTable, length, start, end):
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
            costs.append(compute_cost(temp, adjTable))

    return paths, costs


if __name__ == "__main__":
    path = list()
    adjTable = read_txt()
    allPaths = find_all_path(adjTable, '0', '8', path)
    allCosts = list()
    for i in allPaths:
        allCosts.append(compute_cost(i, adjTable))
    initPath, costs = inition(3, allPaths, adjTable)
     # print(initPath)

    Paths, Costs, N = wheel_selection(initPath, costs, adjTable)

    Paths, Costs = mutation(Paths, Costs, adjTable, len(adjTable), 1, 7)

    Paths, Costs = crossover(Paths, Costs, adjTable)

    gen = 10
    result = list()

    for i in range(gen):
        Paths, Costs, N = wheel_selection(initPath, costs, adjTable)
        # print(Paths, Costs)
        index = Costs.index(max(Costs))
        # print(index)
        result.append([i, Paths[index], Costs[index]])

        while len(Paths) < 4:
            newPath = random.choice(allPaths)
            if newPath not in Paths:
                Paths.append(newPath)
                Costs.append(compute_cost(newPath, adjTable))


        # print(Paths, Costs)

        Paths, Costs = mutation(Paths, Costs, adjTable, len(adjTable), 1, 7)
        # print(Paths, Costs)

        Paths, Costs = crossover(Paths, Costs, adjTable)
        # print(Paths, Costs)

    print(result)
    maxPath = allPaths[allCosts.index(max(costs))]
    print(maxPath)
    print(result[-1])

    ### プロット
    plt.plot(allCosts)
    plt.xlabel("the number of route")
    plt.ylabel("costs")

    my_x_ticks = np.arange(0, 12, 1)
    # my_y_ticks = np.arange(-2, 2, 0.3)
    plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)

    plt.show()
    x = list()
    y = list()
    for i in range(len(result)):
        x.append(result[i][0])
        y.append(result[i][2])

    plt.plot(x, y)
    plt.xlabel("generation")
    plt.ylabel("chosen route")
    my_x_ticks = np.arange(0, 10, 1)
    plt.xticks(my_x_ticks)
    plt.show()



