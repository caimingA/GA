import random
from collections import Counter
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
                # print(key, "|", value[0])
                G.add_edge(key, value[0], weight=1000 / value[2], name=str(value[2]))
    return True


def read_txt():
    adjTable = dict()
    f = open("routeTest.txt", mode='r', encoding="utf-8", )
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
    shortest_way = nx.shortest_path(G, start, end)
    # print(shortest_way)
    length = len(shortest_way) + 4
    # print(length)
    path = nx.all_simple_paths(G, start, end, length)
    paths = list()
    for p in path:
        paths.append(p)
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


# 轮盘赌选择
def wheel_selection(population, costs, adjTable):
    length = len(population)
    selectPaths = list()
    newCosts = list()
    sumcost = sum(costs)
    base = max(costs)
    reshapeCost = [base - i + 1 for i in costs]
    resumcost = sum(reshapeCost)
    # print(costs)
    # print(reshapeCost)
    # 适应度计算 and 累计概率
    # fitness_sum = list()
    q = list()
    for i in range(length):
        # fitness_sum.append(costs[i] / sum(costs))
        # print(i)
        if q:
            q.append(q[i - 1] + reshapeCost[i] / resumcost)
        else:
            q.append(reshapeCost[i] / resumcost)

    # selection
    for j in range(length):
        r = np.random.uniform(0, 1)
        for i in range(len(q)):
            if r < q[i]:
                t = population[i]
                selectPaths.append(t)
                newCosts.append(compute_cost(t, adjTable))
                break

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


def mental_cost(paths, adjTable):
    mentalCost = 0
    for i in range(1, len(paths)):
        for j in adjTable[paths[i - 1]]:
            if j[0] == paths[i]:
                mentalCost += 100*j[1]/j[2]
                print(j[1])
    # print(mentalCost)
    return mentalCost



def pareto(paths, adjTable):
    resultList = list()
    for path in paths:
        physicalCost = compute_cost(path, adjTable)
        mentalCost = mental_cost(path, adjTable)
        temp = [physicalCost, mentalCost]
        if resultList:
            for i in range(len(resultList)):
                if resultList[i][1][0] > temp[0] and resultList[i][1][1] < temp[1]:
                    if temp not in resultList:
                        resultList[i] = [path, temp]
                elif resultList[i][1][0] < temp[0] and resultList[i][1][1] > temp[1]:
                    pass
                else:
                    resultList.append([path, temp])
                 
        else:
            resultList.append([path, temp])

    resultPath = random.sample(resultList, 1)[0]
    # print(resultList)

    return resultPath[0], compute_cost(resultPath[0], adjTable)


if __name__ == "__main__":
    path = list()
    adjTable = read_txt()
    # print(adjTable)
    creat_graph(adjTable)
    # from 0 to 8
    allPaths = find_all_path(adjTable, '1', '11', path)
    print(allPaths)
    allCosts = list()
    for i in allPaths:
        allCosts.append(compute_cost(i, adjTable))
    initPath, costs = inition(10, allPaths, adjTable)
    # print(initPath)

    Paths, Costs, N = wheel_selection(initPath, costs, adjTable)

    Paths, Costs = mutation(Paths, Costs, adjTable, len(adjTable), 1, 7)

    Paths, Costs = crossover(Paths, Costs, adjTable)

    gen = 50
    result = list()

    for i in range(gen):
        # print(Paths)
        Paths, Costs, N = wheel_selection(Paths, Costs, adjTable)
        # print(Paths, Costs)
        # index = Costs.index(min(Costs))
        # print(index)
        resultPath, resultCost = pareto(Paths, adjTable)
        result.append([i, resultPath, resultCost])

        while len(Paths) < 10:
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
    minPath = allPaths[allCosts.index(min(costs))]
    print(minPath)
    print(result[-1])

    ### プロット
    plt.figure('result')
    plt.subplot(121)
    plt.plot(allCosts)
    plt.xlabel("the number of route")
    plt.ylabel("costs")

    my_x_ticks = np.arange(0, 12, 1)
    # my_y_ticks = np.arange(-2, 2, 0.3)
    plt.xticks(my_x_ticks)
    # plt.yticks(my_y_ticks)

    # plt.show()
    x = list()
    y = list()
    for i in range(len(result)):
        x.append(result[i][0])
        y.append(result[i][2])

    plt.subplot(122)
    plt.plot(x, y)
    plt.xlabel("generation")
    plt.ylabel("chosen route")
    # my_x_ticks = np.arange(0, 25, 5)
    # plt.xticks(my_x_ticks)
    plt.show()
