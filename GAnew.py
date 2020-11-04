import math
import random
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import time
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D


random.seed(10)
G = nx.Graph()
adjTable = dict()
# allPaths = list()
# shortPaths = list()
num = 0
total = 0
shelterList = ["28", "38", "85", "103", "104", "109", "112", "124", "170", "178", "189", "194", "203", "220", "274"]

def creat_graph():
    for key, values in adjTable.items():
        # print(key)
        for value in values:
            if int(key) < int(value[0]):
                # print(key, "|", value[0])
                G.add_edge(key, value[0], weight=1000 / value[2], name=str(value[2]))
    return True


def read_txt():
    adjTable = dict()
    f = open("routeKyoto.txt", mode='r', encoding="utf-8", )
    for line in f:
        line = line.replace("\n", "")
        line = line.split(sep=',')
        if line[0] in adjTable:
            adjTable[line[0]].append([line[1], float(line[2]), float(line[3]), float(line[4]), float(line[5])])
        else:
            adjTable[line[0]] = [[line[1], float(line[2]), float(line[3]), float(line[4]), float(line[5])]]  # node1, node2, cost

        if line[1] in adjTable:
            adjTable[line[1]].append([line[0], float(line[2]), float(line[3]), float(line[4]), float(line[5])])
        else:
            adjTable[line[1]] = [[line[0], float(line[2]), float(line[3]), float(line[4]), float(line[5])]]  # node1, node2, cost
    f.close()

    return adjTable


def creat_random(N):
    x0 = A = 48271
    B = 0
    result = list()
    for i in range(N):
        x0 = (A*x0 + B) % num
        result.append(x0)
    return result


def is_path(path):
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
    shortest_way = nx.shortest_path(G, start, end)
    
    return shortest_way


def find_all_path(start, end, path=[]):
    num = 0
    shortest_way = nx.shortest_path(G, start, end)
    # print(shortest_way)
    length = len(shortest_way) + 4
    # print(length)
    path = list(nx.all_simple_paths(G, start, end, length))
    shortPath = list(nx.all_simple_paths(G, start, end, length - 4))
    # paths = list()
    # for p in path:
    #     paths.append(p)
    #     num += 1
    
    return path, len(path), length - 4, shortPath


def find_shelter(k, start):
    lengthList = list()
    result = list()
    
    for end in shelterList:
        flag = 0
        wayLength = len(nx.shortest_path(G, start, end))
        # print(end, ":", wayLength)
        if len(lengthList) < k:
            lengthList.append(wayLength)
            result.append(end)
        else:
            maxLength = lengthList[0]
            maxPoint = 0
            for i in range(k):
                if lengthList[i] > maxLength:
                    maxLength = lengthList[i]
                    maxPoint = i
            if wayLength < maxLength:
                lengthList[maxPoint] = wayLength
                result[maxPoint] = end    

    # print(shortestWayLength)
    return result


def compute_cost(path):
    cost = 0
    for i in range(1, len(path)):
        for j in adjTable[path[i - 1]]:
            if j[0] == path[i]:
                cost += j[2]
    
    return cost


def mental_cost(paths):
    mentalCost = 0
    for i in range(1, len(paths)):
        for j in adjTable[paths[i - 1]]:
            if j[0] == paths[i]:
                mentalCost += 100*j[1]/j[2]
                # print(j[1])
    # print(mentalCost)
    return mentalCost


def cost_list(paths):
    n = 0
    length = 0
    area = 0
    bridge = 0
    build = 0
    for i in range(1, len(paths)):
        for j in adjTable[paths[i - 1]]:
            if j[0] == paths[i]:
                n += 1
                length += j[2]
                area += j[1] * j[2]
                build += j[3] / area
                if j[4] == 1:
                    bridge += j[2]

    # print(length)
    # print(n)
    cost_list = [length, area / length, bridge, build / n]
    return cost_list


def inition(size, allPaths, shortPaths):
    n = len(shortPaths)
    costs = list()
    init = random.sample(allPaths, size) 
    for p in shortPaths:
        if p in init:
            continue
        else:
            init.append(p)
    return init

    
def wheel_selection(population, costs):
    # print(population, '->', costs)
    length = len(population)
    selectPaths = list()
    newCosts = list()
    sumcost = sum(costs)
    
    # 适应度计算 and 累计概率
    q = list()
    for i in range(length):
        if q:
            q.append(q[i - 1] + costs[i] / sumcost)
        else:
            q.append(costs[i] / sumcost)

    # selection
    for j in range(length):
        r = np.random.uniform(0, 1)
        for i in range(len(q)):
            if i == 0:
                if 0 <= r <= q[i]:
                    if not is_exist(selectPaths, population[i]):
                        selectPaths.append(population[i])
                        newCosts.append(costs[i])
            else:
                if q[i - 1] <= r <= q[i]:
                    if not is_exist(selectPaths, population[i]):
                        selectPaths.append(population[i])
                        newCosts.append(costs[i])

    return selectPaths, newCosts


def crossover(paths):
    newPaths = list()
    parents = random.sample(paths, 2)
    for father in range(1, len(parents[0]) - 1):
        for mother in range(1, len(parents[1]) - 1):
            if parents[0][father] == parents[1][mother]:
                child1 = parents[0][: father] + parents[1][mother:]
                # print("c1", child1)
                child1 = is_loop(child1)
                if not is_exist(paths, child1):
                    newPaths.append(child1)
                child2 = parents[1][: mother] + parents[0][father:]
                # print("c2", child2)
                child2 = is_loop(child2)
                if not is_exist(paths, child1):
                    newPaths.append(child2)
                return paths + newPaths
    return paths


def mutation(paths, allPaths):
    path = random.sample(paths, 1)
    length = len(path[0])
    # print("length->", length)
    # print(path)
    if length <= 2:
        return paths
    index = random.randint(1, length - 2)
    tempPaths = random.sample(allPaths, int(total/2))
    for i in tempPaths:
        if str(index) in i:
            if i not in path:
                path.append(i)

    return paths


def sharing(pathi, pathj):
    length = len(pathi)
    overlap = length - len(set(pathi + pathj))
    return overlap / length


def pareto_ranking(paths):
    resultList = list()
    fit = list()
    for path in paths:
        rank = 1
        # print(path)
        # temp = [compute_cost(path), mental_cost(path)]
        temp = cost_list(path)
        if resultList:
            for i in resultList:
                if temp[0] < i[1][0] and temp[1] > i[1][1] and temp[2] <= i[1][2] and temp[3] < i[1][3]:
                    rank += 1
                elif temp[0] > i[1][0] and temp[1] < i[1][1] and temp[2] >= i[1][2] and temp[3] > i[1][3]:
                    i[2] += 1
                else:
                    pass
            resultList.append([path, temp, rank])
        else:
            resultList.append([path, temp, rank])
        fit.append(rank)
        # print(resultList)

    # print(1)
    # sharing
    Fit = list()
    m = list()
    delta = 0 # 重复率　重複の割合 
    for i in resultList:
        s = 1
        for j in resultList:
            d = sharing(i[0], j[0])
            if d > delta:
                s += 1
        m.append(s)
        Fit.append(i[2] / s)
    resultPath = [i[0] for i in resultList]
    return resultPath, Fit


def find_Road(start, end):
    path = list()
    allPaths, num, length, shortPaths = find_all_path(start, end, path)
    # total = len(allPaths)
    # print(shortPaths)
    if length <= 3:
        total = 12
        initPath = inition(10, allPaths, shortPaths)
    elif length <= 6:
        total = 20
        initPath = inition(50, allPaths, shortPaths)
    elif length <= 10:
        total = 40
        initPath = inition(100, allPaths, shortPaths)
    else:
        total = 80
        initPath = inition(100, allPaths, shortPaths)
    # print(allPaths)
    # allCosts = list()
    # for i in allPaths:
    #     allCosts.append(compute_cost(i))
    # initPath = inition(int(total/2))
    # print(initPath)

    Paths, Costs = pareto_ranking(initPath)
    # print(0)
    Paths, Costs = wheel_selection(Paths, Costs)

    gen = 100
    result = list()

    for i in range(gen):
        while len(Paths) < total:
            newPath = random.choice(allPaths)
            if newPath not in Paths:
                Paths.append(newPath)

        # print(Paths, Costs)

        Paths = mutation(Paths, allPaths)
        # print(Paths, Costs)
        # print(2)
        Paths = crossover(Paths)
        # print(Paths, Costs)
        # print(3)
        
        
        # print(len(Paths))
        Paths, Costs = pareto_ranking(Paths)
        Paths, Costs = wheel_selection(Paths, Costs)
        # print(Paths, Costs)
        index = Costs.index(max(Costs))
        # print(index)
        # print(1)
        # resultPath, resultCost = pareto(Paths)
        # result.append([i, resultPath, resultCost])
        result.append([i, Paths[index], cost_list(Paths[index])])
        resultPaths = Paths
        # Paths = list(set(Paths))
        # Paths.sort(key=Paths.index)
        # temp = list()
        # for i in Paths:
        #     if i not in temp:
        #         temp.append(i)
        # Paths = temp
        # print(len(temp), Paths)
    print("start:", start)
    print("end:", end)
    print("number of nodes in shortest route:", length)
    print("number of all route:", num)
    print("**************results as follow**************")
    print(len(resultPaths))
    # print(resultPaths)
    resultDict = dict()
    selectResult = list()
    for i in resultPaths:
        length = len(i)
        if length in resultDict:
            resultDict[length].append(i)
        else:
            resultDict[length] = [i]
    resultKeys = sorted(resultDict.keys())
    if len(resultKeys) == 1:
        print(len(resultDict[resultKeys[0]][0]), "->", resultDict[resultKeys[0]])
        selectResult = resultDict[resultKeys[0]]
    else:
        print(len(resultDict[resultKeys[0]][0]), "->", resultDict[resultKeys[0]])
        print(len(resultDict[resultKeys[1]][0]), "->", resultDict[resultKeys[1]])
        selectResult = resultDict[resultKeys[0]] + resultDict[resultKeys[1]]
    # print(costs)
    # print("max", maxPath)
    # print(cost_list(maxPath))
    # print("min", minPath)
    # print(cost_list(minPath))
    # print(result[-1])
    # print(cost_list(result[-1][1]))

if __name__ == "__main__":
    startTime = time.time()
    adjTable = read_txt()
    creat_graph()
    # print(adjTable)
    k = 3
    endList = find_shelter(k, "1")
    print("shelter points:", endList)
    for end in endList:
        print('--------------------------------------------')
        find_Road("1", end)

    # print(result)
    # maxPath = allPaths[allCosts.index(max(costs))]
    # minPath = allPaths[allCosts.index(min(costs))]
    
    
    endTime = time.time()
    print("total time: ", (endTime-startTime))
    ## プロット

    # fig = plt.figure('result')
    # ax = fig.add_subplot(111, projection='3d')
    # x = list()
    # y = list()
    # z = list() 
    # # # print(result)
    # for i in allPaths:
    #     temp = cost_list(i)
    #     x.append(temp[0])
    #     y.append(temp[1])
    #     z.append(temp[3])
            
    # ax.scatter3D(x, y, z, color = 'g')
    # # for i in range(len(resultPaths)):
    # #     temp = cost_list(resultPaths[i])
    # #     ax.scatter3D(temp[0], temp[1], temp[3], color = 'g')
    # for i in selectResult:
    #     temp = cost_list(i)
    #     ax.scatter3D(temp[0], temp[1], temp[3], color = 'b')
    #     print(temp)
    # plt.show()
    
    # plt.subplot(121)
    # plt.scatter(x, y)
    # plt.scatter(result[99][2][0], result[99][2][1], color = 'r')
    # # plt.plot(allCosts)
    # # plt.xlabel("the number of route")
    # # plt.ylabel("costs")

    # # my_x_ticks = np.arange(0, 12, 1)
    # # my_y_ticks = np.arange(-2, 2, 0.3)
    # # plt.xticks(my_x_ticks)
    # # plt.yticks(my_y_ticks)

    # # plt.show()
    # x = list()
    # y = list()
    # # print(result)
    # for i in range(len(result)):
    #     x.append(result[i][2][0])
    #     y.append(result[i][2][1])

    # plt.subplot(122)
    # plt.scatter(x, y)
    # plt.scatter(x[-1], y[-1], color = 'r')
    # # for i in range(len(x)):
    # #     plt.annotate(result[i][0], xy = (x[i], y[i]), xytext = (x[i]+0.1, y[i]+0.1))
    # plt.xlabel("physical")
    # plt.ylabel("mental")
    # # plt.xlabel("generation")
    # # plt.ylabel("chosen route")
    # # my_x_ticks = np.arange(0, 25, 5)
    # # plt.xticks(my_x_ticks)
    # plt.show()
    #test