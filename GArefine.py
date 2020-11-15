import math
import random
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import time
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D


MAX_FLOAT = float('inf')


G = nx.Graph()
adjTable = dict() # 由文件生成的邻接表
total = 0
# 所有避难所的节点序号
shelterList = ["28", "38", "85", "103", "104", "109", "112", "124", "170", "178", "189", "194", "203", "220", "274"]
bridgeNodeList = ["17", "18", "41", "42", "163", "164", "253", "254"]
FitHistory = [[], [], [], []]
CostHistoryAvg = [[], [], [], [], [], []]
CostHistoryExt = [[], [], [], [], [], []]
HistoryList = list()


# 用network创造拓扑图
def creat_graph():
    for key, values in adjTable.items():
        # print(key)
        for value in values:
            if int(key) < int(value[0]):
                # print(key, "|", value[0])
                G.add_edge(key, value[0], weight=1000 / value[2], name=str(value[2]))
    return True


# 读取txt文件
def read_txt():
    # adjTable = dict()
    f = open("routeKyoto.txt", mode='r', encoding="utf-8")
    for line in f:
        line = line.replace("\n", "")
        line = line.split(sep=',')
        if line[0] in adjTable:
            adjTable[line[0]].append([line[1], float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6])])
        else:
            adjTable[line[0]] = [[line[1], float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6])]]  # node1, node2, cost

        if line[1] in adjTable:
            adjTable[line[1]].append([line[0], float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6])])
        else:
            adjTable[line[1]] = [[line[0], float(line[2]), float(line[3]), float(line[4]), float(line[5]), float(line[6])]]  # node1, node2, cost
    f.close()

    return adjTable


# 寻找所有路径
# return 所有路径 路径的总数 最短路径(节点数)长度 所有最短路径
def find_all_path(start, end):
    num = 0
    path = list()
    shortest_way = nx.shortest_path(G, start, end)
    # print(shortest_way)
    length = len(shortest_way) + 3
    # print(length)
    path = list(nx.all_simple_paths(G, start, end, length))
    shortPath = list(nx.all_simple_paths(G, start, end, length - 4))
    
    return path, len(path), length - 3, shortPath


def is_loop(path):
    return (not (len(path) == len(set(path))))


def drop_loop(path):
    length = len(path)
    for j in range(length - 1, -1, -1):
        for i in range(1, j, 1):
            if path[i] == path[j]:
                newPath = path[:i] + path[j:]
                break
    return newPath


def cost_list(paths, speed, mental_list):
    
    n = 0 # 节点数
    length = 0 # 长度
    area = 0 # 面积
    bridge = 0 # 桥
    build = 0 # 建筑评价
    congestion = 0 # 拥挤度评价
    for i in range(1, len(paths)): # 循环所有点
        for j in adjTable[paths[i - 1]]: # 遍历所有与上一个节点相连接的节点
            if j[0] == paths[i]:
                n += 1
                length += j[2]
                # area += j[1] * j[2]
                area += j[1]
                build += j[3]
                # congestion += j[5] * j[1] * j[2]
                congestion += j[5]
                if j[4] == 1:
                    bridge += j[2]

    # print(paths)
    # print(length, " and ", n)
    cost_list = [
                  length / speed
                , length
                , area / length
                , bridge
                , build / area
                , congestion / area
                ]
    
    return cost_list


# 初始化
def inition(size, allPaths, shortPaths):
    n = len(shortPaths)
    costs = list()
    init = random.sample(allPaths, size - 1)
    # if size <= n:
    #     init = random.sample(shortPaths, size) # 随机选择(size - 1)个
    # else:
    #     init = shortPaths
    #     while len(init) < size:
    #         temp = random.choice(allPaths)
    #         if temp not in init:
    #             init.append(temp)

    for p in shortPaths: # 判断最短路径是否已在初始化的道路里面
        if p in init:
            continue
        else:
            init.append(p)
            break # 加入一次最短路径就跳出
    # # print(len(init))
    return init


def share_R(pathOne, pathTwo):
    total = len(set(pathOne + pathTwo)) - 2 # 减去起点和终点
    overlap = len(pathOne + pathTwo) - 4 - total
    return overlap / total


# fitness计算
def paretp_ranking(paths, sharing, speed, mental_list):
    print("ranking")
    resultList = list()
    fit = list()

    for i in range(len(paths)):
        rank = 1
        costi = cost_list(paths[i], speed, mental_list)
        for j in range(len(HistoryList)):
            costj = cost_list(HistoryList[j], speed, mental_list)
            if costi[0] < costj[0] and costi[1] < costj[1] and costi[2] > costj[2] and costi[3] <= costj[3] and costi[4] < costj[4] and costi[5] < costj[5]:
                rank += 1
        resultList.append([paths[i], costi, rank])
    
    for pathOne in resultList:
        s = 1
        for pathTwo in resultList:
            distance = share_R(pathOne[0], pathTwo[0])
            if distance >= sharing:
                s += 1
        fit.append(pathOne[2] / s)
        # print(pathOne[2] / s, "=>", s)
    # for i in range(len(fit)):
    #     print("index: ", i)
    #     print("fit: ", fit[i])
    #     print("rank: ", resultList[i][2])
    # print("----------------------------")
    return fit


def wheel_select(paths, Fit, excellentList):
    print("select")
    selectPaths = list()
    length = len(paths)
    sumFit = sum(Fit)
    q = list() # 累计概率
    for i in range(length):
        if q:
            q.append(q[i - 1] + Fit[i] / sumFit)
        else:
            q.append(Fit[i] / sumFit) 
    # print(len(q))

    for j in range(length):
        r = random.uniform(0, 1) # 生成用于选择的随机数
        for i in range(length):
            if i == 0:
                if 0 <= r <= q[i]:
                    selectPaths.append(paths[i])
            else:
                if q[i - 1] <= r <= q[i]:
                    selectPaths.append(paths[i])
                    if paths[i] not in selectPaths:
                        # selectPaths.append(paths[i])
                        pass
                    else:
                        excellentList.append(paths[i])
    return selectPaths, excellentList


def crossover(crossPaths, Path):
    print("cross")
    childPath = list()
    copyPaths = crossPaths
    # print(len(paths))
    for father in crossPaths:
        length = len(father)
        index = random.randint(1, length - 2)
        crossPoint = father[index]
        mother = random.choice(crossPaths)
        # print("f", father)
        for mother in copyPaths:
            # print("m", mother)
            if mother != father and crossPoint in mother:
                indexMo = mother.index(crossPoint)
                childOne = father[0: index] + mother[indexMo: ]
                childTwo = mother[0: indexMo] + father[index: ]
                # 除环
                while is_loop(childOne):
                    childOne = drop_loop(childOne)
                while is_loop(childTwo):
                    childTwo = drop_loop(childTwo)

                if childOne not in Path and childTwo not in Path and childOne not in childPath and childTwo not in childPath:
                    childPath.append(childOne)
                    childPath.append(childTwo)
                    break
                elif childOne not in Path and childOne not in childPath:
                    childPath.append(childOne)
                    break
                elif childTwo not in Path and childTwo not in childPath:
                    childPath.append(childTwo)
                    break
                else:
                    pass                

        random.shuffle(copyPaths)
    return childPath


def mutation(Path, MRate, allPaths):
    length = len(Path)
    index = 0
    # num = int(length * MRate) + length + 1
    num = int(length * MRate)
    # while len(Path) < num:
    for i in range(num):
        newPath = random.choice(allPaths)
        if newPath not in Path:
            print("mutation")
            index = random.randint(0, length - 1)
            Path.pop(index)
            Path.append(newPath)
            
    return Path


def copy(Path, excellentList):
    temp = random.choice(excellentList)
    if temp not in Path:
        Path.append(temp)
    return Path


def cost_history(Path, speed, mental_list):
    n = len(Path)
    time = 0 # 时间
    length = 0 # 长度
    area = 0 # 面积
    bridge = 0 # 桥
    build = 0 # 建筑评价
    congestion = 0 # 拥挤度评价
    minTime = MAX_FLOAT
    minLength = MAX_FLOAT
    maxArea = 0
    minBridge = MAX_FLOAT
    minBuild = MAX_FLOAT
    minCongestion = MAX_FLOAT
    for path in Path:
        costs = cost_list(path, speed, mental_list)

        time += costs[0]
        if costs[0] < minTime:
            minTime = costs[0]
        
        length += costs[1]
        if costs[1] < minLength:
            minLength = costs[1]
        
        area += costs[2]
        if costs[2] > maxArea:
            maxArea = costs[2]
        
        bridge += costs[3]
        if costs[3] < minBridge:
            minBridge = costs[3]

        build += costs[4]
        if costs[4] < minBuild:
            minBuild = costs[4]

        congestion += costs[5]
        if costs[5] < minCongestion:
            minCongestion = costs[5]
        
    CostHistoryAvg[0].append(time / n)
    CostHistoryAvg[1].append(length / n)
    CostHistoryAvg[2].append(area / n)
    CostHistoryAvg[3].append(bridge / n)
    CostHistoryAvg[4].append(build / n)
    CostHistoryAvg[5].append(congestion / n)
    CostHistoryExt[0].append(minTime)
    CostHistoryExt[1].append(minLength)
    CostHistoryExt[2].append(maxArea)
    CostHistoryExt[3].append(minBridge)
    CostHistoryExt[4].append(minBuild)
    CostHistoryExt[5].append(minCongestion)


def find_Road(start, end, population, minPopulation, gen, CRate, MRate, sharing, speed, mental_list):
    allPaths, num, length, shortPaths = find_all_path(start, end)
    excellentList = list()
    
    print(num)
    Path = inition(population, allPaths, shortPaths)
    print(len(Path))
    
    for i in range(gen):
        print("----------------------------------------------------")
        print("gen", i)
        
        PathResult = list()

        Fit = paretp_ranking(Path, sharing, speed, mental_list)

        # print(Path)
        # print(Fit)
        
        Path, excellentList = wheel_select(Path, Fit, excellentList)

        length = len(Path)
        print(length)
        # if i == (gen-1):
        #     break
        
        crossPaths = random.sample(Path, int(CRate * length))
        childPath = crossover(crossPaths, Path)
        
        newLength = len(childPath)
        print("child", newLength)
        
        Path = Path + childPath
        for path in Path:
            if path not in HistoryList:
                HistoryList.append(path)

        Fit = paretp_ranking(Path, sharing, speed, mental_list)
        while len(Path) > population:
            minFit = min(Fit)
            index = Fit.index(minFit)
            Fit.pop(index)
            Path.pop(index)
        
        length = len(Path)
        print(length)
        
        Path = mutation(Path, MRate, allPaths)
        
        length = len(Path)
        print(length)

        Fit = paretp_ranking(Path, sharing, speed, mental_list)
        for j in range(len(Path)):
            if Fit[j] >= np.mean(Fit):
                # FitResult.append(Fit[j])
                PathResult.append(Path[j])
        cost_history(PathResult, speed, mental_list)
        

        FitResult = paretp_ranking(PathResult, sharing, speed, mental_list)
        FitHistory[0].append(np.mean(FitResult))
        FitHistory[1].append(np.max(FitResult))
        FitHistory[2].append(np.min(FitResult))
        FitHistory[3].append(np.var(FitResult))
        
        length = len(Path)
        print(length)
        print("----------------------------------------------------")
    
    # Fit = paretp_ranking(Path, sharing, speed, mental_list)  
    # Path, excellentList = wheel_select(Path, Fit, excellentList)
    print(len(Path))
    return PathResult
        

####################################################################################################################################################
if __name__ == "__main__":
    # startTime = time.time()
    start = "17"
    end = "203"
    # end = "17"
    population = 120
    minPopulation = 10
    gen = 100
    CRate = 0.7
    MRate = 0.02
    sharing = 0
    speed = 70
    adjTable = read_txt()
    # print(adjTable)
    creat_graph()
    mental_list = [random.uniform(-1, 1) for i in range(5)]
    # print(mental_list)
    Path = find_Road(start, end, population, minPopulation, gen, CRate, MRate, sharing, speed, mental_list)
    print(len(Path))
    # for path in Path:
    #     print(path)

    ### plot ###
    allPaths, num, length, shortPaths = find_all_path(start, end)
    time_cost = [[], []]
    length_cost = [[], []]
    width_cost = [[], []]
    bridge_cost = [[], []]
    build_cost = [[], []]
    congestion_cost = [[], []]
    for i in range(len(allPaths)):
        costs = cost_list(allPaths[i], speed, mental_list)
        if allPaths[i] in Path:
            pass
        else:
            if i % 2 == 0:
                time_cost[1].append(costs[0])
                length_cost[1].append(costs[1])
                width_cost[1].append(costs[2])
                bridge_cost[1].append(costs[3])
                build_cost[1].append(costs[4])
                congestion_cost[1].append(costs[5])
    for i in range(len(Path)):
        costs = cost_list(Path[i], speed, mental_list)
        time_cost[0].append(costs[0])
        length_cost[0].append(costs[1])
        width_cost[0].append(costs[2])
        bridge_cost[0].append(costs[3])
        build_cost[0].append(costs[4])
        congestion_cost[0].append(costs[5])
    fig = plt.figure("relative", figsize=(80, 80))
    
    
    for i in range(6):
        for j in range(6):
            plt.subplot(6, 6, i * 6 + j + 1)
            xlist = list()
            ylist = list()
            xlabel = ""
            ylabel = ""
            if i == 0:
                xlist = time_cost
                xlabel = "time"
            elif i == 1:
                xlist = length_cost
                xlabel = "length"
            elif i == 2:
                xlist = width_cost
                xlabel = "width"
            elif i == 3:
                xlist = bridge_cost
                xlabel = "bridge"
            elif i == 4:
                xlist = build_cost
                xlabel = "building"
            else:
                xlist = congestion_cost
                xlabel = "congestion"
            
            if j == 0:
                ylist = time_cost
                ylabel = "time"
            elif j == 1:
                ylist = length_cost
                ylabel = "length"
            elif j == 2:
                ylist = width_cost
                ylabel = "width"
            elif j == 3:
                ylist = bridge_cost
                ylabel = "bridge"
            elif j == 4:
                ylist = build_cost
                ylabel = "building"
            else:
                ylist = congestion_cost
                ylabel = "congestion"
            plt.scatter(xlist[1], ylist[1], c="blue")
            plt.scatter(xlist[0], ylist[0], c="red")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)

    print(len(time_cost[0]))
    
    fig.savefig(r"figure\relatity.png")
    # # plt.show()
    # ### plot ###
    genList = [i for i in range(len(FitHistory[0]))]
    fig = plt.figure("Fit", figsize=(20, 10))
    # print(FitHistory)
    print(len(FitHistory[0]))
    print(len(genList))
    # FitHistory[0] = np.array(FitHistory[0])
    plt.plot(genList, FitHistory[0], c="blue", linestyle="-", marker="o", linewidth=1, label="mean")
    plt.plot(genList, FitHistory[1], c="red", linestyle="-", marker="^", linewidth=1, label="max")
    plt.plot(genList, FitHistory[2], c="green", linestyle="-", marker="v", linewidth=1, label="min")
    plt.plot(genList, FitHistory[3], c="c", linestyle="-", marker="d", linewidth=1, label="sd")
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.title("Fit")
    fig.savefig(r"figure\Fit.png")
    # plt.show()

    fig = plt.figure("Time", figsize=(20, 10))
    plt.plot(genList, CostHistoryAvg[0], c="blue", linestyle="-", marker="o", linewidth=1, label="mean")
    plt.plot(genList, CostHistoryExt[0], c="red", linestyle="-", marker="d", linewidth=1, label="min")
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.title("Time")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.grid(axis="y")
    fig.savefig(r"figure\Time.png")
    
    fig = plt.figure("Length", figsize=(20, 10))
    plt.plot(genList, CostHistoryAvg[1], c="blue", linestyle="-", marker="o", linewidth=1, label="mean")
    plt.plot(genList, CostHistoryExt[1], c="red", linestyle="-", marker="d", linewidth=1, label="min")
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.title("Length")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.grid(axis="y")
    fig.savefig(r"figure\Length.png")
    
    fig = plt.figure("Width", figsize=(20, 10))
    plt.plot(genList, CostHistoryAvg[2], c="blue", linestyle="-", marker="o", linewidth=1, label="mean")
    plt.plot(genList, CostHistoryExt[2], c="red", linestyle="-", marker="d", linewidth=1, label="max")
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.title("Width")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.grid(axis="y")
    fig.savefig(r"figure\Width.png")

    fig = plt.figure("Bridge", figsize=(20, 10))
    plt.plot(genList, CostHistoryAvg[3], c="blue", linestyle="-", marker="o", linewidth=1, label="mean")
    plt.plot(genList, CostHistoryExt[3], c="red", linestyle="-", marker="d", linewidth=1, label="min")
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.title("Bridge")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.grid(axis="y")
    fig.savefig(r"figure\Bridge.png")

    fig = plt.figure("Build", figsize=(20, 10))
    plt.plot(genList, CostHistoryAvg[4], c="blue", linestyle="-", marker="o", linewidth=1, label="mean")
    plt.plot(genList, CostHistoryExt[4], c="red", linestyle="-", marker="d", linewidth=1, label="min")
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.title("Build")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.grid(axis="y")
    fig.savefig(r"figure\Build.png")

    fig = plt.figure("Congestion", figsize=(20, 10))
    plt.plot(genList, CostHistoryAvg[5], c="blue", linestyle="-", marker="o", linewidth=1, label="mean")
    plt.plot(genList, CostHistoryExt[5], c="red", linestyle="-", marker="d", linewidth=1, label="min")
    plt.legend(loc='upper left', bbox_to_anchor=(0.2, 0.95))
    plt.title("Congestion")
    plt.xlabel("generation")
    plt.ylabel("fitness")
    plt.grid(axis="y")
    fig.savefig(r"figure\Congetion.png")

    f = open(r"figure\parameter.txt", mode='w', encoding="utf-8")
    f.write('起点：' + start + "\n")
    f.write('終点：' + end + "\n")
    f.write('個体数：' + str(population) + "\n")
    f.write('世代数：' + str(gen) + "\n")
    f.write('交叉率：' + str(CRate) + "\n")
    f.write('変異率：' + str(MRate) + "\n")
    f.write("シェアリング：" + str(sharing) + "\n")
    f.close()