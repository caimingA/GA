from collections import Counter


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


def share_R(pathOne, pathTwo):
    total = len(set(pathOne + pathTwo)) - 2 # 减去起点和终点
    print("total", total)
    overlap = len(pathOne + pathTwo) - 4 - total
    print(len(pathOne + pathTwo))
    print("overlap", overlap)
    return overlap / total


path = ['0', '1', '2', '5', '4', '1', '2', '4', '7', '4', '8']

while is_loop(path):
    path = drop_loop(path)
    print(path)

pathOne = ['0', '1', '2', '4', '8']
pathTwo = ['0', '1', '2', '4', '8']

print(share_R(pathOne, pathTwo))

cost_list = [
                  length / speed
                , length
                , area / length
                , bridge
                , build / area
                , congestion / area
                ]