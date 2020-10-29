from collections import Counter

def is_loop(path):
    print(path)
    counter = dict(Counter(path))
    temp = [key for key, value in counter.items()if value > 1]
    print(temp)
    while temp:
        i = temp[0]
            # print("a")
        b = list()
        for index, nums in enumerate(path):
            if nums == i:
                b.append(index)

        print("index:", b)
            # print()
        path = path[:b[0]] + path[b[1]:]
        counter = dict(Counter(path))
        temp = [key for key, value in counter.items() if value > 1]
        print(path)
        print(temp)

    return path


path = ['0', '1', '2', '5', '4', '1', '2', '4', '7', '4', '8']

print(is_loop(path))