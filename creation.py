import random
if __name__ == '__main__':
    f = open('routeTest.txt', mode='w')
    for i in range(200):
        start = random.randint(0, 60)
        end = start
        while end == start:
            end = random.randint(0, 60)
        width = random.randint(10, 30)
        length = random.randint(100, 1000)
        f.write(str(start) + ',' + str(end) + ',' + str(width) + ',' + str(length) + '\n')
        if i == 199:
            f.write(str(end) + ',' + str(start) + ',' + str(width) + ',' + str(length))
        else:
            f.write(str(end) + ',' + str(start) + ',' + str(width) + ',' + str(length) + '\n')
    f.close()
