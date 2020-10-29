class Human:
    def __init__(self, name, speed, origin):
        self.name = name
        self.speed = speed
        self.origin = origin
        self.isCompute = 1
        self.finish = 0
        self.start = origin
        self.end = origin
        self.arrive = 0
        self.time = 0
        self.length = 0
        self.output = 0

        self.history = list()
        self.allPath = list()
        self.path = list()
        self.allCosts = list()
        self.initPaths = list()
        self.costs = list()
        self.Paths = list()
        self.Costs = list()

    def get_isCompute(self):
        return self.isCompute

    def get_end(self):
        return self.end

    def set_end(self, end):
        self.end = end

    def is_compute(self):
        if self.speed + self.finish > self.length:
            self.isCompute = 1
            return 1
        else:
            self.isCompute = 0
            return 0

    def update_start_end(self, start, end):
        self.start = start
        self.end = end
        self.isCompute = 0

    def set_finish(self, finish):
        self.finish = finish


class Road:
    def __init__(self, begin, end, width, length):
        self.begin = begin
        self.end = end
        self.width = width
        self.length = length

