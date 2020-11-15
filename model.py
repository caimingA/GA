class Human:
    def __init__(self, name, speed, origin):
        self.name = name # 避难者番号
        self.speed = speed # 速度
        self.origin = origin # 起点
        self.isCompute = 1 # 是否要进行计算
        self.finish = 0 # 是否完成了路线
        self.start = origin # 开始的起点
        self.end = origin # 终点
        self.arrive = 0 # 是否完成道路？
        self.time = 0 # 花费时间
        self.length = 0 # ？
        self.output = 0 # ？

        self.history = list() # 走过的点
        self.allPath = list() # ？
        self.path = list() # 接下来要走的道路
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


