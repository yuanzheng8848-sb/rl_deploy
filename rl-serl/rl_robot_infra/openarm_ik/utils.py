

class Center_Expand_Generator:

    def __init__(self, lower, upper):
        self.lower = lower
        self.upper = upper
        self.counter = 0
    
    def set_start(self, start, step):
        
        if start < self.lower or start > self.upper:
            raise ValueError("start value out of range")

        self.value_list = []

        distance = step
        left_done = False
        right_done = False
        
        while not (left_done and right_done):
            # 向右
            right_val = start + distance
            if not right_done and right_val <= self.upper:
                self.value_list.append(right_val)
            else:
                right_done = True
            
            # 向左
            left_val = start - distance
            if not left_done and left_val >= self.lower:
                self.value_list.append(left_val)
            else:
                left_done = True
            
            distance += step
        
        self.counter = 0

    def next(self):
        if self.counter >= len(self.value_list):
            return None
        else:
            value = self.value_list[self.counter]
            self.counter += 1
            return value
    
