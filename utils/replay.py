import numpy as np
from collections import deque


class ExperienceReplay:
    def __init__(self, size):
        self.replay = deque(maxlen=size)
        self.size = size
        
    def add(self, x):
        self.replay.append(x)
    
    def random_sample(self, batch_size):
        '''
        Returns a list of length batch_size containing observations
        '''
        s0, a, r, s1, d = [], [], [], [], []
        for i in range(batch_size):
            idx = np.random.randint(0, min(self.size, len(self.replay)))
            s0.append(self.replay[idx][0])
            a.append(self.replay[idx][1])
            r.append(self.replay[idx][2])
            s1.append(self.replay[idx][3])
            d.append(self.replay[idx][4])
            
        s0, a, r, s1, d = np.array(s0), np.array(a), np.array(r), np.array(s1), np.array(d)
        return s0, a, r, s1, d