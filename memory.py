from collections import deque
import random
import numpy as np

class DataMemory():

    def __init__(self, max_len=50000):
        super(DataMemory, self).__init__()

        self.max_len = max_len
        self.memory = deque()

    def add(self, s_t, a_t, r_t, s_n, terminal):
        self.memory.append((s_t, a_t, r_t, s_n, float(terminal)))

        if len(self.memory) > self.max_len:
            self.memory.popleft()

    def gen_minibatch(self, batch_size=32):
        minibatch = random.sample(self.memory, batch_size)

        s_t, a_t, r_t, s_n, terminal = zip(*minibatch)

        s_t = np.concatenate(s_t)
        s_n = np.concatenate(s_n)
        r_t = np.array(r_t).astype(np.float32).reshape(batch_size, 1)
        terminal = np.array(terminal).astype(np.float32).reshape(batch_size, 1)

        return s_t, a_t, r_t, s_n, terminal
