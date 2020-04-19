import numpy as np
import torch

a = 303

def adjustMaxSize(self, a):
    if a % 16 == 0 and (a /16 - 4)%3 == 0:
        pass
    else:
        while (a % 16 != 0 or (a /16 - 4)%3 != 0):
            a +=1
    return a
# print(a)


# if a % 16 == 0:
#     print("pass")
# elif a % 16 == 1:


a = np.array([1, 2, 3])
a = a.reshape(len(a), 1)
print(a.shape)
a = a.reshape(a.shape[1], a.shape[0])
print(a.shape)

# l = list()
# l.append(1)
# l.append(2)
# print(l)
# l = np.asarray(l)
# print(type(l))
# print(l)

# max_len = 10
# padded = np.zeros((max_len), dtype=np.int64)
# if len(l) > max_len: 
#     padded[:] = l[:max_len]
# else: 
#     padded[:len(l)] = l
# print(padded)

# def max_length(tensor):
#     return max(len(t) for t in tensor)
# print(max_length(l))