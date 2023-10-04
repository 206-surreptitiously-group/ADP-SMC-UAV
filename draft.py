import numpy as np
import matplotlib.pyplot as plt


# dx = 2x + u
if __name__ == '__main__':
    a = np.array([1, 2, 3, 4])
    b = np.array([2, 2, 6, 6])
    print(np.clip(a, -b, b))
