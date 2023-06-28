import numpy as np
import matplotlib.pyplot as plt


# dx = 2x + u
if __name__ == '__main__':
    a = np.array([1, 2, 3, 4])
    b = np.array([0.1, 0.2, 0.3, 0.4])
    # a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # b = np.zeros((3, 3))
    # c = [a, b]
    # print(c)
    # c[1][:] = c[0][:]
    # print(c)
    # c[0][0][0]=100
    # print(c)
    print(a * b)