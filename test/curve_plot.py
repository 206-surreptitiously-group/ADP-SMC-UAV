import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    test_record1 = pd.read_csv('../datasave/nets/position_tracking/data/phase1/test_record.csv', header=0).to_numpy()
    test_record2 = pd.read_csv('../datasave/nets/position_tracking/data/phase2/test_record.csv', header=0).to_numpy()
    test_record3 = pd.read_csv('../datasave/nets/position_tracking/data/phase3/test_record.csv', header=0).to_numpy()

    l1, test1 = test_record1[:, 0], test_record1[:, 1]
    l2, test2 = test_record2[:, 0], test_record2[:, 1]
    l3, test3 = test_record3[:, 0], test_record3[:, 1]

    sumr_list1 = pd.read_csv('../datasave/nets/position_tracking/data/phase1/sumr_list.csv', header=0).to_numpy()
    sumr_list2 = pd.read_csv('../datasave/nets/position_tracking/data/phase2/sumr_list.csv', header=0).to_numpy()
    sumr_list3 = pd.read_csv('../datasave/nets/position_tracking/data/phase3/sumr_list.csv', header=0).to_numpy()

    s1, sumr1 = sumr_list1[:, 0], sumr_list1[:, 1]
    s2, sumr2 = sumr_list2[:, 0], sumr_list2[:, 1]
    s3, sumr3 = sumr_list3[:, 0], sumr_list3[:, 1]

    plt.figure()
    plt.plot(l1, test1, 'r')
    plt.plot(l2 + len(l1), test2, 'g')
    plt.plot(l3 + len(l1) + len(l2), test3, 'b')

    plt.figure()
    plt.plot(s1, sumr1, 'r')
    plt.plot(s2 + len(s1), sumr2, 'g')
    plt.plot(s3 + len(s1) + len(s2), sumr3, 'b')

    plt.show()
