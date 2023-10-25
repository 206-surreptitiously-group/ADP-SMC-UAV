import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    POS = False
    if POS:
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
    else:
        test_record1 = pd.read_csv('../datasave/nets/att_maybe_optimal1/test_record.csv', header=0).to_numpy()
        l1, test1 = test_record1[:, 0], test_record1[:, 1]
        sumr_list1 = pd.read_csv('../datasave/nets/att_maybe_optimal1/sumr_list.csv', header=0).to_numpy()
        s1, sumr1 = sumr_list1[:, 0], sumr_list1[:, 1]

        len_l1 = len(l1)
        len_s1 = len(s1)
        rs1 = 0.0
        rt1 = 0.5

        l1 = l1[int(rs1 * len_l1): int(rt1 * len_l1)]
        test1 = test1[int(rs1 * len_l1): int(rt1 * len_l1)]
        s1 = s1[int(rs1 * len_s1): int(rt1 * len_s1)]
        sumr1 = sumr1[int(rs1 * len_s1): int(rt1 * len_s1)]

        test_record2 = pd.read_csv('../datasave/nets/att_maybe_optimal2/test_record.csv', header=0).to_numpy()
        l2, test2 = test_record2[:, 0], test_record2[:, 1]
        sumr_list2 = pd.read_csv('../datasave/nets/att_maybe_optimal2/sumr_list.csv', header=0).to_numpy()
        s2, sumr2 = sumr_list2[:, 0], sumr_list2[:, 1]

        len_l2 = len(l2)
        len_s2 = len(s2)
        rs2 = 0.0
        rt2 = 1.0

        l2 = l2[int(rs2 * len_l2): int(rt2 * len_l2)]
        test2 = test2[int(rs2 * len_l2): int(rt2 * len_l2)]
        s2 = s2[int(rs2 * len_s2): int(rt2 * len_s2)]
        sumr2 = sumr2[int(rs2 * len_s2): int(rt2 * len_s2)]

        plt.figure()
        plt.plot(l1, test1, 'r')
        plt.plot(l2 + len(l1), test2, 'g')

        plt.figure()
        plt.plot(s1, sumr1, 'r')
        plt.plot(s2 + len(s1), sumr2, 'g')

    plt.show()
