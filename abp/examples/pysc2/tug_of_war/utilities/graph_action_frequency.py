import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import pprint
import operator
import collections

from abp.examples.pysc2.tug_of_war.utilities import episodes


def main():
    # data_file = input("Please enter the file name you want to load:\t")
    # data = torch.load(data_file)
    data = torch.load('../test_random_vs_random_2l.pt')
    data = np.array(data).tolist()
    
    eps = episodes.Episodes(data, len(data), 1)
    move_set = eps.get_move_set()

    sorted_move_set = sorted(move_set.items(), key=operator.itemgetter(1), reverse=True)
    #sorted_refined_move_set = [i for i in sorted_move_set if i[1] >= 10]

    





if __name__ == "__main__":
    main()
