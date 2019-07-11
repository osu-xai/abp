import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
import pprint
import operator
import collections

from abp.examples.pysc2.tug_of_war.utilities import episodes


def main(data, action_group_length):

    eps = episodes.Episodes(data, len(data), action_group_length)

    marine_dict, baneling_dict, immortal_dict = eps.get_end_building_frequencies()
    
    sorted_marine_dict = sorted(marine_dict.items(), key=operator.itemgetter(1), reverse=True)
    sorted_baneling_dict = sorted(baneling_dict.items(), key=operator.itemgetter(1), reverse=True)
    sorted_immortal_dict = sorted(immortal_dict.items(), key=operator.itemgetter(1), reverse=True)

    print("========================= Marine Building Frequencies for Final Wave ===========================")
    for i in range(len(sorted_marine_dict)):
        print(str(sorted_marine_dict[i][0]) + "\t: " + str(sorted_marine_dict[i][1]))

    print("======================== Baneling Building Frequencies for Final Wave ==========================")
    for i in range(len(sorted_baneling_dict)):
        print(str(sorted_baneling_dict[i][0]) + "\t: " + str(sorted_baneling_dict[i][1])) 
    
    print("======================== Immortal Building Frequencies for Final Wave ==========================")
    for i in range(len(sorted_immortal_dict)):
        print(str(sorted_immortal_dict[i][0]) + "\t: " + str(sorted_immortal_dict[i][1]))
    
    input("press enter to continue...")



if __name__ == "__main__":
    data_file = input("Please enter the file name you want to load: ")
    data = torch.load(data_file)
    data = np.array(data).tolist()
    action_group_length = input("Please enter the size you would like to group actions by: ")
    action_group_length = int(action_group_length)
    main(data,action_group_length)
