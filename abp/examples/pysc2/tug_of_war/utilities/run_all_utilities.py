import torch
import numpy as np
from abp.examples.pysc2.tug_of_war.utilities import graph_action_frequency
from abp.examples.pysc2.tug_of_war.utilities import building_frequencies
from abp.examples.pysc2.tug_of_war.utilities import graph_average_buildings
from abp.examples.pysc2.tug_of_war.utilities import graph_win_loss_sequence
from abp.examples.pysc2.tug_of_war.utilities import graph_win_percentage

def main():
    file_to_load = input("enter the path for desired *.pt file: ")
    data = torch.load(file_to_load)
    data = np.array(data).tolist()

    action_group_length = input("please enter the size you would like to group actions by: ")
    action_group_length = int(action_group_length)

    graph_win_percentage.main(data, action_group_length)
    graph_win_loss_sequence.main(data, action_group_length)
    graph_action_frequency.main(data,action_group_length)
    graph_average_buildings.main(data, action_group_length)
    building_frequencies.main(data, action_group_length)
    

if __name__ == "__main__":
    main()