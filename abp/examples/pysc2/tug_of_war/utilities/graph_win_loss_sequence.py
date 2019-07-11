import torch
import matplotlib.pyplot as plt
import sys
import numpy as np

from abp.examples.pysc2.tug_of_war.utilities import episodes


def main(data, action_group_length):
    
    eps = episodes.Episodes(data, len(data), action_group_length)

    win_loss_sequence = eps.get_win_loss_sequence()
    win_total_line_graph(win_loss_sequence)



def win_total_line_graph(win_total_timeline):
    x_less = []
    x_more = []
    y_less = []
    y_more = []
    zero = []
    x = []
    for i in range(len(win_total_timeline)):
        zero.append(0)
        x.append(i)

    fig1 = plt.figure(num=None, figsize=(15, 10), dpi=200, facecolor='w', edgecolor='k')
    plt.plot(x, win_total_timeline, color='b', linewidth=0.5)
    plt.plot(x, zero, color='r', linewidth=0.5)
    plt.title('Player 1 Win-Loss Sequence\n(Win +1) (Loss -1)') 
    plt.ylabel('Wins for Player 1')
    plt.xlabel('Episodes (Games)')

    plt.show()


if __name__ == "__main__":
    data_file = input("Please enter the file name you want to load: ")
    data = torch.load(data_file)
    data = np.array(data).tolist()
    action_group_length = input("Please enter the size you would like to group actions by: ")
    action_group_length = int(action_group_length)
    main(data,action_group_length)
