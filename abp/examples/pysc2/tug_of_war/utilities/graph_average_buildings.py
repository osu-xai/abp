import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

from abp.examples.pysc2.tug_of_war.utilities import episodes


def main():
    # data_file = input("Please enter the file name you want to load:\t")
    # data = torch.load(data_file)
    data = torch.load('../test_random_vs_random_2l.pt')
    data = np.array(data).tolist()
    
    eps = episodes.Episodes(data, len(data), 1)
    p1_buildings = eps.get_p1_buildings()
    p2_buildings = eps.get_p2_buildings()
    p1_win_total = eps.get_total_p1_wins()
    p2_win_total = eps.get_episode_count() - p1_win_total

    average_case_graph(p1_buildings, p2_buildings, p1_win_total, p2_win_total)
    


def average_case_graph(p1_buildings, p2_buildings, p1_win_total, p2_win_total):
    labels = 'Marines', 'Banelings', 'Immortals'
    #create averages
    if(p1_win_total != 0):
        sizes_ally_winning = [p1_buildings[3]/p1_win_total, p1_buildings[4]/p1_win_total, p1_buildings[5]/p1_win_total]
        sizes_enemy_losing = [p2_buildings[0]/p1_win_total, p2_buildings[1]/p1_win_total, p2_buildings[2]/p1_win_total]
    
    if(p2_win_total != 0):
        sizes_ally_losing = [p1_buildings[0]/p2_win_total, p1_buildings[1]/p2_win_total, p1_buildings[2]/p2_win_total]
        sizes_enemy_winning = [p2_buildings[3]/p2_win_total, p2_buildings[4]/p2_win_total, p2_buildings[5]/p2_win_total]

    colors = ['gold', 'yellowgreen', 'lightcoral']
    explode = (0.1, 0, 0, 0)  # explode 1st slice

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
        return my_autopct

    # Plot
    if (p1_win_total != 0):
        fig1 = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
        ally_winner = fig1.add_axes([.1, .5, .35, .35], aspect=1)
        ally_winner.pie(sizes_ally_winning, labels=labels, colors=colors,
        autopct=make_autopct(sizes_ally_winning), shadow=True, startangle=140)
        plt.title('Ally Average Buildings (Winner)')

    if (p2_win_total != 0):
        ally_loser = fig1.add_axes([.1, .1, .35, .35], aspect=1)
        ally_loser.pie(sizes_ally_losing, labels=labels, colors=colors,
        autopct=make_autopct(sizes_ally_losing), shadow=True, startangle=140)
        plt.title('Ally Average Buildings (Losing)')

        enemy_winning = fig1.add_axes([.5, .5, .35, .35], aspect=1)
        enemy_winning.pie(sizes_enemy_winning, labels=labels, colors=colors,
        autopct=make_autopct(sizes_enemy_winning), shadow=True, startangle=140)
        plt.title('Enemy Average Buildings (Winner)')

    if (p1_win_total != 0):
        enemy_losing = fig1.add_axes([.5, .1, .35, .35], aspect=1)
        enemy_losing.pie(sizes_enemy_losing, labels=labels, colors=colors,
        autopct=make_autopct(sizes_enemy_losing), shadow=True, startangle=140)
        plt.title('Enemy Average Buildings (Losing)')

    fig1.suptitle('Average End State of Games by Winning Player and Losing Player', fontsize=16)

    plt.show()
    plt.close()


if __name__ == "__main__":
    main()
