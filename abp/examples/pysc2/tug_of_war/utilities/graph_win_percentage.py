import sys
import torch
import matplotlib.pyplot as plt
import numpy as np

from abp.examples.pysc2.tug_of_war import episodes


def main():
    # data_file = input("Please enter the file name you want to load:\t")
    # data = torch.load(data_file)
    data = torch.load('../test_random_vs_random_2l.pt')
    data = np.array(data).tolist()
    
    eps = episodes.Episodes(data, len(data), 1)
    total_p1_wins = eps.get_total_p1_wins()
    num_of_eps = eps.get_episode_count()
    total_p2_wins = num_of_eps - total_p1_wins
    win_percentage_graph(total_p1_wins, total_p2_wins)



def win_percentage_graph(p1_wins, p2_wins):
    labels = 'Ally', 'Enemy'
    sizes = [p1_wins, p2_wins]
    colors = ['blue', 'red']
    explode = (0.1, 0, 0, 0)  # explode 1st slice

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
        return my_autopct

    # Plot
    fig1 = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    ally = fig1.add_axes([.1, .1, .7, .7], aspect=1)
    ally.pie(sizes, labels=labels, colors=colors,
    autopct=make_autopct(sizes), shadow=True, startangle=140)
    plt.title('Ally Wins vs. Enemy Wins') 


    plt.show()
    plt.close()


if __name__ == "__main__":
    main()