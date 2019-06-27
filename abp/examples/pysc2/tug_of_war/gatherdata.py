import torch
import numpy as np
import matplotlib.pyplot as plt
import pprint

a_mar = 0
a_vik = 1
a_col = 2
a_pyl = 3
a_nex = 4
e_mar = 5
e_vik = 6
e_col = 7
e_pyl = 8
e_nex = 9

def main():
    data = torch.load('sadq_v_sadq.pt')
    data = np.array(data).tolist()

    #print_ally_episode_win, print_enemy_episode_win, show_win_percentage, show_average_cases, i_lower, i_upper, show_win_timeline = get_options(len(data)-1)
    #forrealls##ally_wins, enemy_wins, sum_ally_units_win, sum_enemy_units_win, win_total_timeline = gather_data(data, print_ally_episode_win, print_enemy_episode_win, show_win_percentage, show_average_cases, i_lower, i_upper, show_win_timeline)
    ally_wins, enemy_wins, sum_ally_units_win, sum_enemy_units_win, win_total_timeline, strategies = gather_data(data, 0, 0, 0, 0, 0, 11145, 1)

    # if(show_win_percentage):
    #     win_percentage_graph(ally_wins, enemy_wins)
    # if(show_win_timeline):
    #     win_total_line_graph(win_total_timeline)
    # if(show_average_cases):
    #     average_case_graph(sum_ally_units_win, ally_wins, sum_enemy_units_win, enemy_wins)

def gather_data(data, print_ally_episode_win, print_enemy_episode_win, show_win_percentage, show_average_cases, i_lower, i_upper, show_win_timeline):
    ally_wins = 0
    enemy_wins = 0
    episodes = 0


    sum_ally_units_win = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    sum_enemy_units_win = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #0-4 ally: mar, vik, col, pyl, nexus, 5-9 enemy: mar, vik, col, pyl, nexus
    win_total_timeline = [0]
    strategies ={  }
                

    for i in range(i_lower, i_upper):
        player_1_input = ""
        strategies_current = ""
        if (i == 0):
            for n in range(4):
                for o in range(4):
                    player_1_input += (str(data[i+n][0][o]) + ",")
            strategies.update({player_1_input : 1})
            
        elif (((data[i][0][a_mar]) + (data[i][0][a_vik]) + (data[i][0][a_col]) + (data[i][0][a_pyl]) + (data[i][0][e_mar]) + (data[i][0][e_vik]) + (data[i][0][e_col]) + (data[i][0][e_pyl])) < ((data[i-1][0][a_mar]) + (data[i-1][0][a_vik]) + (data[i-1][0][a_col]) + (data[i-1][0][a_pyl]) + (data[i-1][0][e_mar]) + (data[i-1][0][e_vik]) + (data[i-1][0][e_col]) + (data[i-1][0][e_pyl]))):               
            episodes += 1
           
            for n in range(4):
                for o in range(4):
                    strategies_current += (str(data[(i+1)+n][0][o]) + ",")

            if (strategies_current in strategies):
                current_count = strategies[strategies_current]
                strategies[strategies_current] = current_count + 1

            else:
                strategies.update({strategies_current : 1})

            if (data[i-1][0][a_nex]) > (data[i-1][0][e_nex]):
                if (print_ally_episode_win):
                    print_episode_end_state_and_next_state(i-1,data) 
                    ally_graph(i-1,data)
                
                ally_wins += 1
                win_total_timeline.append((win_total_timeline[len(win_total_timeline)-2]) + 1)
                
                for x in range(0,10):
                    sum_ally_units_win[x] = data[i-1][1][x] + sum_ally_units_win[x]
            
            
            elif (data[i-1][0][a_nex]) < (data[i-1][0][e_nex]):
                if (print_enemy_episode_win):
                    print_episode_end_state_and_next_state(i-1,data) 
                    enemy_graph(i-1,data)
                
                enemy_wins += 1
                win_total_timeline.append((win_total_timeline[len(win_total_timeline)-2])-1)

                for x in range(0,10):
                    sum_enemy_units_win[x] = data[i-1][1][x] + sum_enemy_units_win[x]
    
    print("_______________________________________________________________________")
    print("------------- Player 1 First Four Moves Frequency Table ---------------")
    pprint.pprint(strategies)

    return ally_wins, enemy_wins, sum_ally_units_win, sum_enemy_units_win, win_total_timeline, strategies



def get_options(data_range):
    print_ally_episode_win = -1
    print_enemy_episode_win = -1
    show_win_percentage = -1
    show_average_cases = -1
    show_win_timeline = -1

    cases_hi = -1
    cases_lo = -1

    while(cases_lo >= cases_hi or cases_lo < 0):
        cases_lo = input("Select a lower bound for data to look at (0 - " + str(data_range) + "):\t")
        cases_hi = input("Select an upper bound for data to look at (0 - " + str(data_range) + "):\t")
        try:
            val = int(cases_lo)
            val1 = int(cases_hi)
        except ValueError:
            print("That's not an int!")
            cases_lo = -1
            cases_hi = -1
        cases_hi = int(cases_hi)
        cases_lo = int(cases_lo)

    while(print_ally_episode_win != '0' and print_ally_episode_win != '1'):
        print_ally_episode_win = input("Do you want to print ally win graphs every episode? (1 - 0):\t")

    while(print_enemy_episode_win != '0' and print_enemy_episode_win != '1'):
        print_enemy_episode_win = input("Do you want to print enemy win graphs every episode? (1 - 0):\t")

    while(show_win_percentage != '0' and show_win_percentage != '1'):
        show_win_percentage = input("Do you want to see win percentages for each player? (1 - 0):\t")
    
    while(show_win_timeline != '0' and show_win_timeline != '1'):
        show_win_timeline = input("Do you want to see a timeline of wins and loses? (1 - 0):\t")

    while(show_average_cases != '0' and show_average_cases != '1'):
        show_average_cases = input("Do you want to have the average cases reported? (1 - 0):\t")

    return int(print_ally_episode_win), int(print_enemy_episode_win), int(show_win_percentage), int(show_average_cases), int(cases_lo), int(cases_hi), int(show_win_timeline)




def print_episode_end_state_and_next_state(i,data):
    print("-------------------------------------------------------------------------------------------------------------------------")
    print("i:\t" + str(i) + "\t\tfriendly nexus: " + str(data[i][0][a_nex]) + "\t\tenemey nexus: " + str(data[i][0][e_nex]))
    print("i+1:\t" + str(i+1) + "\t\tfriendly nexus: " + str(data[i+1][0][4]) + "\t\tenemey nexus: " + str(data[i+1][0][9]))
    print("\tmarine: " + str(data[i][0][a_mar]) + "\tvikings: " + str(data[i][0][a_vik]) + "\tcolossus: " + str(data[i][0][a_col]) + "\tpylons: " + str(data[i][0][a_pyl]) + "\tE marine: " + str(data[i][0][e_mar]) + "\tE vikings: " + str(data[i][0][e_vik]) + "\tE colossus: " + str(data[i][0][e_col]) + "\tE pylons: " + str(data[i][0][e_pyl]))
    print("\tmarine: " + str(data[i+1][0][a_mar]) + "\tvikings: " + str(data[i+1][0][a_vik]) + "\tcolossus: " + str(data[i+1][0][a_col]) + "\tpylons: " + str(data[i+1][0][a_pyl]) + "\tE marine: " + str(data[i+1][0][e_mar]) + "\tE vikings: " + str(data[i+1][0][e_vik]) + "\tE colossus: " + str(data[i+1][0][e_col]) + "\tE pylons: " + str(data[i+1][0][e_pyl]))
    print("-------------------------------------------------------------------------------------------------------------------------")


def average_case_graph(sum_ally_units_win, ally_wins, sum_enemy_units_win, enemy_wins):
    labels = 'Marines', 'Vikings', 'Colossus', 'Pylons'
    if(ally_wins != 0):
        sizes_ally_winning = [sum_ally_units_win[a_mar]/ally_wins, sum_ally_units_win[a_vik]/ally_wins, sum_ally_units_win[a_col]/ally_wins, sum_ally_units_win[a_pyl]/ally_wins]
        sizes_enemy_losing = [sum_ally_units_win[e_mar]/ally_wins, sum_ally_units_win[e_vik]/ally_wins, sum_ally_units_win[e_col]/ally_wins, sum_ally_units_win[e_pyl]/ally_wins]
    if(enemy_wins != 0):
        sizes_ally_losing = [sum_enemy_units_win[a_mar]/enemy_wins, sum_enemy_units_win[a_vik]/enemy_wins, sum_enemy_units_win[a_col]/enemy_wins, sum_enemy_units_win[a_pyl]/enemy_wins]
        sizes_enemy_winning = [sum_enemy_units_win[e_mar]/enemy_wins, sum_enemy_units_win[e_vik]/enemy_wins, sum_enemy_units_win[e_col]/enemy_wins, sum_enemy_units_win[e_pyl]/enemy_wins]

    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    explode = (0.1, 0, 0, 0)  # explode 1st slice

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
        return my_autopct

    # Plot
    if (ally_wins != 0):
        fig1 = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
        ally_winner = fig1.add_axes([.1, .5, .35, .35], aspect=1)
        ally_winner.pie(sizes_ally_winning, labels=labels, colors=colors,
        autopct=make_autopct(sizes_ally_winning), shadow=True, startangle=140)
        plt.title('Ally Average (Winner)\nNexus Health (' + str(int(sum_ally_units_win[4]/ally_wins)) + ')')

    if (enemy_wins != 0):
        ally_loser = fig1.add_axes([.1, .1, .35, .35], aspect=1)
        ally_loser.pie(sizes_ally_losing, labels=labels, colors=colors,
        autopct=make_autopct(sizes_ally_losing), shadow=True, startangle=140)
        plt.title('Ally Average (Losing)\nNexus Health (' + str(int(sum_enemy_units_win[4]/enemy_wins)) + ')')

        enemy_winning = fig1.add_axes([.4, .5, .35, .35], aspect=1)
        enemy_winning.pie(sizes_enemy_winning, labels=labels, colors=colors,
        autopct=make_autopct(sizes_enemy_winning), shadow=True, startangle=140)
        plt.title('Enemy Average (Winner)\nNexus Health (' + str(int(sum_enemy_units_win[9]/enemy_wins)) + ')')

    if (ally_wins != 0):
        enemy_losing = fig1.add_axes([.4, .1, .35, .35], aspect=1)
        enemy_losing.pie(sizes_enemy_losing, labels=labels, colors=colors,
        autopct=make_autopct(sizes_enemy_losing), shadow=True, startangle=140)
        plt.title('Enemy Average (Losing)\nNexus Health (' + str(int(sum_ally_units_win[9]/ally_wins)) + ')')

    fig1.suptitle('Average End State of Games by Winning Player and Losing Player', fontsize=16)

    plt.show()
    plt.close()


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
        if win_total_timeline[i] <= 0:
            y_less.append(win_total_timeline[i])
            x_less.append(i)

        else:
            y_more.append(win_total_timeline[i])
            x_more.append(i)

    fig1 = plt.figure(num=None, figsize=(15, 10), dpi=200, facecolor='w', edgecolor='k')
    plt.plot(x, win_total_timeline, color='b', linewidth=0.5)
    plt.plot(x, zero, color='r', linewidth=0.5)
    plt.title('Ally Win-Loss Sequence\n(Win +1) (Loss -1)') 


    plt.show()



def win_percentage_graph(ally_wins, enemy_wins):
    labels = 'Ally', 'Enemy'
    sizes = [ally_wins, enemy_wins]
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





def ally_graph(i,data):

    labels = 'Marines', 'Vikings', 'Colossus', 'Pylons'
    sizes_ally = [data[i][0][0], data[i][0][1], data[i][0][2], data[i][0][3]]
    sizes_enemy = [data[i][0][5], data[i][0][6], data[i][0][7], data[i][0][8]]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    explode = (0.1, 0, 0, 0)  # explode 1st slice

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
        return my_autopct

    # Plot
    fig1 = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    ally = fig1.add_axes([-.1, .1, .7, .7], aspect=1)
    ally.pie(sizes_ally, explode=explode, labels=labels, colors=colors,
    autopct=make_autopct(sizes_ally), shadow=True, startangle=140)
    plt.title('Ally (Winner)')
    enemy = fig1.add_axes([.4, .1, .7, .7], aspect=1)
    enemy.pie(sizes_enemy, explode=explode, labels=labels, colors=colors,
    autopct=make_autopct(sizes_enemy), shadow=True, startangle=140)
    plt.title('Enemy (Loser)')

    plt.show()
    plt.close()




def enemy_graph(i,data):

    labels = 'Marines', 'Vikings', 'Colossus', 'Pylons'
    sizes_ally = [data[i][0][0], data[i][0][1], data[i][0][2], data[i][0][3]]
    sizes_enemy = [data[i][0][5], data[i][0][6], data[i][0][7], data[i][0][8]]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']
    explode = (0.1, 0, 0, 0)  # explode 1st slice

    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
        return my_autopct

    # Plot
    fig1 = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')

    enemy = fig1.add_axes([.4, .1, .7, .7], aspect=1)
    enemy.pie(sizes_enemy, explode=explode, labels=labels, colors=colors,
    autopct=make_autopct(sizes_enemy), shadow=True, startangle=140)
    plt.title('Enemy (Winner)')

    ally = fig1.add_axes([-.1, .1, .7, .7], aspect=1)
    ally.pie(sizes_ally, explode=explode, labels=labels, colors=colors,
    autopct=make_autopct(sizes_ally), shadow=True, startangle=140)
    plt.title('Ally (Loser)')

    plt.show()
    plt.close()

if __name__ == "__main__":
    main()
