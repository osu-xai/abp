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
    
    move_set = eps.get_binned_move_sets()
    #sorted_refined_move_set = [i for i in sorted_move_set if i[1] >= 10]
    sorted_move_set = []
    for i in range(len(move_set)):
        if(move_set[i]):
            sorted_move_set.append(sorted(move_set[i].items(), key=operator.itemgetter(1), reverse=True))
            print("|================================================= " + "Wave " + str(i) + " - " + str(i+action_group_length-1) + " ================================================|")
            for j in range(len(sorted_move_set[i])):
                print(sorted_move_set[i][j])
    
    input("press enter to continue...")


# def graph_frequencies(move_set):
#     N = 40
#     f1=f2=f3=f4=f5=f6=f7=f8=f9=f10=f11=f12=f13=f14=f15=f16=f17=f18=f19=f20=f21=f22=f23=f24=f25=f26=f27=f28=f29=f30=f31=f32=f33=f34=f35=f36=f37=f38=f39=f40=[]
#     for i in range(len(move_set)):
#         for j in range(len(move_set[i])):
#             if j == 0:
#                 f1.append(move_set[i][j][1])
#             elif j == 1:
#                 f2.append(move_set[i][j][1])
#             elif j == 2:
#                 f3.append(move_set[i][j][1])
#             elif j == 3:
#                 f4.append(move_set[i][j][1])
#             elif j == 4:
#                 f5.append(move_set[i][j][1])
#             elif j == 5:
#                 f6.append(move_set[i][j][1])
#             elif j == 6:
#                 f7.append(move_set[i][j][1])
#             elif j == 7:
#                 f8.append(move_set[i][j][1])
#             elif j == 8:
#                 f9.append(move_set[i][j][1])
#             elif j == 9:
#                 f10.append(move_set[i][j][1])
#             elif j == 10:
#                 f11.append(move_set[i][j][1])
#             elif j == 11:
#                 f12.append(move_set[i][j][1])
#             elif j == 12:
#                 f13.append(move_set[i][j][1])
#             elif j == 13:
#                 f14.append(move_set[i][j][1])
#             elif j == 14:
#                 f15.append(move_set[i][j][1])
#             elif j == 15:
#                 f16.append(move_set[i][j][1])
#             elif j == 16:
#                 f17.append(move_set[i][j][1])
#             elif j == 17:
#                 f18.append(move_set[i][j][1])
#             elif j == 18:
#                 f19.append(move_set[i][j][1])
#             elif j == 19:
#                 f20.append(move_set[i][j][1])
#             elif j == 20:
#                 f21.append(move_set[i][j][1])
#             elif j == 21:
#                 f22.append(move_set[i][j][1])
#             elif j == 22:
#                 f23.append(move_set[i][j][1])
#             elif j == 23:
#                 f24.append(move_set[i][j][1])
#             elif j == 24:
#                 f25.append(move_set[i][j][1])
#             elif j == 25:
#                 f26.append(move_set[i][j][1])
#             elif j == 26:
#                 f27.append(move_set[i][j][1])
#             elif j == 27:
#                 f28.append(move_set[i][j][1])
#             elif j == 28:
#                 f29.append(move_set[i][j][1])
#             elif j == 29:
#                 f30.append(move_set[i][j][1])
#             elif j == 30:
#                 f31.append(move_set[i][j][1])
#             elif j == 31:
#                 f32.append(move_set[i][j][1])
#             elif j == 32:
#                 f33.append(move_set[i][j][1])
#             elif j == 33:
#                 f34.append(move_set[i][j][1])
#             elif j == 34:
#                 f35.append(move_set[i][j][1])
#             elif j == 35:
#                 f36.append(move_set[i][j][1])
#             elif j == 36:
#                 f37.append(move_set[i][j][1])
#             elif j == 37:
#                 f38.append(move_set[i][j][1])
#             elif j == 38:
#                 f39.append(move_set[i][j][1])
#             elif j == 39:
#                 f40.append(move_set[i][j][1])

#     ind = np.arange(N)    # the x locations for the groups
#     width = 0.01       # the width of the bars: can also be len(x) sequence
#     fig1 = plt.figure(num=None, figsize=(16, 10), dpi=100, facecolor='w', edgecolor='k')
#     print(f1)
#     p1= plt.bar(len(f1), f1 ,width)
#     p2= plt.bar(len(f2), f2 ,width)
#     p3= plt.bar(len(f3), f3 ,width)
#     # p4= plt.bar(len(f4), f4 ,width)
#     # p5= plt.bar(len(f5), f5 ,width)
#     # p6= plt.bar(len(f6), f6 ,width)
#     # p7= plt.bar(len(f7), f7 ,width)
#     # p8= plt.bar(len(f8), f8 ,width)
#     # p9= plt.bar(len(f9), f9 ,width)
#     # p10= plt.bar(len(f10), f10 ,width)
#     # p11= plt.bar(len(f11), f11 ,width)
#     # p12= plt.bar(len(f12), f12 ,width)
#     # p13= plt.bar(len(f13), f13 ,width)
#     # p14= plt.bar(len(f14), f14 ,width)
#     # p15= plt.bar(len(f15), f15 ,width)
#     # p16= plt.bar(len(f16), f16 ,width)
#     # p17= plt.bar(len(f17), f17 ,width)
#     # p18= plt.bar(len(f18), f18 ,width)
#     # p19= plt.bar(len(f19), f19 ,width)
#     # p20= plt.bar(len(f20), f20 ,width)
#     # p21= plt.bar(len(f21), f21 ,width)
#     # p22= plt.bar(len(f22), f22 ,width)
#     # p23= plt.bar(len(f23), f23 ,width)
#     # p24= plt.bar(len(f24), f24 ,width)
#     # p25= plt.bar(len(f25), f25 ,width)
#     # p26= plt.bar(len(f26), f26 ,width)
#     # p27= plt.bar(len(f27), f27 ,width)
#     # p28= plt.bar(len(f28), f28 ,width)
#     # p29= plt.bar(len(f29), f29 ,width)
#     # p30= plt.bar(len(f30), f30 ,width)
#     # p31= plt.bar(len(f31), f31 ,width)
#     # p32= plt.bar(len(f32), f32 ,width)
#     # p33= plt.bar(len(f33), f33 ,width)
#     # p34= plt.bar(len(f34), f34 ,width)
#     # p35= plt.bar(len(f35), f35 ,width)
#     # p36= plt.bar(len(f36), f36 ,width)
#     # p37= plt.bar(len(f37), f37 ,width)
#     # p38= plt.bar(len(f38), f38 ,width)
#     # p39= plt.bar(len(f39), f39 ,width)
#     # p40= plt.bar(len(f40), f40 ,width)

#     # plt.ylabel('Scores')
#     # plt.title('Scores by group and gender')
#     # plt.xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
#     # plt.yticks(np.arange(0, 81, 10))
#     # plt.legend((p1[0], p2[0]), ('Men', 'Women'))

#     plt.show()





if __name__ == "__main__":
    data_file = input("Please enter the file name you want to load: ")
    data = torch.load(data_file)
    data = np.array(data).tolist()
    action_group_length = input("Please enter the size you would like to group actions by: ")
    action_group_length = int(action_group_length)
    main(data,action_group_length)
