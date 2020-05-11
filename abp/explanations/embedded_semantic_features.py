import visdom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torch.optim import Adam, SGD
import torch
from random import uniform, randint, sample, random, choices
from PIL import Image
import os
import math
from collections import OrderedDict 
from datetime import datetime

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
SUB_FIGURE_SIZE = (25, 15)
SUB_PLOT_NUN = 4

VERSION = ""
TITLES = []
XLABELS = []
norm_vector_GVFs_all_v1 = np.array([1500, 100, 30, 200, 5000])

def normalization(values):
    norm_values = np.array(values).copy()
    if VERSION in ["GFVs_all_1"]:
        norm_values[8 : 10] *= norm_vector_GVFs_all_v1[0]
        norm_values[10 : 22] *= norm_vector_GVFs_all_v1[1]
        norm_values[22 : 82] *= norm_vector_GVFs_all_v1[2]
        norm_values[82 : 94] *= norm_vector_GVFs_all_v1[3]
        norm_values[94 : 130] *= norm_vector_GVFs_all_v1[4]
    return norm_values.tolist()

def denormalization(values):
    denorm_values = np.array(values).copy()
    if VERSION in ["GFVs_all_1"]:
        denorm_values[8 : 10] /= norm_vector_GVFs_all_v1[0]
        denorm_values[10 : 22] /= norm_vector_GVFs_all_v1[1]
        denorm_values[22 : 82] /= norm_vector_GVFs_all_v1[2]
        denorm_values[82 : 94] /= norm_vector_GVFs_all_v1[3]
        denorm_values[94 : 130] /= norm_vector_GVFs_all_v1[4]
    return denorm_values.tolist()

def plot(values, save_name, title = 'decomposition values'):
    
    global VERSION
    global XLABELS
    if VERSION in ["GFVs_all_1"]:
        if "features" in save_name:
            values = normalization(values)
        if "weights" in save_name:
            values = denormalization(values)
        detail_plot(values, save_name)
    plt.clf()
    x_pos = []
    x_name = []
    for i in range(len(values)):
        x_pos.append(i + 1)
        if len(values) < 20:
            x_name.append("f_{}".format(i + 1))
        elif values[i] == max(values):
            x_name.append("f_{}".format(i + 1))
        else:
            x_name.append("")
        
    if VERSION == "v10":
        x_name = XLABELS
    plt.bar(x_pos, values, align='center', alpha=0.5)
    
    plt.xticks(x_pos, x_name, rotation=40)
    plt.ylabel('Accumulated Future Value of Features')
    plt.title(title)

#     vis.matplot(plt)
    plt.savefig(save_name + ".jpg")
#     print(VERSION)

    
def detail_plot(values, save_name):
    plt.clf()
    global TITLES
    global XLABELS
#     print(TITLES)
    fig, axs = plt.subplots(math.ceil(len(TITLES) / SUB_PLOT_NUN), SUB_PLOT_NUN, figsize = SUB_FIGURE_SIZE)
    if "MSX" in save_name or "IG" in save_name:
        ylim = (min(values) - 1/10 * min(values), max(values) * 1.1)
        plt.setp(axs, ylim = ylim)
#     print(SUB_PLOT_NUN, math.ceil(len(TITLES) / SUB_PLOT_NUN))
    for t_idx, (title, (idx, length)) in enumerate(TITLES.items()):
        x_pos = []
        x_name = []
        v = values[idx : idx + length]
        for i in range(len(v)):
            x_pos.append(i + 1)
            x_name.append("f_{}".format(i + 1))
        
#         print(title)
#         print(t_idx)
#         print(XLABELS)
        c, r = int(t_idx % SUB_PLOT_NUN), int(t_idx / SUB_PLOT_NUN)
        axs[r][c].bar(x_pos, v, align='center', alpha=0.5)
        axs[r][c].set_xticks(np.arange(len(x_name)) + 1)
        axs[r][c].set_xticklabels((XLABELS[t_idx]), rotation = 40, fontsize=8)
        axs[r][c].set_ylabel('Accumulated Future Value')
        axs[r][c].set_title(title)
        axs[r][c].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
#         plt.bar(x_pos, values, align='center', alpha=0.5)

#         plt.xticks(x_pos, x_name)
#         plt.ylabel('Accumulated Future Value of Features')
#         plt.title(title)

#     #     vis.matplot(plt)
#         plt.savefig(save_name + ".jpg")
#     print(save_name)
    fig.savefig("{}_detail.jpg".format(save_name))
    fig.clear()
    matplotlib.use('Agg')
    plt.close(fig) 
    
def plot_msx_plus(values, save_name, title = 'decomposition values'):
    if VERSION in ["GFVs_all_1"]:
        detail_plot(values, save_name)
    global XLABELS
    plt.clf()
    x_pos = []
    x_name = []
    idx = np.argsort(np.array(values))[::-1]
    values = sorted(values, reverse=True)
#     print(XLABELS)
#     print(idx)
    for i in range(len(values)):
        x_pos.append(i + 1)
        if VERSION == "v10":
            x_name.append(XLABELS[idx[i]])
        elif len(values) < 20:
            x_name.append("f_{}".format(idx[i] + 1))
        elif values[i] == max(values):
            x_name.append("f_{}".format(i + 1))
        else:
            x_name.append("")
    plt.bar(x_pos, values, align='center', alpha=0.5)
    plt.xticks(x_pos, x_name, rotation=40)
    plt.ylabel('Future value of feature')
    plt.title(title)
    plt.savefig(save_name + ".jpg")
#     vis.matplot(plt)
    
def MSX(vector):
    vector = np.array(vector)
    indeces = np.argsort(vector)[::-1]
    negative_sum = sum(vector[vector < 0])
    pos_sum = 0
    MSX_idx = []
    for idx in indeces:
        pos_sum += vector[idx]
        MSX_idx.append(idx)
        if pos_sum > abs(negative_sum):
            break
    return MSX_idx, vector[MSX_idx]

def plot_action_group(values, group, save_name, elements = [], title = 'decomposition values'):
    plt.clf()
    x = np.arange(len(group))  # the label locations

    fig, ax = plt.subplots()

    # Add some text for labels, title and custom x-axis tick labels, etc.
    plt.ylabel('Accumulated Future Value of Features')
    plt.title(title)

    # set width of bar
    length = len(values[0])
    barWidth = 1 / (len(values) + 1)
    
    for i in range(len(values)):
        r = [j + barWidth * i for j in range(length)]
        if len(elements) > 0:
            plt.bar(r, values[i], width=barWidth, edgecolor='white', label=elements[i])
        else:
            plt.bar(r, values[i], width=barWidth, edgecolor='white')
    
    r = [j + barWidth * (i + 1) for j in range(length)]
    for rr in r[:-1]:
        plt.axvline(x = rr, alpha = 0.5, linestyle='--')

    # Add xticks on the middle of the group bars
    plt.xlabel('action', fontweight='bold')
    center_pos = (1 - barWidth * 2) / 2
    plt.xticks([r + center_pos for r in range(length)], group, ha='center', rotation=40)

    # Create legend & Show graphic
    if len(elements) > 0:
        plt.legend()
#     vis.matplot(plt)
    plt.savefig(save_name + ".jpg")

def differenc_vector_state(model, target, baseline, save_name, txt_info = [], verbose = True, iteration = 100, show_image = True):
    t_feature, t_value = target
    b_feature, b_value = baseline
        
    (msx_idx, msx_value), _ = intergated_gradients(model, t_feature, txt_info, save_name, 
                                                   baseline = b_feature, verbose = verbose, iteration = iteration)
    return msx_idx, msx_value

def differenc_vector_action(model, target, baseline, save_name, txt_info = [], verbose = True, iteration = 100, show_image = True):
    t_feature, t_value = target
    b_feature, b_value = baseline

    (msx_idx, msx_value), intergated_grad = intergated_gradients(model, t_feature, txt_info, save_name,
                                                                 baseline = b_feature, verbose = verbose, iteration = iteration)

    return msx_idx, msx_value, intergated_grad

def intergated_gradients(model, x, txt_info, save_name, iteration = 100, baseline = None, verbose = True):
    y_baseline = model(baseline).item()
    x = x.view(1, -1)
    x.size()[1]
    optimizer = Adam(model.parameters(), lr = 0.001)
    if baseline is None:
        baseline = torch.zeros_like(x)
#     elif verbose:
#         txt_info.append("baseline: {}\n".format(baseline))

    intergated_grad = torch.zeros_like(x)

    for i in range(iteration):
        new_input = baseline + ((i + 1) / iteration * (x - baseline))
        new_input = new_input.clone().detach().requires_grad_(True)

        y = model(new_input)
        loss = abs(y_baseline - y)
        
        optimizer.zero_grad()
        loss.backward()
        intergated_grad += (new_input.grad) / iteration
    if verbose:
#         txt_info.append("input:{}\n".format(x))
        txt_info.append("weights:{}\n".format(intergated_grad.tolist()[0]))
        plot(intergated_grad.tolist()[0], save_name + "_weights", title = 'Weights')

    intergated_grad *= x - baseline

    MSX_idx, MSX_values = MSX(intergated_grad.tolist()[0])
    if verbose:
        txt_info.append("Intergrated Gradient:{}\n".format(intergated_grad.tolist()[0]))
        plot(intergated_grad.tolist()[0], save_name + "_IG", title = 'Intergrated Gradient')

        msx_vector = np.zeros(len(x[0]))
        msx_vector[MSX_idx] = MSX_values
        plot_msx_plus(msx_vector, save_name + "_MSX+", title = 'MSX+')

    return (MSX_idx, MSX_values), intergated_grad

def esf_action_pair(fq_model, state, frame, state_actions, actions, save_path,
                    txt_info = None, pick_actions = [1, 1, 1], decision_point = "undifined", version = ""):
    set_VERSION_and_LABELS(version)
    if txt_info is None:
        txt_info = []
    exp_path_dp = save_path + "/{}".format(decision_point)
#     if not os.path.isdir(exp_path_dp):
#         os.mkdir(exp_path_dp)
    os.makedirs(exp_path_dp, exist_ok=True)
    state_txt = pretty_print(state, text = "State:\n")
    txt_info.append(state_txt)
    im = Image.fromarray(frame)
    im.save("{}/state.jpg".format(exp_path_dp))
    
    with torch.no_grad():
        v_features, q_value = fq_model.predict_batch(state_actions)
    q_value = q_value.view(-1)
    q_sort_idx = q_value.argsort(descending = True).view(-1)
    q_best_idx = q_sort_idx[0]
    q_best_value = q_value[q_sort_idx[0]]
    
    txt_info.append("target action: {}\n".format(pretty_print_action(actions[q_best_idx].tolist())))
    txt_info.append("target features: {}\n".format(v_features[q_best_idx].tolist()))
    txt_info.append("target value: {}\n".format(q_best_value.item()))
    plot(v_features[q_best_idx].tolist(), "{}/target_features".format(exp_path_dp), title = 'Target GVFs')
    txt_info.append("=====================================\n")
    
    
    if len(q_sort_idx) > (sum(pick_actions) + 1):
        random_idx = LongTensor(np.random.choice(q_sort_idx[pick_actions[0] + 1: -pick_actions[1]].tolist(), 
                                                    pick_actions[2], replace = False))
        q_sort_idx = torch.cat((q_sort_idx[1 : pick_actions[0] + 1], q_sort_idx[random_idx], q_sort_idx[-pick_actions[1]:]))
    
    show_image = True
    for i, sub_action in enumerate(q_sort_idx):
        if i < pick_actions[0]:
            entail = "(best)"
        elif i < (pick_actions[0] + pick_actions[1]):
            entail = "(rand)"
        else:
            entail = "(worst)"
        save_name = "{}/subaction_#{}{}".format(exp_path_dp, i + 1, entail)       
        txt_info.append("\nbaseline subaction_#{}{}: {}".format(i + 1, entail, pretty_print_action(actions[sub_action].tolist())))
        txt_info.append("baseline features: {}\n".format(v_features[sub_action].tolist()))
        txt_info.append("baseline value: {}\n".format(q_value[sub_action].item()))
        plot(v_features[sub_action].tolist(), save_name + "_features", title = 'Baseline GVFs')
        sub_action = sub_action.item()
        
        msx_idx, msx_value, intergated_grad = differenc_vector_action(fq_model.q_model, 
                              (v_features[q_best_idx], q_value[q_best_idx])
                              ,(v_features[sub_action], q_value[sub_action]), save_name, txt_info,
                            verbose = True, iteration = 30, show_image = show_image)
        
    if show_image:
        show_image = False
    txt_info.append("\n")
    file_name = "{}/info.txt".format(exp_path_dp)
    save_txt(file_name, txt_info)

def esf_state_pair(fq_model, state_1, frame_1, state_2, frame_2, env, save_path,
                   entail = "",txt_info = None, version = ""):
    set_VERSION_and_LABELS(version)
    if txt_info is None:
        txt_info = []
        
    action_1 = env.get_big_A(state_1[env.miner_index], state_1[env.pylon_index])
    action_2 = env.get_big_A(state_2[env.miner_index], state_2[env.pylon_index])
    state_actions_1 = env.combine_sa(state_1, action_1)
    state_actions_2 = env.combine_sa(state_2, action_2)
    
    exp_path_dp = save_path + "/dp_{}_vs_dp_{}_{}".format(int(state_1[-1]), int(state_2[-1]), entail)
    if not os.path.isfile(exp_path_dp):
        os.makedirs(exp_path_dp, exist_ok=True)
    else:
        exp_path_dp += datetime.now()
    print(exp_path_dp)
    state_txt_1 = pretty_print(state_1, text = "State 1:\n")
    txt_info.append(state_txt_1)
    
    if frame_1 is not None:
        im = Image.fromarray(frame_1)
        im.save("{}/state_1.jpg".format(exp_path_dp))
    
    if frame_2 is not None:
        im = Image.fromarray(frame_2)
        im.save("{}/state_2.jpg".format(exp_path_dp))
    
    with torch.no_grad():
        v_features_1, q_value_1 = fq_model.predict_batch(env.normalization(state_actions_1))
        v_features_2, q_value_2 = fq_model.predict_batch(env.normalization(state_actions_2))
    q_value_1 = q_value_1.view(-1)
    q_value_2 = q_value_2.view(-1)
    
    best_q_value_1, best_idx_1 = q_value_1.max(0)
    best_q_value_2, best_idx_2 = q_value_2.max(0)
    
    best_features_1 = v_features_1[best_idx_1]
    best_features_2 = v_features_2[best_idx_2]
    
    if best_q_value_1 > best_q_value_2:
        target_q, target_best_idx, target_features, target_actions = best_q_value_1, best_idx_1, best_features_1, action_1
        baseline_q, baseline_best_idx, baseline_features, baseline_actions = best_q_value_2, best_idx_2, best_features_2, action_2
    else:
        target_q, target_best_idx, target_features, target_actions = best_q_value_2, best_idx_2, best_features_2, action_2
        baseline_q, baseline_best_idx, baseline_features, baseline_actions = best_q_value_1, best_idx_1, best_features_1, action_1
    
    txt_info.append("target state best action: {}\n".format(pretty_print_action(target_actions[target_best_idx].tolist())))
    txt_info.append("target features: {}\n".format(target_features.tolist()))
    txt_info.append("target value: {}\n".format(target_q.item()))
    plot(target_features.tolist(), "{}/target_features".format(exp_path_dp), title = 'Target GVFs')
    txt_info.append("=====================================\n")
    
    save_name = "{}/subaction_#{}".format(exp_path_dp, "N")
    txt_info.append("\nbaseline subaction_#{}".format("N"))
    state_txt_2 = pretty_print(state_2, text = "State 2:\n")
    txt_info.append(state_txt_2)
    txt_info.append("baseline state best action: {}\n".format(pretty_print_action(baseline_actions[baseline_best_idx].tolist())))
    txt_info.append("baseline features: {}\n".format(baseline_features.tolist()))
    txt_info.append("baseline value: {}\n".format(baseline_q.item()))
    plot(baseline_features.tolist(), save_name + "_features", title = 'Baseline GVFs')

    msx_idx, msx_value, intergated_grad = differenc_vector_action(fq_model.q_model, 
                          (target_features, target_q), (baseline_features, baseline_q), save_name, txt_info,
                        verbose = True, iteration = 30)
    txt_info.append("\n")
    file_name = "{}/info.txt".format(exp_path_dp)
    save_txt(file_name, txt_info)
     
def save_txt(fn, txt_info):
    f = open(fn, "w")
    for text in txt_info:
#         print(text)
        f.write(text)
    

def pretty_print(state, text = ""):
    state_list = state.copy().tolist()
    state = []
    all_text = ""
    for s in state_list:
        state.append(str(int(s)))
    all_text += text
    all_text += ("Wave:\t" + state[-1])
    all_text += (" Minerals:\t" + state[0])
    all_text += (" Building_Self\n")
    all_text += ("T:{:^5},{:^5},{:^5},B:{:^5},{:^5},{:^5},P:{:^5}\n".format(
        state[1],state[2],state[3],state[4],state[5],state[6],state[7]))
    all_text += ("Building_Enemy\n")
    all_text += ("T:{:^5},{:^5},{:^5},B:{:^5},{:^5},{:^5},P:{:^5}\n".format(
        state[8],state[9],state[10],state[11],state[12],state[13],state[14]))
    
    all_text += ("Unit_Self\n")
    all_text += ("     M  ,  B  ,  I \n")
    all_text += ("T1:{:^5},{:^5},{:^5}\n".format(
        state[15],state[16],state[17]))

    all_text += ("T2:{:^5},{:^5},{:^5}\n".format(
        state[18],state[19],state[20]))

    all_text += ("T3:{:^5},{:^5},{:^5}\n".format(
        state[21],state[22],state[23]))

    all_text += ("T4:{:^5},{:^5},{:^5}\n".format(
        state[24],state[25],state[26]))

    all_text += ("B1:{:^5},{:^5},{:^5}\n".format(
        state[27],state[28],state[29]))

    all_text += ("B2:{:^5},{:^5},{:^5}\n".format(
        state[30],state[31],state[32]))

    all_text += ("B3:{:^5},{:^5},{:^5}\n".format(
        state[33],state[34],state[35]))

    all_text += ("B4:{:^5},{:^5},{:^5}\n".format(
        state[36],state[37],state[38]))

    all_text += ("Unit_Enemy\n")
    all_text += ("     M  ,  B  ,  I \n")
    all_text += ("T1:{:^5},{:^5},{:^5}\n".format(
        state[39],state[40],state[41]))

    all_text += ("T2:{:^5},{:^5},{:^5}\n".format(
        state[42],state[43],state[44]))

    all_text += ("T3:{:^5},{:^5},{:^5}\n".format(
        state[45],state[46],state[47]))

    all_text += ("T4:{:^5},{:^5},{:^5}\n".format(
        state[48],state[49],state[50]))

    all_text += ("B1:{:^5},{:^5},{:^5}\n".format(
        state[51],state[52],state[53]))

    all_text += ("B2:{:^5},{:^5},{:^5}\n".format(
        state[54],state[55],state[56]))

    all_text += ("B3:{:^5},{:^5},{:^5}\n".format(
        state[57],state[58],state[59]))

    all_text += ("B4:{:^5},{:^5},{:^5}\n".format(
        state[60],state[61],state[62]))

    all_text += ("Hit_Point\n")
    all_text += ("S_T: {:^5},S_B: {:^5},E_T: {:^5},E_B: {:^5}\n\n".format(
        state[63],state[64],state[65],state[66]))
    return all_text
                        
    
def pretty_print_action(action):
    for i in range(len(action)):
        action[i] = int(action[i])
    txt = ""
    txt += ("Top_M: {}, Top_B: {}, Top_I: {}\n".format(action[0], action[1], action[2]))
    txt += ("Bottom_M: {}, Bottom_B: {}, Bottom_I: {}\n".format(action[3], action[4], action[5]))
    txt += ("Pylon: {}\n".format(action[6]))
    return txt
def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = numpy.fromstring ( fig.canvas.tostring_argb(), dtype=numpy.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = numpy.roll ( buf, 3, axis = 2 )
    return buf

def set_VERSION_and_LABELS(version):
    global VERSION
    global TITLES
    global XLABELS
    VERSION = version
    
    if VERSION == "GFVs_all_1":
        TITLES = OrderedDict({"Destory Prob and Lowest HP Prob": (0, 8),
                  "Minerals to be Earned": (8, 2),
                  "Expected Units to Spawn": (10, 12),
                  "Future Surviving Friendly (Top)": (22, 15),
                  "Future Surviving Friendly (Bottom)": (37, 15),
                  "Future Surviving Enemy (Top)": (52, 15),
                  "Future Surviving Enemy (Bottom)": (67, 15),
                  "Damage Inflicted on Nexus By Units": (82, 12),
                  "Friendly Damage Inflicted on Enemy": (94, 18),
                  "Enemy Damage Inflicted on Friendly": (112, 18),
                  "Probability to End by Tie-breaker": (130, 1)
                 })
        XLABELS = [
            ["FTD", "FBD", "ETD", "EBD", "FTL", "FBL", "ETL", "EBL"],
            ["F_M", "E_M"],
            ["FT_M", "FT_B", "FT_I", "FB_M", "FB_B", "FB_I", "ET_M", "ET_B", "ET_I", "EB_M", "EB_B", "EB_I"],
            ["FTR1_M", "FTR1_B", "FTR1_I","FTR2_M", "FTR2_B", "FTR2_I", "FTR3_M", "FTR3_B", "FTR3_I","FTR4_M", "FTR4_B", "FTR4_I","FTR5_M", "FTR5_B", "FTR5_I"],
            ["FBR1_M", "FBR1_B", "FBR1_I","FBR2_M", "FBR2_B", "FBR2_I", "FBR3_M", "FBR3_B", "FBR3_I","FBR4_M", "FBR4_B", "FBR4_I","FBR5_M", "FBR5_B", "FBR5_I"],
            ["ETR1_M", "ETR1_B", "ETR1_I","ETR2_M", "ETR2_B", "ETR2_I", "ETR3_M", "ETR3_B", "ETR3_I","ETR4_M", "ETR4_B", "ETR4_I","ETR5_M", "ETR5_B", "ETR5_I"],
            ["EBR1_M", "EBR1_B", "EBR1_I","EBR2_M", "EBR2_B", "EBR2_I", "EBR3_M", "EBR3_B", "EBR3_I","EBR4_M", "EBR4_B", "EBR4_I","EBR5_M", "EBR5_B", "EBR5_I"],
            ["FT_M", "FT_B", "FT_I", "FB_M", "FB_B", "FB_I", "ET_M", "ET_B", "ET_I", "EB_M", "EB_B", "EB_I"],
            ["TM_To_M", "TM_To_B", "TM_To_I", "TB_To_M", "TB_To_B", "TB_To_I", "TI_To_M", "TI_To_B", "TI_To_I", 
             "BM_To_M", "BM_To_B", "BM_To_I", "BB_To_M", "BB_To_B", "BB_To_I", "BI_To_M", "BI_To_B", "BI_To_I"],
            ["TM_To_M", "TM_To_B", "TM_To_I", "TB_To_M", "TB_To_B", "TB_To_I", "TI_To_M", "TI_To_B", "TI_To_I", 
             "BM_To_M", "BM_To_B", "BM_To_I", "BB_To_M", "BB_To_B", "BB_To_I", "BI_To_M", "BI_To_B", "BI_To_I"],
            ["ETB"]
        ]
    if VERSION == "v10":
        XLABELS = ["FT_M", "FT_B", "FT_I", "FB_M", "FB_B", "FB_I", "ET_M", "ET_B", "ET_I", "EB_M", "EB_B", "EB_I", "FT_N", "FB_N", "ET_N", "EB_N", "ETB"]


def pretty_print_to_state(pretty_state):
    state = np.zeros(68)
    count = 0
    self_units = []
    enemy_units = []
    for i, ps in enumerate(pretty_state):
        numbers = [float(word) for word in ps.split() if word.isdigit()]
        if len(numbers) == 0:
            continue
        numbers_np = np.array(numbers)
        if count == 0:
            state[-1], state[0] = numbers_np[0], numbers_np[1]
        if count == 1:
            state[1:8] = numbers_np
        if count == 2:
            state[8:15] = numbers_np
        if count in list(range(3, 11)):
            self_units.extend(numbers)
        if count == 10:
            state[15:39] = self_units
        if count in list(range(11, 19)):
            enemy_units.extend(numbers)
        if count == 18:
            state[39:63] = enemy_units
        if count == 19:
            state[63: 67] = numbers_np
        count += 1