import visdom
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import Adam, SGD
import torch
from random import uniform, randint, sample, random, choices
from PIL import Image
import os

FloatTensor = torch.cuda.FloatTensor
LongTensor = torch.cuda.LongTensor
def plot(values, save_name, title = 'decomposition values'):
    plt.clf()
    x_pos = []
    x_name = []
    for i in range(len(values)):
        x_pos.append(i + 1)
        x_name.append("f_{}".format(i + 1))

    plt.bar(x_pos, values, align='center', alpha=0.5)
    
    plt.xticks(x_pos, x_name)
    plt.ylabel('Accumulated Future Value of Features')
    plt.title(title)

#     vis.matplot(plt)
    plt.savefig(save_name + ".jpg")
    
def plot_msx_plus(values, save_name, title = 'decomposition values'):
    plt.clf()
    x_pos = []
    x_name = []
    idx = np.argsort(np.array(values))[::-1]
    values = sorted(values, reverse=True)
    for i in range(len(values)):
        x_pos.append(i + 1)
        x_name.append("f_{}".format(idx[i] + 1))
    
    
    plt.bar(x_pos, values, align='center', alpha=0.5)
    plt.xticks(x_pos, x_name)
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
    plt.xticks([r + center_pos for r in range(length)], group, ha='center')

    # Create legend & Show graphic
    if len(elements) > 0:
        plt.legend()
#     vis.matplot(plt)
    plt.savefig(save_name + ".jpg")

def differenc_vector_state(model, target, baseline, env_name, txt_info, verbose = True, iteration = 100):
    t_feature, t_frame, t_value = target
    b_feature, b_frame, b_value = baseline
    vis = visdom.Visdom(env = env_name)
    if verbose:
        txt_info.append("target value: {}\n".format(t_value.item()))
        txt_info.append("baseline value: {}\n".format(b_value.item()))
        vis.images(t_frame, b_frame)
        
    (msx_idx, msx_value), _ = intergated_gradients(model, t_feature, txt_info, vis, 
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
                    txt_info = None, pick_actions = [1, 1, 1], decision_point = "undifined"):
    if txt_info is None:
        txt_info = []
    exp_path_dp = save_path + "/{}".format(decision_point)
#     if not os.path.isdir(exp_path_dp):
#         os.mkdir(exp_path_dp)
    os.makedirs(exp_path_dp, exist_ok=True)
    state_txt = pretty_print(state, text = "State:\n")
#     vis = visdom.Visdom(env = decision_point)
#     vis.clear_event_handlers(decision_point)
#     vis_txt(vis, state_txt)
#     vis.images(frame)

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
    plot(v_features[q_best_idx].tolist(), "{}/target_features".format(exp_path_dp), title = 'GVFs')
    txt_info.append("=====================================\n")
    
    
    if len(q_sort_idx) > (sum(pick_actions) + 1):
        random_idx = LongTensor(np.random.choice(q_sort_idx[pick_actions[0] + 1: -pick_actions[1]].tolist(), 
                                                    pick_actions[2], replace = False))
        q_sort_idx = torch.cat((q_sort_idx[1 : pick_actions[0] + 1], q_sort_idx[-pick_actions[1]:], q_sort_idx[random_idx]))
    
    show_image = True
    for i, sub_action in enumerate(q_sort_idx):
        save_name = "{}/subaction_#{}".format(exp_path_dp, i + 1)       
        txt_info.append("\nbaseline subaction_#{}: {}".format(i, pretty_print_action(actions[sub_action].tolist())))
        txt_info.append("baseline features: {}\n".format(v_features[sub_action].tolist()))
        txt_info.append("baseline value: {}\n".format(q_value[sub_action].item()))
        plot(v_features[q_best_idx].tolist(), save_name + "_features", title = 'GVFs')
        sub_action = sub_action.item()
        
        msx_idx, msx_value, intergated_grad = differenc_vector_action(fq_model.q_model, 
                              (v_features[q_best_idx], q_value[q_best_idx])
                              ,(v_features[sub_action], q_value[sub_action]), save_name, txt_info,
                            verbose = True, iteration = 30, show_image = show_image)
        
#         ig = np.array(intergated_grad[0].tolist())
#         IGs.append(ig)
#         baseline_values.append(q_best_values[i] - sum(ig))
#         orginal_pos_MSX = np.zeros(len(ig))
#         orginal_pos_MSX[msx_idx] = ig[msx_idx]
#         MSX_values.append(orginal_pos_MSX)
    if show_image:
        show_image = False
    txt_info.append("\n")
    file_name = "{}/info.txt".format(exp_path_dp)
    save_txt(file_name, txt_info)
def esf_state_pair():
    pass


# def vis_txt(vis, txt_info):
#     all_text = ""
#     for text in txt_info:
#         all_text += text
#     vis.text(all_text)
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
        state.append(str(s))
    all_text += text
    all_text += ("Wave:\t" + state[-1])
    all_text += ("Minerals:\t" + state[0])
    all_text += ("Building_Self\n")
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
    all_text += ("S_T:{:^5},S_B{:^5},E_T{:^5},E_B:{:^5}\n\n".format(
        state[63],state[64],state[65],state[66]))
    return all_text
                        
    
def pretty_print_action(action):
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