from tkinter import *
from tkinter.ttk import *
import tkinter as tk
from PIL import Image, ImageTk
import glob
import sys, os

COMBOBOX_WIDTH = 40
SUB_FIGURES_SIZE = (400, 300)
STATE_SIZE = (500, 300)
TARGET_FEATURES_SIZE = (400, 300)
STATE_W = 300
BL_W = 500
STATE_INFO_SIZE = (40, 50)
BL_INFO_SIZE = (100, 25)
BUTTON_SIZE = (5, 5)

def read_data(path, action):
    action_img_list = glob.glob("{}/{}*.jpg".format(path, action))
    
    for a_img in action_img_list:
        im = Image.open(a_img).resize(SUB_FIGURES_SIZE)
        render = ImageTk.PhotoImage(im)
        if "weights" in a_img:
            weights_img = render
        if "features" in a_img:
            features_img = render
        if "IG" in a_img:
            IG_img = render
        if "MSX" in a_img:
            MSX_img = render

    state_img_path = "{}/state.jpg".format(path)
    im = Image.open(state_img_path).resize(STATE_SIZE)
    state_img = ImageTk.PhotoImage(im)
    
    target_features_img_path = "{}/target_features.jpg".format(path)
    im = Image.open(target_features_img_path).resize(TARGET_FEATURES_SIZE)
    target_features_img = ImageTk.PhotoImage(im)
    
    lines = open("{}/info.txt".format(path)).readlines()
    
    state_info_txt, subaction_info_txt = "", ""
    state_flag = False
    subaction_flag = False
    for line in lines:
        if not state_flag:
            if "===========" not in line:
                state_info_txt += line
            else:
                state_flag = True
        elif action in line:
            subaction_flag = True
        if subaction_flag:
            subaction_info_txt += line
            if line[0] == "\n":
                break
        
    return state_info_txt, subaction_info_txt, state_img, target_features_img, weights_img, features_img, IG_img, MSX_img
    
def read_case():
    cases = []
    for d1 in os.scandir("."):
        if d1.is_dir():
            for d2 in os.scandir(d1.path):
                if ":" in d2.path:
                    cases.append("{}/{}".format(d2.path.split("/")[-2], d2.path.split("/")[-1]))
    return sorted(cases)

def select_game(event):
    games = []
#     path = event.widget.get()
#     if event is not None:
    path = combo_version.get()
    for d in os.scandir(path):
        if "game" in d.path:
            games.append(d.path.split("/")[-1])
#     print(games)
    games = sorted(games)
    combo_game['values']= (games)
    combo_game.current(0)
    select_dp(None)
    
def select_dp(event):
    dps = []
    
    path = "{}/{}".format(combo_version.get(), combo_game.get())
    for d in os.scandir(path):
        if "dp" in d.path:
            dps.append(d.path.split("/")[-1])
#     dps = sorted(dps)
#     print(dps)
    dps_sorted = []
    for i in range(len(dps)):
        dps_sorted.append("dp_{}".format(i + 1))
#     print(dps_sorted)
    combo_dp['values']= (dps_sorted)
    combo_dp.current(0)
    select_action(None)
    
def select_action(event):
    actions = set()
    path = "{}/{}/{}".format(combo_version.get(), combo_game.get(), combo_dp.get())
    for d in os.scandir(path):
        if "subaction_#" in d.path:
            actions.add(d.path.split("/")[-1][:12])
    actions = sorted(list(actions))
    combo_action['values']= (actions)
    combo_action.current(len(actions) - 1)
    show_exp(None)
    
def show_exp(event):
    path = "{}/{}/{}".format(combo_version.get(), combo_game.get(), combo_dp.get())
    action = combo_action.get()
    
    state_info_txt, subaction_info_txt, state_img, target_features_img, weights_img, features_img, IG_img, MSX_img = read_data(path, action)
    
#     print(state_info_txt)
#     s_txt.set(state_info_txt)
    state_info.config(state=NORMAL)
    state_info.delete(1.0, END)
    state_info.insert(END, state_info_txt)
    state_info.config(state=DISABLED)
    
    state.config(image = state_img)
    state.image = state_img
    
    state_feature.config(image = target_features_img)
    state_feature.image = target_features_img
    
#     bl_txt.set(subaction_info_txt)
    baseline_info.config(state=NORMAL)
    baseline_info.delete(1.0, END)
    baseline_info.insert(END, subaction_info_txt)
    baseline_info.config(state=DISABLED)
    
    baseline_feature.config(image = weights_img)
    baseline_feature.image = weights_img
    
    baseline_weight.config(image = features_img)
    baseline_weight.image = features_img
    
    baseline_ig.config(image = IG_img)
    baseline_ig.image = IG_img

    baseline_MSX.config(image = MSX_img)
    baseline_MSX.image = MSX_img

def key(event):
    if event.char == 'd':
        next_dp(None)
    if event.char == 'a':
        prev_dp(None)
        
    if event.char == 's':
        next_action(None)
    if event.char == 'w':
        prev_action(None)

def next_dp(event):
    curr_value = combo_dp.get()
#     print(curr_value)
    dp_num = int(curr_value[3]) * 10 + int(curr_value[4]) if len(curr_value) > 4 else int(curr_value[3])
    
    if dp_num == len(combo_dp['values']):
        return
    dp_num += 1
    combo_dp.current(dp_num - 1)
    select_action(None)

def prev_dp(event):
    curr_value = combo_dp.get()
    dp_num = int(curr_value[3]) * 10 + int(curr_value[4]) if len(curr_value) > 4 else int(curr_value[3])
    
    if dp_num == 1:
        return
    dp_num -= 1
    combo_dp.current(dp_num - 1)
    select_action(None)
    
def next_action(event):
    curr_value = combo_action.get()
    action_num = int(curr_value[-2]) * 10 + int(curr_value[-1]) if len(curr_value) > 12 else int(curr_value[-1])
    
    if action_num == len(combo_action['values']):
        return
    action_num += 1
    combo_action.current(action_num - 1)
    show_exp(None)

def prev_action(event):
    curr_value = combo_action.get()
    action_num = int(curr_value[-2]) * 10 + int(curr_value[-1]) if len(curr_value) > 12 else int(curr_value[-1])
    
    if action_num == 1:
        return
    action_num -= 1
    combo_action.current(action_num - 1)
    show_exp(None)

window = Tk()
window.title("Show gqf explanation")
window.geometry('1800x1100')

cases = read_case()
combo_version = Combobox(window, width = COMBOBOX_WIDTH, state="readonly")
combo_version['values']= (cases)
combo_version.current(0)
combo_version.grid(column=0, row=0)
combo_version.bind("<<ComboboxSelected>>", select_game)

combo_game = Combobox(window, width = COMBOBOX_WIDTH, state="readonly")
combo_game.grid(column=1, row=0)
combo_game.bind("<<ComboboxSelected>>", select_dp)

combo_dp = Combobox(window, width = COMBOBOX_WIDTH, state="readonly")
combo_dp.grid(column=2, row=0)
combo_dp.bind("<<ComboboxSelected>>", select_action)

combo_action = Combobox(window, width = COMBOBOX_WIDTH, state="readonly")
combo_action.grid(column=3, row=0)
combo_action.bind("<<ComboboxSelected>>", show_exp)

# s_txt = StringVar()
# s_txt.set("1111")
# s_txt = tk.Text(window, height=2, width=30)
# s_txt.insert(tk.END, "Just a text Widget\nin two lines\n")
# state_info = Label(window, wraplength = STATE_W, textvariable = s_txt)
# state_info.grid(column=0, row=2, rowspan=2)
# state_label_info.pack()

state_info = tk.Text(window, height=STATE_INFO_SIZE[1], width=STATE_INFO_SIZE[0])
scroll = tk.Scrollbar(window, command=state_info.yview)
state_info.configure(yscrollcommand=scroll.set)
state_info.insert(tk.END,'1111')
state_info.grid(column=0, row=2, rowspan=2)

state = Label(window, text = "2222")
state.grid(column=0, row=1, columnspan = 2)
# state_label_img.pack()

state_feature = Label(window, text = "3333")
state_feature.grid(column=1, row=2)

# bl_txt = StringVar()
# bl_txt.set("4444")
# baseline_info = Label(window, wraplength = BL_W, textvariable = bl_txt)
# baseline_info.grid(column=2, row=1)
baseline_info = tk.Text(window, height=BL_INFO_SIZE[1], width=BL_INFO_SIZE[0])
scroll = tk.Scrollbar(window, command=baseline_info.yview)
baseline_info.configure(yscrollcommand=scroll.set)
baseline_info.insert(tk.END,'4444')
baseline_info.grid(column=2, row=1, columnspan = 2)

baseline_feature = Label(window, text = "5555")
baseline_feature.grid(column=2, row=2)

baseline_weight = Label(window, text = "6666")
baseline_weight.grid(column=3, row=2)

baseline_ig = Label(window, text = "7777")
baseline_ig.grid(column=2, row=3)

baseline_MSX = Label(window, text = "8888")
baseline_MSX.grid(column=3, row=3)

b_next = Button(window, text = ">dp")
b_next.bind("<Button-1>", next_dp)
b_next.grid(column=4, row=1)

b_prev = Button(window, text = "<dp")
b_prev.bind("<Button-1>", prev_dp)
b_prev.grid(column=5, row=1)

a_next = Button(window, text = ">action")
a_next.bind("<Button-1>", next_action)
a_next.grid(column=4, row=2)

a_prev = Button(window, text = "<action")
a_prev.bind("<Button-1>", prev_action)
a_prev.grid(column=5, row=2)

window.bind("<Key>", key)
select_game(None)
window.mainloop()
