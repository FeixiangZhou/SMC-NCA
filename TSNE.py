import importlib
import os, sys
import warnings
warnings.filterwarnings('ignore')

import argparse
import random
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.manifold import TSNE
from collections import OrderedDict
import matplotlib.pyplot as plt

# base_dir_name = "/data/3dpoint/project/action/mstcn_data/50salads/features/"
# base_dir_name = "/data/3dpoint/project/action/mstcn_data/50salads/results/ICC_SS_TAS/unsupervised_C2FTCN_splitfull/features_dump_ori/"
# base_dir_name = "/data/3dpoint/project/action/mstcn_data/50salads/results/ICC_SS_TAS/unsupervised_C2FTCN_splitfull/my_triploss/run_summary_bs5_N7_3_20_multiclusterthree2/features_dump/"


# train_split_file = "/data/3dpoint/project/action/mstcn_data/50salads/splits/all_files.txt"
# ground_truth_files_dir =  "/data/3dpoint/project/action/mstcn_data/50salads/groundTruth/"
# label_id_csv = "/data/3dpoint/project/action/mstcn_data/50salads/mapping.csv"


#gtea

# base_dir_name = "/data/3dpoint/project/action/mstcn_data/gtea/features/"
# base_dir_name = "/data/3dpoint/project/action/mstcn_data/gtea/results/ICC_SS_TAS/unsupervised_C2FTCN_splitfull/features_dump_ori/"
# base_dir_name = "/data/3dpoint/project/action/mstcn_data/gtea/results/ICC_SS_TAS/unsupervised_C2FTCN_splitfull/my_triploss/run_summary_bs5_N7_3_2_multiclusterthree2/features_dump/"
#
#
# train_split_file = "/data/3dpoint/project/action/mstcn_data/gtea/splits/all_files.txt"
# ground_truth_files_dir =  "/data/3dpoint/project/action/mstcn_data/gtea/groundTruth/"
# label_id_csv = "/data/3dpoint/project/action/mstcn_data/gtea/mapping.csv"


# pdmb
# base_dir_name = "/data/3dpoint/project/action/mstcn_data/gtea/features/"
# base_dir_name = "/data/3dpoint/project/action/mstcn_data/gtea/results/ICC_SS_TAS/unsupervised_C2FTCN_splitfull/features_dump_ori/"
base_dir_name = "/data/3dpoint/project/action/mstcn_data/pdmb/results/ICC_SS_TAS/unsupervised_C2FTCN_splitfull/my_triploss/run_summary_bs5_N7_3_2_multiclusterthree_epoch_149_run2/features_dump_our_epoch430/"


train_split_file = "/data/3dpoint/project/action/mstcn_data/pdmb/splits/all_files.txt"
ground_truth_files_dir =  "/data/3dpoint/project/action/mstcn_data/pdmb/groundTruth/"
label_id_csv = "/data/3dpoint/project/action/mstcn_data/pdmb/mapping.csv"








all_train_data_files = open(train_split_file).read().split("\n")[0:-1]
print(all_train_data_files)

df = pd.read_csv(label_id_csv)

label_id_to_label_name = {}
label_name_to_label_id_dict = {}
for i, ele in df.iterrows():
    label_id_to_label_name[ele.label_id] = ele.label_name
    label_name_to_label_id_dict[ele.label_name] = ele.label_id

data_list = []
label_list = []

for i, video_id in enumerate(all_train_data_files):
    video_id = video_id.split(".txt")[0]
    filename = os.path.join(ground_truth_files_dir, video_id + ".txt")

    with open(filename, 'r') as f:
        recog_content = f.read().split('\n')[0:-1]  # framelevel recognition is in 6-th line of file
        f.close()

    recog_content = [label_name_to_label_id_dict[e] for e in recog_content]  # name to number [17,17----]
    # recog_content = np.array(recog_content)
    label_list.append(recog_content)




    total_frames = len(recog_content)  # 9694
    print(total_frames)


    image_path = os.path.join(base_dir_name, video_id + ".npy")
    elements = np.load(image_path) #[2048,9694]
    # elements = elements.T #--------------------------------------------------i3d
    print(elements.shape)
    data_list.append(elements)

label_all =  label_list[0]
for i in label_list[1:]:
    label_all = label_all + i

label_all = label_all[::20]  #  gtea-4 50salads-200
label_all = np.array(label_all)
print("all_label--", len(label_all)) #577595-50salads 31225-gtea
index  = np.where(label_all==8)[0]
label_all = np.delete(label_all, index)
print("after delete--", len(label_all)) #577595-50salads 31225-gtea

elements = np.vstack(data_list)
elements = elements[::20,:]
elements = np.delete(elements, index,0)

print("all_data--", elements.shape)  #(577595, 2048)


def plot_embedding(data, label, title=None):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    # plt.figure(figsize=(10, 10))
    for i in range(data.shape[0]):
        print(i)
        # label[i] == 0
        log = label_id_to_label_name[ label[i]]

        if label[i]==0:
            c = 'grey'
        if label[i]==1:
            c = 'gold'
        if label[i]==2:
            c = 'darkviolet'
        if label[i]==3:
            c = 'turquoise'
        if label[i]==4:
            c = 'r'
        if label[i] == 5:
            c = 'g'
        if label[i] == 6:
            c = 'b'
        if label[i] == 7:
            c = 'c'
        if label[i] == 8:
            c = 'm'
        if label[i] == 9:
            c = 'y'
        if label[i] == 10:
            c = 'salmon'
        if label[i] == 11:
            c = 'darkorange'
        if label[i] == 12:
            c = 'lightgreen'
        if label[i] == 13:
            c = 'plum'
        if label[i] == 14:
            c = 'tan'
        if label[i] == 15:
            c = 'khaki'
        if label[i] == 16:
            c = 'gold'
        if label[i] == 17:
            c = 'pink'
        if label[i] == 18:
            c = 'skyblue'


        # print(i)
        # plt.text(data[i, 0], data[i, 1], str(label[i]),color=plt.cm.Set1(label[i] / 10.), fontdict={'weight': 'bold', 'size': 9})
        plt.scatter(data[i, 0],data[i, 1], s=30, marker='.', color=c, label=log)
        # plt.scatter3d(data[i, 0],data[i, 1], s=30, marker='.', color=plt.cm.Set3(label[i]), label=log)

    # plt.xticks([])
    # plt.yticks([])
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    # plt.legend(by_label.values(), by_label.keys(), loc=4)
    # lg = plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1.0), loc='upper left', ncol=5)
    plt.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(0.5, -0.5),loc=8, ncol=5,  borderaxespad = 0.)
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)

    # plt.title(title)
    # if label[0] ==1:
    # plt.savefig('tsne_our.png')
    # plt.savefig('tsne_50salads_i3d_inter200.png', bbox_inches='tight', pad_inches=1)
    # plt.savefig('tsne_50salads_ori_inter200.png', bbox_inches='tight', pad_inches=1)
    # plt.savefig('tsne_50salads_our.png', bbox_inches='tight', pad_inches=1)



    # plt.savefig('tsne_gtea_i3d2.png', bbox_inches='tight', pad_inches=1)
    # plt.savefig('tsne_gtea_ori2.png', bbox_inches='tight', pad_inches=1)
    # plt.savefig('tsne_gtea_our2.png', bbox_inches='tight', pad_inches=1)


    plt.savefig('tsne_pdmb_our_inter200.png', bbox_inches='tight', pad_inches=1)
    # plt.savefig('tsne_50salads_ori_inter200.png', bbox_inches='tight', pad_inches=1)
    # plt.savefig('tsne_50salads_our.png', bbox_inches='tight', pad_inches=1)

# Random state.
RS = 20150101
X_embedded = TSNE(n_components=2, perplexity=150, learning_rate=200.0, n_iter=1000, random_state=RS).fit_transform(elements)  #gtea-per=150   #50salads-per=100
## X_embedded = TSNE(n_components=2, init='pca', random_state=0).fit_transform(X)

plot_embedding(X_embedded, label_all)


