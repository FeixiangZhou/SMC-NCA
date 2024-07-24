# import importlib
import os, sys
from sklearn.cluster import KMeans
from utility.perform_linear import get_linear_acc
from utility.dump_features import DumpFeatures
import warnings
import copy
import glob
import pandas as pd
warnings.filterwarnings('ignore')
import time

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model_mlp import C2F_TCN, NCA_discriminator

from utils import calculate_mof
from postprocess import PostProcess
from dataset import Breakfast, collate_fn_override, collate_fn_override_withselect
from utility.savemodel import saveepochcheckpont,load_best_model, saveepochcheckpont_semi_best, resume_checkpoint, saveepochcheckpont_semi
from utility.savemodel import saveepochcheckpont_semi_best_model_d
from utility.savemodel import load_best_model_semi, load_best_model_semi_after_niter, load_unsupervised_model, load_best_unsupervised_model, load_best_model_semi_model_d
import torch.nn.functional as F
from lossfunction_multibatch_semi_nca import CriterionClass

from utility.misc import AverageMeter, accuracy
from utility.utils import enable_dropout


seed = 42  
model_path = "/data/3dpoint/project/action/mstcn_data/50salads/results/SMC/unsupervised_C2FTCN_splitfull/my_triploss/run_summary_bs5_test/best_50salads_c2f_tcn.wt"
# model_path = "/data/3dpoint/project/action/mstcn_data/gtea/results/SMC/unsupervised_C2FTCN_splitfull/my_triploss/run_summary_bs4_test/best_gtea_c2f_tcn.wt"
# model_path = "/data/3dpoint/project/action/mstcn_data/breakfast/results/SMC/unsupervised_C2FTCN_splitfull/my_triploss/run_summary_bs50_test/best_breakfast_c2f_tcn.wt"


path = "/semi_per_0.05/run1"


my_parser = argparse.ArgumentParser()
my_parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
my_parser.add_argument('--split_number', default=1, type=int, required=False, help="Split for which results needs to be updated")
my_parser.add_argument('--semi_per', default=0.05, type=float, required=False, help="Percentage of semi-supervised data to be used as trainingdata")

my_parser.add_argument('--output_dir', default="/data/3dpoint/project/action/mstcn_data/50salads/results/SMC", type=str, required=False, help="Output directory where the programs outputs (checkpoints, logs etc.) must be stored")
my_parser.add_argument('--base_dir', default="/data/3dpoint/project/action/mstcn_data/50salads/", type=str, required=False)

# my_parser.add_argument('--output_dir', default="/data/3dpoint/project/action/mstcn_data/gtea/results/SMC", type=str, required=False, help="Output directory where the programs outputs (checkpoints, logs etc.) must be stored")
# my_parser.add_argument('--base_dir', default="/data/3dpoint/project/action/mstcn_data/gtea/", type=str, required=False)

# my_parser.add_argument('--output_dir', default="/data/3dpoint/project/action/mstcn_data/breakfast/results/SMC", type=str, required=False, help="Output directory where the programs outputs (checkpoints, logs etc.) must be stored")
# my_parser.add_argument('--base_dir', default="/data/3dpoint/project/action/mstcn_data/breakfast/", type=str, required=False)


my_parser.add_argument('--model_wt',default=model_path,  type=str, required=False)
my_parser.add_argument('--dataset_name', type=str, required=False)
my_parser.add_argument('--wd', type=float, required=False)
my_parser.add_argument('--lr_unsuper', type=float, required=False)
my_parser.add_argument('--lr_proj', type=float, required=False)
my_parser.add_argument('--lr_main', type=float, required=False)
my_parser.add_argument('--gamma_proj', type=float, required=False)
my_parser.add_argument('--gamma_main', type=float, required=False)
my_parser.add_argument('--epochs_unsuper', type=int, required=False)
my_parser.add_argument('--mse', action='store_true')
my_parser.add_argument(
  "--steps_proj",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=int,
  default=[600],  # default if nothing is provided
)
my_parser.add_argument(
  "--steps_main",  # name on the CLI - drop the `--` for positional/required parameters
  nargs="*",  # 0 or more values expected => creates a list
  type=int,
  default=[600, 1200],  # default if nothing is provided
)
my_parser.add_argument('--chunk_size', type=int, required=False)
my_parser.add_argument('--max_frames_per_video', type=int, required=False)
my_parser.add_argument('--weights', type=str, required=False)
my_parser.add_argument('--features_file_name', type=str, required=False)
my_parser.add_argument('--feature_size', type=int, default=2048, required=False)
my_parser.add_argument('--epochs', type=int, required=False)
my_parser.add_argument('--num_samples_frames', type=int, required=False)
my_parser.add_argument('--epsilon', type=float, required=False)
my_parser.add_argument('--delta', type=float, required=False)
my_parser.add_argument('--ftype', default='i3d', type=str)
my_parser.add_argument('--outft', type=int, default=256)
my_parser.add_argument('--no_unsuper', action='store_true')
my_parser.add_argument('--perdata', type=int, default=100)
my_parser.add_argument('--iter_num', type=int, nargs="*")
my_parser.add_argument('--getOutDir', action='store_true')
my_parser.add_argument('--train_split', type=str, required=False, help="File to be used for unsupervised feature learning")
my_parser.add_argument('--cudad', type=str, required=False)

#for debug
my_parser.add_argument('--tau-p', default=0, type=float,
                    help='confidece threshold for positive pseudo-labels, default 0.70')
my_parser.add_argument('--tau-n', default=0.05, type=float,
                    help='confidece threshold for negative pseudo-labels, default 0.05')
my_parser.add_argument('--kappa-p', default=0.05, type=float,
                    help='uncertainty threshold for positive pseudo-labels, default 0.05')
my_parser.add_argument('--kappa-n', default=0.005, type=float,
                    help='uncertainty threshold for negative pseudo-labels, default 0.005')
my_parser.add_argument('--no-uncertainty', default=True, action='store_true',
                    help='use uncertainty in the pesudo-label selection, default true')
args = my_parser.parse_args()

seed = 42
print(args.split_number)



# Ensure deterministic behavior
def set_seed():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


def get_out_dir(args, iter_num):

    args.output_dir = args.output_dir + path + "/icc{}_semi{}_split{}".format(iter_num, args.semi_per, args.split_number)

    if args.wd is not None:
        args.output_dir = args.output_dir + "_wd{:.5f}".format(args.wd)

    if args.chunk_size is not None:
        args.output_dir = args.output_dir + "_chunk{}".format(args.chunk_size)

    if args.max_frames_per_video is not None:
        args.output_dir = args.output_dir + "_maxf{}".format(args.max_frames_per_video)


    if args.delta is not None:
        args.output_dir = args.output_dir + "_dh{:.4f}".format(args.delta)

    if args.weights is not None:
        args.output_dir = args.output_dir + "_wts{}".format(args.weights.replace(',','-'))
        args.weights = list(map(int, args.weights.split(",")))
        print("Weights being used is ", args.weights)

    if args.epsilon is not None:
        args.output_dir = args.output_dir + "_epsilon{:.4f}".format(args.epsilon)

    if args.feature_size != 2048:
        args.output_dir = args.output_dir + "_ft_size{}".format(args.feature_size)

    if args.num_samples_frames is not None:
        args.output_dir = args.output_dir + "_avdiv{}".format(args.num_samples_frames)

    if args.epochs_unsuper:
        args.output_dir = args.output_dir + f"_epu{args.epochs_unsuper}"

    if args.dataset_name == "50salads":
        if args.epochs_unsuper is None:
            if args.semi_per == 0.05:
                args.epochs_unsuper = 30
            else:
                args.epochs_unsuper = 30
        if args.epochs is None:
            args.epochs = 400
        if args.lr_unsuper is None:
            args.lr_unsuper =3e-4
        if args.lr_proj is None:
            args.lr_proj = 1e-2
        if args.lr_main is None:
            args.lr_main = 1e-5
        if args.gamma_proj is None:
            args.gamma_proj = 0.1
        if args.gamma_main is None:
            args.gamma_main = 5
        args.steps_proj = [50, 100]
        args.steps_main = [50, 100]
        if args.chunk_size is None:
            args.chunk_size = 20
        if args.max_frames_per_video is None:
            args.max_frames_per_video = 960
        # if args.lr is None:
        #     args.lr = 3e-4
        if args.wd is None:
            args.wd = 1e-3
        args.batch_size = 5
        args.num_class = 19
        args.back_gd = ['action_start', 'action_end']
        if args.weights is None:
            args.weights = [1., 1., 1., 1., 1., 1.]
        if args.epsilon is None:
            args.epsilon = 0.05
        if args.delta is None:
            args.delta = 0.05
        args.high_level_act_loss = False
        if args.num_samples_frames is None:
            args.num_samples_frames = 80
    elif args.dataset_name == "breakfast":
        if args.epochs_unsuper is None:
            args.epochs_unsuper = 20
        if args.epochs is None:
            args.epochs = 200
        if args.lr_proj is None:
            if args.semi_per == 0.05:
                args.lr_proj = 1e-2
            else:
                args.lr_proj = 1e-2
        if args.lr_main is None:
            if args.semi_per == 0.05:
                args.lr_main = 3e-5
            else:
                args.lr_main = 1e-5
        if args.gamma_proj is None:
            args.gamma_proj = 0.1
        if args.gamma_main is None:
            args.gamma_main = 2

        args.steps_proj = [50]
        args.steps_main = [50]
        if args.lr_unsuper is None:
            args.lr_unsuper = 5e-5
        if args.chunk_size is None:
            args.chunk_size = 10
        if args.max_frames_per_video is None:
            args.max_frames_per_video = 600
        # if args.lr is None:
        #     args.lr = 1e-4
        if args.wd is None:
            args.wd = 3e-3
        args.batch_size = 50
        args.num_class = 48
        args.back_gd = ['SIL']
        if args.weights is None:
            args.weights = [1., 1., 1., 1., 1., 1.]
        if args.epsilon is None:
            args.epsilon = 0.03
        if args.delta is None:
            args.delta = 0.03
        args.high_level_act_loss = True
        if args.num_samples_frames is None:
            args.num_samples_frames = 30
   
    elif args.dataset_name == "gtea":
        if args.epochs_unsuper is None:
            args.epochs_unsuper = 30
        if args.lr_main is None:
            args.lr_main = 1e-5
        if args.lr_proj is None:
            args.lr_proj = 1e-2
        if args.gamma_main is None:
            args.gamma_main = 5
        if args.gamma_proj is None:
            args.gamma_proj = 0.5
       
        args.epochs = 400
        args.steps_proj = [50, 100]
        args.steps_main = [50, 100]
       
        if args.chunk_size is None:
            args.chunk_size = 4
        if args.max_frames_per_video is None:
            args.max_frames_per_video = 600
        if args.lr_unsuper is None:
            args.lr_unsuper =1e-4
        if args.wd is None:
            args.wd = 3e-4
        args.batch_size = 4
        args.num_class = 11
        args.back_gd = ['background']
        if args.weights is None:
            args.weights = [1., 1., 1., 1., 1., 1.]
        if args.epsilon is None:
            args.epsilon = 0.02
        if args.delta is None:
            args.delta = 0.02
        args.high_level_act_loss = False
        if args.num_samples_frames is None:
            args.num_samples_frames = 40

    step_p_str = "_".join(map(str, args.steps_proj))
    step_m_str = "_".join(map(str, args.steps_main))

    optim_sche_format = f"lrp_{args.lr_proj}_lrm_{args.lr_main}_gp_{args.gamma_proj}_gm_{args.gamma_main}_sp_{step_p_str}_sm_{step_m_str}"
    args.output_dir = args.output_dir + optim_sche_format

    args.output_dir = args.output_dir + "/"
    print("printing in output dir = ", args.output_dir)
    if args.getOutDir:
        import sys
        sys.exit(1)

    args.project_name="{}-split{}".format(args.dataset_name, args.split_number)
    if args.semi_per >= 1:
        args.train_split_file = args.base_dir + "/splits/train.split{}.bundle".format(args.split_number)
    else:
        args.train_split_file = args.base_dir + "/semi_supervised/train.split{}_amt{}.bundle".format(args.split_number, args.semi_per)  #train中部分用来训练
        print("train split file name = ", args.train_split_file)
        args.unsupervised_train_split_file = args.base_dir + "/splits/train.split{}.bundle".format(args.train_split)

    args.test_split_file = args.base_dir + "/splits/test.split{}.bundle".format(args.split_number)
    if args.features_file_name is None:
       args.features_file_name = args.base_dir + "/features/"
    args.ground_truth_files_dir = args.base_dir + "/groundTruth/"
    args.label_id_csv = args.base_dir + "mapping.csv"
    args.all_files = args.base_dir + "/splits/all_files.txt"
    args.base_test_split_file = args.base_dir + "/splits/test.split{}.bundle"
    return args


def dump_actual_true_data(args):
    '''
     true label to pseudo_labels_dir
    :param args:
    :return:
    '''
    if not os.path.exists(args.pseudo_labels_dir):
        os.mkdir(args.pseudo_labels_dir)
    # os.system('rm -rf ' + args.pseudo_labels_dir + "/*txt")
    labeled_data_files = open(args.train_split_file).read().split("\n")[0:-1] #rgb-07-1.txt， rgb-17-1.txt， rgb-15-2.txt
    for ele in labeled_data_files:
        print('cp ' + args.ground_truth_files_dir + ele + " "  + args.pseudo_labels_dir)
        os.system('cp ' + args.ground_truth_files_dir + ele + " "  + args.pseudo_labels_dir) #拷贝GT到.../pseudo_labels_dir/
        video_id = ele.split(".txt")[0]

        filename = os.path.join(args.ground_truth_files_dir, video_id + ".txt")
        with open(filename, 'r') as f:
            recog_content = f.read().split('\n')[0:-1]
            f.close()
        print("len labels-----",len(recog_content))

        out_path = os.path.join(args.pseudo_labels_dir, video_id + "_select.txt")
        with open(out_path, "w") as fp:
            fp.write("\n".join([str(1)] * len(recog_content)))
            fp.write("\n")

        out_path_nl_mask = os.path.join(args.pseudo_labels_dir, video_id + "_nlmask.txt")
        nlmask_arr = torch.ones((len(recog_content), args.num_class)) #[len, 19]
        np.savetxt(out_path_nl_mask, nlmask_arr)

    new_files = glob.glob(args.pseudo_labels_dir + "/*.txt")  #3 txt
    print(f"Dumped {len(new_files)} into {args.pseudo_labels_dir} directory")


def get_label_idcsv(args):
    '''
    return GT dict
    '''
    df = pd.read_csv(args.label_id_csv)
    label_id_to_label_name = {}
    label_name_to_label_id_dict = {}
    for i, ele in df.iterrows():
        label_id_to_label_name[ele.label_id] = ele.label_name
        label_name_to_label_id_dict[ele.label_name] = ele.label_id
    return label_id_to_label_name, label_name_to_label_id_dict

def dump_pseudo_labels(video_id, video_value, label_id_to_label_name, args):

    pred_value = video_value[0]  #[output_pred, count, new_mi]
    video_path = os.path.join(args.ground_truth_files_dir, video_id + ".txt")
    with open(video_path, 'r') as f:
        recog_content = f.read().split('\n')[0:-1]  # framelevel recognition is in 6-th line of file
        f.close()

    label_name_arr = [label_id_to_label_name[i.item()] for i in pred_value]
    new_label_name_expanded = [] # np.empty(len(recog_content), dtype=np.object_)
    for i, ele in enumerate(label_name_arr):
        st = i * args.chunk_size
        end = st + args.chunk_size
        if end > len(recog_content):
            end = len(recog_content)
        for j in range(st, end):
            new_label_name_expanded.append(ele)    #labels to original length
        if len(new_label_name_expanded) >= len(recog_content):
            break

    out_path = os.path.join(args.pseudo_labels_dir, video_id + ".txt")
    with open(out_path, "w") as fp:
        fp.write("\n".join(new_label_name_expanded))
        fp.write("\n")


def dump_pseudo_labels_selection(video_id, video_value, label_id_to_label_name, args):

    select_idx = video_value[3]  # [output_pred, count, new_min, selected_idx, interm_nl_mask]
    interm_nl_mask = video_value[4] #[19,count]
    video_path = os.path.join(args.ground_truth_files_dir, video_id + ".txt")
    with open(video_path, 'r') as f:
        recog_content = f.read().split('\n')[0:-1]  # framelevel recognition is in 6-th line of file
        f.close()

    new_label_name_expanded = []
    nl_mask_expanded = []
    for i, ele in enumerate(select_idx):
        st = i * args.chunk_size
        end = st + args.chunk_size
        if end > len(recog_content):
            end = len(recog_content)
        for j in range(st, end):
            new_label_name_expanded.append(str(ele))
            nl_mask_expanded.append(interm_nl_mask[:,i])
        if len(new_label_name_expanded) >= len(recog_content):
            break

    out_path = os.path.join(args.pseudo_labels_dir, video_id + "_select.txt")
    with open(out_path, "w") as fp:
        fp.write("\n".join(new_label_name_expanded))
        fp.write("\n")

    out_path_nl_mask = os.path.join(args.pseudo_labels_dir, video_id + "_nlmask.txt")
    nl_mask_expanded = np.stack(nl_mask_expanded)  #[len(allframe), 19 ]
    # print("nl_mask_expanded----", nl_mask_expanded.shape)
    np.savetxt(out_path_nl_mask, nl_mask_expanded)


def get_unlabbeled_data_and_dump_pseudo_labels(args, model, device):
    '''
    :param args:
    :param model:
    :param device:
    :return:
    '''
    label_id_to_label_name, _ = get_label_idcsv(args)
    model.eval() #-------------------------

    if not args.no_uncertainty:
        f_pass = 10
        enable_dropout(model)  #enable droupout
    else:
        f_pass = 1

    all_files_data = open(args.all_files).read().split("\n")[0:-1]
    train_file_dumped = open(args.train_split_file).read().split("\n")[0:-1]
    validation_data_files = open(args.test_split_file).read().split("\n")[0:-1]

    # full_minus_train_dataset = list(set(all_files_data) - set(train_file_dumped)- set(validation_data_files))
    full_minus_train_dataset = list(set(all_files_data) - set(train_file_dumped))
    unlabeled_dataset = get_data(args, full_minus_train_dataset, train=False, pseudo_data=False)
    unlabeled_dataset_loader = make_loader(unlabeled_dataset, batch_size=args.batch_size, train=False)
    
    results_dict = {}
    pseudo_idx = []
    pseudo_target = []
    pseudo_maxstd = []
    gt_target = []
    idx_list = []
    gt_list = []
    target_list = []
    nl_mask = []

    for i, unlabelled_item in enumerate(unlabeled_dataset_loader):
        unlabelled_data_features = unlabelled_item[0].to(device).permute(0, 2, 1)
        unlabelled_data_count = unlabelled_item[1]
        out_prob = []
        out_prob_nl = []
        for _ in range(f_pass):
            unlabelled_data_output = model(unlabelled_data_features, args.weights)
            # print("unlabelled_data_features-------", unlabelled_data_features.shape)
            unlabelled_output_probabilities = torch.softmax(unlabelled_data_output[1], dim=1) #torch.Size([47, 19, 960])
            out_prob.append(unlabelled_output_probabilities)

            unlabelled_output_probabilities_nl = torch.softmax(unlabelled_data_output[1] / 2.0, dim=1) #torch.Size([47, 19, 960])
            out_prob_nl.append(unlabelled_output_probabilities_nl)

        out_prob = torch.stack(out_prob)  # [10, 47, 19, 960]
        print("out_prob-----", out_prob.shape)
        out_prob_nl = torch.stack(out_prob_nl)

        out_std = torch.std(out_prob, dim=0)  # [47, 19, 960]
        out_std_nl = torch.std(out_prob_nl, dim=0)

        out_prob = torch.mean(out_prob, dim=0)  # [47, 19, 960]
        out_prob_nl = torch.mean(out_prob_nl, dim=0)  # [47, 19, 960]

        for output_prob, output_std, output_std_nl, output_prob_nl, vn, count in zip(out_prob, out_std, out_std_nl, out_prob_nl,  unlabelled_item[5], unlabelled_data_count):
            max_prob_out = torch.max(output_prob[:, :count], dim=0)[0].squeeze().detach().cpu().numpy() #[count] 每帧的最大概率
            # print("max_prob_out----", np.max(max_prob_out) )
            output_prob_nl2 = output_prob_nl[:, :count].detach().cpu().numpy()

            # choose positive
            # selected_idx = (max_prob_out >= 0.9)
            # selected_idx = selected_idx.astype(int)
            # print("len unique selected_idx---", len(np.unique(selected_idx)) )

            output_pred = torch.argmax(output_prob[:, :count], dim=0).squeeze().detach().cpu().numpy() #label-idx [count]
            new_min = np.mean(max_prob_out)

            max_std = output_std[:,:count].gather(0, torch.from_numpy(output_pred).to(device).view(1, -1))  # [1, count]
            max_std = max_std.detach().cpu().numpy()
            output_std_nl2 = output_std_nl[:,:count].detach().cpu().numpy() #[19, count]


            # selecting negative pseudo-labels
            interm_nl_mask = ((output_std_nl2 < args.kappa_n) * ( output_prob_nl2< args.tau_n)) * 1  # [19,count]

            # manually setting the argmax value to zero
            for enum, item in enumerate(output_pred):
                interm_nl_mask[item, enum] = 0

            # selecting positive pseudo-labels
            if not args.no_uncertainty:
                selected_idx = (max_prob_out >= args.tau_p) * (np.squeeze(max_std) < args.kappa_p)  # [count] #[False, false---]
                selected_idx = selected_idx.astype(int)
                # print("sum selected_idx-----", np.sum(selected_idx))
            else:
                selected_idx = max_prob_out >= args.tau_p
                selected_idx = selected_idx.astype(int)



            if vn in results_dict:
                prev_pred, prev_count, prev_min, prev_selected_idx, prev_interm_nl_mask = results_dict[vn]
                output_pred = np.concatenate([prev_pred, output_pred])
                selected_idx = np.concatenate([prev_selected_idx, selected_idx])
                interm_nl_mask = np.concatenate([prev_interm_nl_mask, interm_nl_mask], 1) #[19. count+count2]
                count = count + prev_count
                new_min = np.mean([prev_min, new_min])
            
            results_dict[vn] = [output_pred, count, new_min, selected_idx, interm_nl_mask]

    sort_video_values = sorted(results_dict.items(), key=lambda x: x[1][2], reverse=True) #降序 len =47 [（videoname,[output_pred, count, new_mi] ）,...]

    videos_labelled = []
    high_level_dict = {}
    # per_high_level_act_budget = config.budget / 10
    videos_added = 0
    for i in sort_video_values:
        #Write pseudo-labels for each video in turn to txt
        videos_added += 1
        dump_pseudo_labels(i[0], i[1], label_id_to_label_name, args)  # get pseudo_labels .original length
        dump_pseudo_labels_selection(i[0], i[1], label_id_to_label_name, args)
        videos_labelled.append(i[0] + ".txt")

    new_files = glob.glob(args.pseudo_labels_dir + "/*.txt")
    print(f"Contains {len(new_files)} into {args.pseudo_labels_dir} directory")
    

def model_pipeline():
    origargs = my_parser.parse_args()

    if origargs.iter_num is None:
        origargs.iter_num = [1,2,3,4]
    else:
        origargs.iter_num = origargs.iter_num

    if origargs.dataset_name is None:
        origargs.dataset_name = origargs.base_dir.split("/")[-2]
        print(f"Picked up last directory name to be dataset name {origargs.dataset_name}")

    if not os.path.exists(origargs.output_dir):
        os.mkdir(origargs.output_dir)
        print(f"Created the directory {origargs.output_dir}")

    if origargs.dataset_name == "breakfast":
        origargs.num_class = 48
    elif origargs.dataset_name == "50salads":
        origargs.num_class = 19
    elif origargs.dataset_name == "gtea":
        origargs.num_class = 11

    # Device argsuration
    if origargs.cudad is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = origargs.cudad

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, model_d = get_model(origargs)
    model = model.to(device)
    model_d = model_d.to(device)

    # model.load_state_dict(torch.load(origargs.model_wt))   #from unsupervised representation learning model
    model.load_state_dict(load_unsupervised_model(origargs.model_wt))
    print(f"Loaded model with successfully from path {origargs.model_wt}")

    for iter_n in origargs.iter_num:
        print("iter_n----------", iter_n)
        hyperparameters = copy.deepcopy(origargs)

        set_seed()
        hyperparameters = get_out_dir(hyperparameters, iter_n)
        if not os.path.exists(hyperparameters.output_dir):
            os.mkdir(hyperparameters.output_dir)
        args = hyperparameters

        # initial model_d
        # torch.nn.init.xavier_uniform_(model_d.model[0].weight)
        # torch.nn.init.xavier_uniform_(model_d.model[3].weight)

        train_loader, test_loader, criterion, optimizer_group, postprocessor, scheduler_group = make(args, model, model_d, device)

        # DL ce loss + DL contrastive loss + DL nca loss
        # training on labeled data
        model = train(device, model, model_d, train_loader, criterion, optimizer_group[:2], args, test_loader, postprocessor, scheduler_group, unsupervised=False, onlyunsupervised=False, iter=iter_n)


        print("------------supervised training on DL finished--------\n")

        model.load_state_dict(load_best_model_semi(args))
        acc, avg_acc, _ = test(device, model, test_loader, criterion, postprocessor, args, args.epochs + 1, '', False, False, model_d)
        model_d.load_state_dict(load_best_model_semi_model_d(args))

        if iter_n == origargs.iter_num[-1]:
            break

        # Create unsupervised directory and dump current 5% data and rest model evaluation data
        args.pseudo_labels_dir = os.path.join(args.output_dir, 'pseudo_labels_dir') + "/"
        dump_actual_true_data(args)
        get_unlabbeled_data_and_dump_pseudo_labels(args, model, device)

        #'''
        # Train the unsupervised model on DL DU (with pseudo_labels)
        print("--------------Train the unsupervised model -------\n")

        # # initial model_d
        # torch.nn.init.xavier_uniform_(model_d.model[0].weight)
        # torch.nn.init.xavier_uniform_(model_d.model[3].weight)

        all_files_data = open(args.all_files).read().split("\n")[0:-1]
        unsuper_traindataset = get_data(args, all_files_data, train=True, pseudo_data=True)
        unsuper_trainloader = make_loader_withselect(unsuper_traindataset, batch_size=args.batch_size, train=True)
        unsuper_testdataset = get_data(args, all_files_data, train=False, pseudo_data=False)
        unsuper_testloader = make_loader(unsuper_testdataset, batch_size=args.batch_size, train=False)

        # D_all contrastive loss + D_all nca loss
        model = train(device, model, model_d, unsuper_trainloader, criterion, [optimizer_group[2]], args, unsuper_testloader, None, None, unsupervised=True, onlyunsupervised=True, iter=iter_n)


        #load best unsupervised model
        model.load_state_dict(load_best_model_semi(args,"unsuper"))
        model_d.load_state_dict(load_best_model_semi_model_d(args, "unsuper"))

    return model

# def load_best_model(args):
#     return torch.load(args.output_dir + '/best_' + args.dataset_name + '_c2ftcn.wt')

def make(args, model, model_d, device):
    # Make the data
    all_train_data_files = open(args.train_split_file).read().split("\n")[0:-1]
    # labeled_data_files = open(args.train_split_file).read().split("\n")[0:-1]  # rgb-07-1.txt， rgb-17-1.txt， rgb-15-2.txt
    print("train_data_files----", all_train_data_files)

    if len(all_train_data_files[-1]) <= 1:
        all_train_data_files = all_train_data_files[0:-1]
        print("all_train_data_files----", all_train_data_files)

    print("Length of files picked up for semi-supervised training is ", len(all_train_data_files))
    validation_data_files = open(args.test_split_file).read().split("\n")[0:-1]
    print("Length of files picked up for semi-supervised validation is ", len(validation_data_files))

    train, test = get_data(args, all_train_data_files, train=True), get_data(args, validation_data_files, train=False)
    train_loader = make_loader(train, batch_size=args.batch_size, train=True)
    test_loader = make_loader(test, batch_size=args.batch_size, train=False)

    # Make the model
    # model = get_model(args).to(device)
    
    # num_params = sum([p.numel() for p in model.parameters()])
    # print("Number of parameters = ", num_params/1e6, " million")

    # Make the loss and optimizer
    criterion = get_criterion(args)

    set_params_for_proj = set(model.outc0.parameters()) | set(model.outc1.parameters()) | \
                        set(model.outc2.parameters()) | set(model.outc3.parameters()) | \
                        set(model.outc4.parameters()) | set(model.outc5.parameters())

    set_params_for_proj2 =  model.parameters()

    set_params_main_model = set(model.parameters()) - set_params_for_proj

    #nca model
    params_withdisc = list(set_params_main_model) + list(model_d.parameters())

    # optimizer_group = [torch.optim.Adam(list(set_params_for_proj), lr=args.lr_proj, weight_decay=args.wd),
    #                    torch.optim.Adam(params_withdisc, lr=args.lr_main, weight_decay=args.wd),
    #                    torch.optim.Adam(list(set_params_main_model), lr=args.lr_unsuper, weight_decay=args.wd)]

    # optimizer_group = [torch.optim.Adam(set_params_for_proj2, lr=args.lr_proj, weight_decay=args.wd),
    #                    torch.optim.Adam(params_withdisc, lr=args.lr_main, weight_decay=args.wd),
    #                    torch.optim.Adam(list(set_params_main_model), lr=args.lr_unsuper, weight_decay=args.wd)]

    # unspervised need update model_d
    optimizer_group = [torch.optim.Adam(list(set_params_for_proj), lr=args.lr_proj, weight_decay=args.wd),
                       torch.optim.Adam(params_withdisc, lr=args.lr_main, weight_decay=args.wd),
                       torch.optim.Adam(params_withdisc, lr=args.lr_unsuper, weight_decay=args.wd)]


    scheduler_group = [torch.optim.lr_scheduler.MultiStepLR(optimizer_group[0], milestones=args.steps_proj, gamma=args.gamma_proj),
                        torch.optim.lr_scheduler.MultiStepLR(optimizer_group[1], milestones=args.steps_main, gamma=args.gamma_main)]
    # scheduler_group = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_group[0], factor=0.1, verbose=True),
    #                     torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_group[1], factor=2, verbose=True)]
    
    # postprocessor declaration
    postprocessor = PostProcess(args)
    postprocessor = postprocessor.to(device)
    
    return train_loader, test_loader, criterion, optimizer_group, postprocessor, scheduler_group


def addpesu_make(args, model,model_d, device):

    # Make the data
    all_train_data_files = open(args.all_files).read().split("\n")[0:-1]   #all_files
    if len(all_train_data_files[-1]) <= 1:
        all_train_data_files = all_train_data_files[0:-1]
    print("Length of files picked up for semi-supervised training is ", len(all_train_data_files))
    validation_data_files = open(args.test_split_file).read().split("\n")[0:-1]
    print("Length of files picked up for semi-supervised validation is ", len(validation_data_files))

    part_train_data_files = list(set(all_train_data_files)- set(validation_data_files))

    train, test = get_data(args, part_train_data_files, train=True, pseudo_data=True), get_data(args, validation_data_files, train=False, pseudo_data=False)
    # train, test = get_data(args, all_train_data_files, train=True, pseudo_data=True), get_data(args, validation_data_files, train=False, pseudo_data=False)
    train_loader = make_loader_withselect(train, batch_size=args.batch_size, train=True)
    test_loader = make_loader(test, batch_size=args.batch_size, train=False)

    # Make the model
    # model = get_model(args).to(device)

    # num_params = sum([p.numel() for p in model.parameters()])
    # print("Number of parameters = ", num_params/1e6, " million")

    # Make the loss and optimizer
    criterion = get_criterion(args)

    # 只训练model中的最后线性层
    set_params_for_proj = set(model.outc0.parameters()) | set(model.outc1.parameters()) | \
                          set(model.outc2.parameters()) | set(model.outc3.parameters()) | \
                          set(model.outc4.parameters()) | set(model.outc5.parameters())

    set_params_main_model = set(model.parameters()) - set_params_for_proj
    params_withdisc = list(set_params_main_model) + list(model_d.parameters())

    optimizer_group = [torch.optim.Adam(list(set_params_for_proj), lr=args.lr_proj, weight_decay=args.wd),
                       torch.optim.Adam(params_withdisc, lr=args.lr_main, weight_decay=args.wd),
                       torch.optim.Adam(params_withdisc, lr=args.lr_unsuper, weight_decay=args.wd)]

    scheduler_group = [
        torch.optim.lr_scheduler.MultiStepLR(optimizer_group[0], milestones=args.steps_proj, gamma=args.gamma_proj),
        torch.optim.lr_scheduler.MultiStepLR(optimizer_group[1], milestones=args.steps_main, gamma=args.gamma_main)
        ]
    # scheduler_group = [torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_group[0], factor=0.1, verbose=True),
    #                     torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_group[1], factor=2, verbose=True)]

    # postprocessor declaration
    postprocessor = PostProcess(args)
    postprocessor = postprocessor.to(device)

    return train_loader, test_loader, criterion, optimizer_group, postprocessor, scheduler_group


def get_criterion(args):
    return CriterionClass(args)

def get_data(args, split_file_list, train=True, pseudo_data=False):
    '''
        unlabeled_dataset = get_data(args, full_minus_train_dataset, train=False, pseudo_data=False)

    '''
    if train is True:
        fold='train'
    else:
        fold='val'
    dataset = Breakfast(args, pseudo_data, fold=fold, list_data=split_file_list)
    return dataset


def make_loader(dataset, batch_size, train=True):
    def _init_fn(worker_id):
        np.random.seed(int(seed))
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=train,
                                         pin_memory=True, num_workers=0, collate_fn=collate_fn_override,
                                         worker_init_fn=_init_fn)  #  ori num_workers=7
    return loader


def make_loader_withselect(dataset, batch_size, train=True):
    def _init_fn(worker_id):
        np.random.seed(int(seed))
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size,
                                         shuffle=train,
                                         pin_memory=True, num_workers=0, collate_fn=collate_fn_override_withselect,
                                         worker_init_fn=_init_fn)  #  ori num_workers=7
    return loader

def get_model(args):
    set_seed()
    model_discr =  NCA_discriminator(1536)
    return C2F_TCN(n_channels=args.feature_size, n_classes=args.num_class, n_features=args.outft), model_discr



def unsupervised_test(args, epoch, model, test_loader, device):
    set_seed()
    dump_dir = args.output_dir + "/features_dump/"
    dump_featres = DumpFeatures(args)
    dump_featres.forward(model, test_loader, device, dump_dir, wts=args.weights)

    acc, all_result = get_linear_acc(args.label_id_csv, dump_dir, args.ground_truth_files_dir, args.perdata, 
                                     args.base_test_split_file, args.chunk_size, False, False)
    # acc = test(model, test_loader, criterion, postprocessor, args, args.epochs, '')
    print_str =  f"Epoch{epoch}: Linear f1@10, f1@25, f1@50, edit, MoF = " + \
          f"{all_result[1]:.1f} & {all_result[2]:.1f} & {all_result[3]:.1f} & {all_result[0]:.1f} & {acc:.1f}\n"
    print(print_str)

    with open(args.output_dir + "/run_summary.txt", "a+") as fp:
        fp.write(print_str)
    # avg_acc = (acc + all_result[0] + all_result[1] + all_result[2]  + all_result[3]) / 5
    avg_acc = (acc + all_result[0]) / 2

    return acc, avg_acc, all_result


def train(device, model, model_d, loader, criterion, optimizer_group, args, test_loader, postprocessor, scheduler_group, unsupervised, onlyunsupervised, iter):
    '''
    model = train(device, model, train_loader, criterion, optimizer_group[:2], args, test_loader, postprocessor, scheduler_group, unsupervised=False)
    '''

    if (unsupervised == True and onlyunsupervised == True):
        epochs = args.epochs_unsuper
        print_epochs = 10
        prefix = "unsuper"
    else:
        epochs = args.epochs
        print_epochs = 2
        prefix = ""

    best_acc = 0
    avg_best_acc = 0
    accs = []
    start_epoch = 0

    if onlyunsupervised is False:
        if iter==1:
            start_epoch, model, model_d, optimizer = resume_checkpoint(args, model, model_d)

    for epoch in range(start_epoch, epochs + 1):
        print("#############epoch##################", epoch)
        model.train()
        model_d.train()
        for i, item in enumerate(loader):
            samples = item[0].to(device).permute(0, 2, 1) #torch.Size([3, 2048, 960])
            count = item[1].to(device)
            labels = item[2].to(device) #[3,960] [50,960]
            # print('labels----', labels)

            #ablation_4
            # if (unsupervised == True and onlyunsupervised ==True):
            #     select_index = item[8].to(device) #[5,960]
            #     nl_mask = item[-1].to(device) #[5, 960, 19]
            # else:
            #     select_index = torch.ones(labels.shape, dtype=torch.long).to(device)  #all=1
            #     nl_mask = torch.ones((labels.shape[0], labels.shape[1], args.num_class), dtype=torch.long).to(device)  #all=1
                # print('nl_mask----', nl_mask.shape)

            # ablation_2
            select_index = torch.ones(labels.shape, dtype=torch.long).to(device)  # all=1
            nl_mask = torch.ones((labels.shape[0], labels.shape[1], args.num_class), dtype=torch.long).to(device)  # all=1

            if args.dataset_name == 'breakfast':
               activity_labels = np.array([name.split('_')[-1] for name in item[5]])
            elif args.dataset_name == '50salads':
               activity_labels = None 
            elif args.dataset_name == 'gtea':
               activity_labels = None 

            # Forward pass ➡
            projection_out, pred_out, achor_out = model(samples, args.weights) #torch.Size([3, 1536, 960])  torch.Size([3, 19, 960])

            loss_dict = criterion(count, projection_out, pred_out, labels, activity_labels, item[0], unsupervised, onlyunsupervised, achor_out, select_index, nl_mask, model_d ) #cross en loss

            loss = loss_dict['full_loss']

            # Backward pass ⬅
            for optim in optimizer_group:
                optim.zero_grad()
            loss.backward()

            # Step with optimizer
            for optim in optimizer_group:
                optim.step()

            train_log(loss_dict, epoch)
            # time.sleep(0.01)

        if scheduler_group is not None:
            # Step for scheduler
            for sch in scheduler_group:
                sch.step()

        if epoch % print_epochs == 0:
            saveepochcheckpont_semi(args, epoch, model,prefix)

        if epoch % print_epochs == 0:
            if unsupervised is True and onlyunsupervised is True:
               acc, avg_acc,  all_result = unsupervised_test(args, epoch, model, test_loader, device)   #liner evaluation
            else:
               print("###############----test----###############")
               acc, avg_acc, overlap_scores = test(device, model, test_loader, criterion, postprocessor, args, epoch, prefix, unsupervised, onlyunsupervised, model_d) # predict label evaluation
            # if acc > best_acc:
            if avg_acc > best_acc:
                best_acc = avg_acc
                saveepochcheckpont_semi_best(args, epoch, model, prefix)
                saveepochcheckpont_semi_best_model_d(args, epoch, model_d, prefix)

            accs.append(avg_acc)
        accs.sort(reverse=True)
        print(f'Best avg accuracies till now -> {" ".join(["%.2f"%item for item in accs[:3]])}')
        # time.sleep(0.003)
    return model


def train_log(loss_dict, epoch):
    final_dict = {"epoch": epoch}
    final_dict.update(loss_dict)
    print(f"Loss after " + str(epoch).zfill(5) + f" examples: {loss_dict['full_loss']:.3f}")

def test(device, model, test_loader, criterion, postprocessors, args, epoch, dump_prefix, unsupervised, onlyunsupervised, model_d):
    model.eval()
    # Run the model on some test examples
    with torch.no_grad():
        correct, total = 0, 0
        avg_loss = []
        avg_total_loss = []
        for i, item in enumerate(test_loader):
            samples = item[0].to(device).permute(0, 2, 1)  #torch.Size([10, 2048, 960])
            print("test samples------", samples.shape)

            count = item[1].to(device)
            labels = item[2].to(device)


            select_index = torch.ones(labels.shape, dtype=torch.long).to(device)  # all=1
            nl_mask = torch.ones((labels.shape[0], labels.shape[1], args.num_class), dtype=torch.long).to(device)  # all=1
            src_mask = torch.arange(labels.shape[1], device=labels.device)[None, :] < count[:, None] # torch.Size([10, 960])

            src_mask = src_mask.to(device)

            if args.dataset_name == 'breakfast':
               activity_labels = np.array([name.split('_')[-1] for name in item[5]])
            elif args.dataset_name == '50salads':
               activity_labels = None 
            elif args.dataset_name == 'gtea':
               activity_labels = None 

            # Forward pass ➡
            projection_out, pred_out, achor_out  = model(samples, args.weights)
            
            loss_dict = criterion(count, projection_out, pred_out, labels, activity_labels, item[0], unsupervised, onlyunsupervised, achor_out, select_index, nl_mask, model_d) #cross en loss

            loss = loss_dict['ce_loss']
            avg_loss.append(loss_dict['ce_loss'])
            avg_total_loss.append(loss_dict['full_loss'])
            
            pred = torch.argmax(pred_out, dim=1)
            correct += float(torch.sum((pred == labels) * src_mask).item())
            total += float(torch.sum(src_mask).item())
            postprocessors(pred_out, item[5], labels, count)
            
        # Add postprocessing and check the outcomes
        path = os.path.join(args.output_dir, dump_prefix + "predict_" + args.dataset_name)
        if not os.path.exists(path):
            os.mkdir(path)
        postprocessors.dump_to_directory(path)
        final_edit_score, map_v, overlap_scores = calculate_mof(args.ground_truth_files_dir, path, args.back_gd)
        # avg_score = (map_v + final_edit_score + overlap_scores[0] + overlap_scores[1] + overlap_scores[2]) / 5  #ave of all metrics
        avg_score = (map_v + final_edit_score) / 2 # ave acc and edit
        postprocessors.start()
        acc = 100.0 * correct / total
        print(f"Accuracy of the model on the {total} " +f"test images: {acc}%")
        
        final_dict = {"test_accuracy": 100.0 * correct / total}
        final_dict.update({"ce_test_loss": sum(avg_loss) / len(avg_loss)})
        final_dict.update({"total_test_loss": sum(avg_total_loss) / len(avg_total_loss)})
        final_dict.update({"test_actual_acc": map_v})
        final_dict.update({"test_edit_score": final_edit_score})
        final_dict.update({"f1@50": overlap_scores[-1]})
        with open(args.output_dir + "/results_file.txt", "a+") as fp:
            print_string = "Epoch={}: {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f}\n".format(epoch, overlap_scores[0], overlap_scores[1],
                                                overlap_scores[2], final_edit_score, map_v, avg_score)
            print(print_string)
            fp.write(print_string)

        if epoch == (args.epochs + 1):
            with open(args.output_dir + "/" + dump_prefix + "final_results_file.txt", "a+") as fp:
                fp.write("Epoch={}: {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} & {:.1f} \n".format(epoch, overlap_scores[0], overlap_scores[1],
                                                    overlap_scores[2], final_edit_score, map_v, avg_score))

    # Save the model in the exchangeable ONNX format
#     torch.onnx.export(model, "model.onnx")
#     avg_score = (map_v + final_edit_score) / 2
#     avg_score = (map_v + final_edit_score + overlap_scores[0] + overlap_scores[1] +overlap_scores[2]) / 5
    return map_v, avg_score, overlap_scores

start_time = time.time()
model = model_pipeline()

end_time = time.time()
duration = end_time - start_time

mins = duration / 60 
print("Time taken to complete 4 iteration ", mins, " mins")
