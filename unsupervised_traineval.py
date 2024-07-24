import importlib
import os, sys
from sklearn.cluster import KMeans
from utility.perform_linear import get_linear_acc, get_linear_acc_on_split
from utility.dump_features import DumpFeatures
from utility.savemodel import saveepochcheckpont,saveepochcheckpont_best, load_best_model,resume_checkpoint2
import warnings
warnings.filterwarnings('ignore')

import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from model_mlp import C2F_TCN


from utils import calculate_mof
from postprocess import PostProcess
from dataset import Breakfast, collate_fn_override
from lossfunction_multibatch_smc import CriterionClass   #dynamic cluster three



seed = 42

#breakfast
# netname = '/my_triploss/run_summary_bs50_test'

#geta
# netname = '/my_triploss/run_summary_bs4_test'

#50salads
netname = '/my_triploss/run_summary_bs5_test'



my_parser = argparse.ArgumentParser()

my_parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
my_parser.add_argument('--output_dir', default="/data/3dpoint/project/action/mstcn_data/50salads/results/SMC", type=str, required=False, help="Output directory where the programs outputs (checkpoints, logs etc.) must be stored")
my_parser.add_argument('--base_dir', default="/data/3dpoint/project/action/mstcn_data/50salads/", type=str, help="Base directory where dataset's all files like features, split, ground truth is present")


# my_parser.add_argument('--output_dir', default="/data/3dpoint/project/action/mstcn_data/gtea/results/SMC", type=str, required=False, help="Output directory where the programs outputs (checkpoints, logs etc.) must be stored")
# my_parser.add_argument('--base_dir', default="/data/3dpoint/project/action/mstcn_data/gtea/", type=str, help="Base directory where dataset's all files like features, split, ground truth is present")

# my_parser.add_argument('--output_dir', default="/data/3dpoint/project/action/mstcn_data/breakfast/results/SMC", type=str, required=False, help="Output directory where the programs outputs (checkpoints, logs etc.) must be stored")
# my_parser.add_argument('--base_dir', default="/data/3dpoint/project/action/mstcn_data/breakfast/", type=str, help="Base directory where dataset's all files like features, split, ground truth is present")



my_parser.add_argument('--wd', type=float, required=False)
my_parser.add_argument('--lr', type=float, required=False)
my_parser.add_argument('--chunk_size', type=int, required=False)
my_parser.add_argument('--max_frames_per_video', type=int, required=False)
my_parser.add_argument('--weights', type=str, required=False)
my_parser.add_argument('--features_file_name', type=str, required=False)
my_parser.add_argument('--feature_size', type=int, required=False)
my_parser.add_argument('--clustCount', type=int, required=False)
my_parser.add_argument('--epsilon', type=float, required=False)
my_parser.add_argument('--tau', default=0.1, type=float)
my_parser.add_argument('--epochs', type=int, default=100, required=False)
my_parser.add_argument('--num_samples_frames', type=int, required=False)
my_parser.add_argument('--delta', type=float, required=False)
my_parser.add_argument('--outft', type=int, default=256)
my_parser.add_argument('--nohigh', action='store_true')
my_parser.add_argument('--perdata', type=int, default=100, help="Linear evaluation with amount of percentage data")
my_parser.add_argument('--getOutDir', action="store_true", help="Run program only to get where model checkpoint is stored")
my_parser.add_argument('--eval_only', default=False, action="store_true", help="Run program only to get the linear evaluation scores") #true-eval false-train
my_parser.add_argument('--no_time', default=False, action="store_true", help="Run program with no time-proximity condition")
my_parser.add_argument('--val_split', type=int, required=False, help="By default it learns on all splits and evaluates on all splits")
my_parser.add_argument('--train_split', type=str, required=False, help="By default it is all splits except val_split")
my_parser.add_argument('--cudad', type=str, required=False, help="Specify the cuda number in string which the program needs to be run on")
my_parser.add_argument('--dataset_name', type=str, required=False, help="If last directory name of base is not dataset name, then specify dataset name 50salads, breakfast or gtea")
args = my_parser.parse_args()


# Ensure deterministic behavior
def set_seed():
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
set_seed()

if args.dataset_name is None:
    args.dataset_name = args.base_dir.split("/")[-2]
    print(f"Picked up last directory name to be dataset name {args.dataset_name}")


# Device argsuration
if args.cudad is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cudad

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if args.val_split is None:
    args.val_split = "full"

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)
    print(f"Created the directory {args.output_dir}")
args.output_dir = args.output_dir + "/unsupervised_{}_split{}".format("C2FTCN", args.val_split) + netname

if args.train_split is not None:
    args.output_dir = args.output_dir + f"_ts{args.train_split}"
else:
    args.train_split = f"{args.val_split}"

if args.wd is not None:
    args.output_dir = args.output_dir + "_wd{:.5f}".format(args.wd)

if args.lr is not None:
    args.output_dir = args.output_dir + "_lr{:.6f}".format(args.lr)

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

if args.feature_size is not None:
    args.output_dir = args.output_dir + "_ft_size{}".format(args.feature_size)
else:
    args.feature_size = 2048

if args.num_samples_frames is not None:
    args.output_dir = args.output_dir + "_avdiv{}".format(args.num_samples_frames)

if args.clustCount is not None:
    args.output_dir = args.output_dir + "_clusC{}".format(args.clustCount)

if args.dataset_name == "50salads":
    if args.epsilon is None:
        args.epsilon = 0.05
    if args.clustCount is None:
        args.clustCount = 40
    if args.chunk_size is None:
        args.chunk_size = 20
    if args.max_frames_per_video is None:
        args.max_frames_per_video = 960
    if args.lr is None:
        args.lr = 1e-3
    if args.wd is None:
        args.wd = 1e-3
    args.batch_size = 5
    args.num_class = 19
    args.back_gd = ['action_start', 'action_end']
    if args.weights is None:
        args.weights = [1, 1, 1, 1, 1, 1]
    if args.delta is None:
        args.delta = 0.5
    args.high_level_act_loss = False
    if args.num_samples_frames is None:
        args.num_samples_frames = 80
elif args.dataset_name == "breakfast":
    if args.epsilon is None:
        args.epsilon = 0.03
    if args.clustCount is None:
        args.clustCount = 100
    if args.chunk_size is None:
        args.chunk_size = 10
    if args.max_frames_per_video is None:
        args.max_frames_per_video = 600
    if args.lr is None:
        args.lr = 1e-4
    if args.wd is None:
        args.wd = 3e-3
    args.batch_size = 50
    args.num_class = 48
    args.back_gd = ['SIL']
    if args.weights is None:
        args.weights = [1, 1, 1, 1, 1, 1]
    if args.delta is None:
        args.delta = 0.03
    if args.nohigh:
        args.high_level_act_loss = False
        args.output_dir = args.output_dir + "_noactloss_"
    else:
        args.high_level_act_loss = True
    if args.num_samples_frames is None:
        args.num_samples_frames = 20
elif args.dataset_name == "gtea":
    if args.epsilon is None:
        args.epsilon = 0.02
    if args.clustCount is None:
        args.clustCount = 30
    if args.chunk_size is None:
        args.chunk_size = 4
    if args.max_frames_per_video is None:
        args.max_frames_per_video = 600
    if args.lr is None:
        args.lr = 1e-3
    if args.wd is None:
        args.wd = 3e-4
    args.batch_size = 4
    args.num_class = 11
    args.back_gd = ['background']
    if args.weights is None:
        args.weights = [1, 1, 1, 1, 1, 1]
    if args.delta is None:
        args.delta = 0.02
    args.high_level_act_loss = False
    if args.num_samples_frames is None:
        args.num_samples_frames = 40

args.output_dir = args.output_dir + "/"
print("printing in output dir = ", args.output_dir)
if args.getOutDir is True:
    sys.exit(1)

args.project_name="{}-split{}".format(args.dataset_name, args.val_split)
print("project_name = ", args.project_name)


if args.val_split != "full":
    args.train_split_file = args.base_dir + "splits/train.split{}.bundle".format(args.train_split)
    print("Picking the training file from ", args.train_split_file)
else:
    args.train_split_file = args.base_dir + "splits/all_files.txt"
    print("Picking the training file from ", args.train_split_file)

args.base_test_split_file = args.base_dir + "splits/test.split{}.bundle"
# args.base_test_split_file = args.base_dir + "splits/train.split{}.bundle"


args.test_split_file = args.base_dir + "splits/all_files.txt"
# args.test_split_file = args.base_dir + "splits/train.split{}.bundle".format(args.val_split)

if args.features_file_name is None:
    args.features_file_name = args.base_dir + "features/"
args.ground_truth_files_dir = args.base_dir + "groundTruth/"
args.label_id_csv = args.base_dir + "mapping.csv"


def model_pipeline(hyperparameters):
    if not os.path.exists(hyperparameters.output_dir):
        os.mkdir(hyperparameters.output_dir)

    args = hyperparameters
    model, train_loader, test_loader, criterion, optimizer, postprocessor = make(args)
    # print('model---', model)

    if not args.eval_only:
        # and use them to train the model
        train(model, train_loader, criterion, optimizer, args, test_loader, postprocessor)

    # and test its final performance
    model.load_state_dict(load_best_model(args))

    set_seed()
    dump_dir = args.output_dir + "/features_dump/"
    dump_featres = DumpFeatures(args)
    dump_featres.forward(model, test_loader, device, dump_dir, wts=args.weights)

    if args.val_split != "full":
        val_split_file = args.base_test_split_file.format(args.val_split)
        acc, all_result = get_linear_acc_on_split(val_split_file, args.train_split_file, args.label_id_csv, dump_dir, args.ground_truth_files_dir, 
                                                  args.perdata, args.chunk_size, False, False)
    else:
        acc, all_result = get_linear_acc(args.label_id_csv, dump_dir, args.ground_truth_files_dir, args.perdata, 
                                         args.base_test_split_file, args.chunk_size, False, False)
    print_str = f"Best Results:Linear f1@10, f1@25, f1@50, edit, MoF = " + \
          f"{all_result[1]:.1f} & {all_result[2]:.1f} & {all_result[3]:.1f} & {all_result[0]:.1f} & {acc:.1f}\n"
    print(print_str)

    with open(args.output_dir + "/run_summary.txt", "a+") as fp:
        fp.write(print_str)
    print(f'final_test_acc_avg:{acc:.2f}')

    return model


def make(args):
    # Make the data
    train_file= args.train_split_file
    print('train_split_file----', train_file)

    test_file = args.test_split_file
    print('test_split_file----', test_file)


    all_train_data_files = open(args.train_split_file).read().split("\n")[0:-1]
    print('all_train_data_files---',all_train_data_files )
    validation_data_files = open(args.test_split_file).read().split("\n")[0:-1]
    print('validation_data_files---',validation_data_files )


    train, test = get_data(args, all_train_data_files, train=True), get_data(args, validation_data_files, train=False)
    train_loader = make_loader(args, train, batch_size=args.batch_size, train=True)
    test_loader = make_loader(args, test, batch_size=args.batch_size, train=False)

    # Make the model
    # model = get_model(args).to(device)
    model = get_model(args)

    # using 2 Gpus
    print("Let's use", torch.cuda.device_count(), "GPUs")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model)
    model.to(device)
    
    num_params = sum([p.numel() for p in model.parameters()])
    print("Number of parameters = ", num_params/1e6, " million")

    # Make the loss and optimizer
    criterion = get_criterion(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    
    # postprocessor declaration
    postprocessor = PostProcess(args)
    postprocessor = postprocessor.to(device)
    
    return model, train_loader, test_loader, criterion, optimizer, postprocessor

def get_criterion(args):
    return CriterionClass(args)

def get_data(args, split_file_list, train=True, pseudo_data=False):
    if train is True:
        fold='train'
    else:
        fold='val'
    dataset = Breakfast(args, pseudo_data, fold=fold, list_data=split_file_list)
    return dataset


def make_loader(args, dataset, batch_size, train=True):
    def _init_fn(worker_id):
        np.random.seed(int(seed))
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=batch_size, 
                                         shuffle=train,
                                         pin_memory=True, num_workers=2, collate_fn=collate_fn_override,
                                         worker_init_fn=_init_fn)
    return loader


def get_model(args):
    set_seed()
    return C2F_TCN(n_channels=args.feature_size, n_classes=args.num_class, n_features=args.outft)
    # return MultiStageModel( dim = args.feature_size, num_classes=args.num_class)




def train(model, loader, criterion, optimizer, args, test_loader, postprocessor):
    # wandb.watch(model, criterion, log="all", log_freq=10)

    # Run training and track with wandb
    total_batches = len(loader) * args.epochs  #4 *100
    best_acc = 0
    avg_best_acc = 0
    accs = []

    #resume epoch
    start_epoch, model, optimizer = resume_checkpoint2(args, model, optimizer)
    # model.load_state_dict(checkpoint)

    for epoch in range(start_epoch, args.epochs + 1):
        print("epoch------", epoch)
        model.train()
        loss_epoch =[]
        for i, item in enumerate(loader):

            samples = item[0].to(device).permute(0, 2, 1)
            count = item[1].to(device)
            labels = item[2].to(device)
            mask  = torch.ones((labels.shape[0], args.num_class, labels.shape[1])).to(device)

            if args.dataset_name == 'breakfast':
                activity_labels = np.array([name.split('_')[-1] for name in item[5]])
            elif args.dataset_name == '50salads':
                activity_labels = None
            elif args.dataset_name == 'gtea':
                activity_labels = None

            # Forward pass ➡
            projection_out, pred_out, achor_out = model(samples, args.weights) #torch.Size([5, 1536, 960]) torch.Size([5, 19, 960])
            print("out-----", projection_out.shape, pred_out.shape, achor_out.shape)
            loss_dict = criterion(count, projection_out, pred_out, labels, activity_labels, item[0], achor_out)

            loss = loss_dict['full_loss']
            print("loss-----", loss.item())
            loss_epoch.append(loss.data.item())
            # Backward pass ⬅
            optimizer.zero_grad()
            loss.backward()


            # Step with optimizer
            optimizer.step()
            train_log(loss_dict, epoch)

        # if epoch % 5 == 0:
        #     saveepochcheckpont(args, epoch, model, optimizer)
        #
        loss_everyepoch =  sum(loss_epoch)/len(loss_epoch)
        print("loss_everyepoch-------", loss_everyepoch)
        # if (epoch % 5 == 0):
        if (epoch >=10):
            set_seed()
            dump_dir = args.output_dir + "features_dump/"
            dump_featres = DumpFeatures(args)
            dump_featres.forward(model, test_loader, device, dump_dir, wts=args.weights) #输出上采样后的特征
            saveepochcheckpont(args, epoch, model, optimizer)


            if args.val_split != "full":
                val_split_file = args.base_test_split_file.format(args.val_split)
                acc, all_result = get_linear_acc_on_split(val_split_file, args.train_split_file, args.label_id_csv, dump_dir,
                                                          args.ground_truth_files_dir, args.perdata, args.chunk_size,
                                                           False, False)
            else:
                acc, all_result = get_linear_acc(args.label_id_csv, dump_dir, args.ground_truth_files_dir, args.perdata, 
                                                 args.base_test_split_file, args.chunk_size, False, False)

            if acc >= best_acc:
                best_acc = acc
                saveepochcheckpont_best(args, epoch, model, optimizer)

            print_str = f"Epoch{epoch}: Lr, loss, Linear f1@10, f1@25, f1@50, edit, MoF = " + \
                  f"{args.lr} & {loss_everyepoch:.3f} & {all_result[1]:.1f} & {all_result[2]:.1f} & {all_result[3]:.1f} & {all_result[0]:.1f} & {acc:.1f}, best={best_acc:.1f}\n"
            print(print_str)

            with open(args.output_dir + "/run_summary.txt", "a+") as fp:
                fp.write(print_str)

            accs.append(acc)
        # torch.save(model.state_dict(), args.output_dir + netname+ '/last_' + args.dataset_name + '_c2f_tcn.wt')
        accs.sort(reverse=True)
        # scheduler.step()
        # wandb.log({'avgbest_test_acc': avg_best_acc}, epoch)
        print(f'Best accuracies till now -> {" ".join(["%.2f"%item for item in accs[:3]])}')


def train_log(loss_dict, epoch):
    final_dict = {"epoch": epoch}
    final_dict.update(loss_dict)
    print(f"Loss after " + str(epoch).zfill(5) + f" examples: {loss_dict['full_loss']:.3f}")

model = model_pipeline(args)
