import torch
import pandas as pd
import ast
import numpy as np
import h5py
from torchvision import transforms
import os
from PIL import Image
from collections import defaultdict
from itertools import chain as chain
import random

import sys
import argparse


def collate_fn_override(data):
    """
       data:
    """
    data = list(filter(lambda x: x is not None, data))
    data_arr, count, labels, clip_length, start, video_id, labels_present_arr, aug_chunk_size, targets = zip(*data)
    return torch.stack(data_arr), torch.tensor(count), torch.stack(labels), torch.tensor(clip_length),\
            torch.tensor(start), video_id, torch.stack(labels_present_arr), torch.tensor(aug_chunk_size, dtype=torch.int)

def collate_fn_override_withselect(data):
    """
       data:
    """
    data = list(filter(lambda x: x is not None, data))
    data_arr, count, labels, clip_length, start, video_id, labels_present_arr, aug_chunk_size, targets, select_index, nl_mask = zip(*data)
    return torch.stack(data_arr), torch.tensor(count), torch.stack(labels), torch.tensor(clip_length),\
            torch.tensor(start), video_id, torch.stack(labels_present_arr), torch.tensor(aug_chunk_size, dtype=torch.int),torch.stack(select_index), torch.stack(nl_mask)



class Breakfast(torch.utils.data.Dataset):
    '''

    '''

    def __init__(self, args, pseudo_data, fold, list_data, 
                 zoom_crop=(0.5, 2), smallest_cut=1.0):
        self.fold = fold
        self.max_frames_per_video = args.max_frames_per_video
        self.feature_size = args.feature_size
        self.base_dir_name = args.features_file_name
        self.frames_format = "{}/{:06d}.jpg"
        self.pseudo_data = pseudo_data
        if pseudo_data:
            self.ground_truth_files_dir = args.pseudo_labels_dir
            print("Picking up data from the pseudo directory")
        else:
            self.ground_truth_files_dir = args.ground_truth_files_dir   # False---'mstcn_data/50salads/groundTruth/'
        self.chunk_size = args.chunk_size
        self.num_class = args.num_class
        self.zoom_crop = zoom_crop
        self.smallest_cut = smallest_cut
        self.validation = True if fold == 'val' else False 
        self.args = args
        self.data = self.make_data_set(list_data)
        
        
    def make_data_set(self, data):
        #debug
        # data = data[0:2]


        df=pd.read_csv(self.args.label_id_csv)
        label_id_to_label_name = {}
        label_name_to_label_id_dict = {}
        for i, ele in df.iterrows():
            label_id_to_label_name[ele.label_id] = ele.label_name
            label_name_to_label_id_dict[ele.label_name] = ele.label_id

        # data = open(fold_file_name).read().split("\n")[:-1]
        data_arr = []
        num_video_not_found = 0
        for i, video_id in enumerate(data):
            video_id = video_id.split(".txt")[0]
            filename = os.path.join(self.ground_truth_files_dir, video_id + ".txt")
            # print("filename---", filename)

            with open(filename, 'r') as f:
                recog_content = f.read().split('\n')[0:-1]  # framelevel recognition is in 6-th line of file
                f.close()

            if self.pseudo_data:
                filename_select = os.path.join(self.ground_truth_files_dir, video_id + "_select.txt")
                select_index = np.loadtxt(filename_select)
                # print("select_index-----", select_index )

                filename_nlmask = os.path.join(self.ground_truth_files_dir, video_id + "_nlmask.txt")
                nlmask  = np.loadtxt(filename_nlmask)   #[len, 19 ]
                # print("nlmask------", nlmask.shape)

            recog_content = [label_name_to_label_id_dict[e] for e in recog_content]  #name to number [17,17----]
            
            total_frames = len(recog_content) #9694
            
            # if not os.path.exists(os.path.join(self.base_dir_name, video_id + ".npy")):
            #     print("Not found video with id", os.path.join(self.base_dir_name, video_id + ".npy"))
            #     num_video_not_found += 1
            #     continue

            start_frame_arr = []
            end_frame_arr = []
            for st in range(0, total_frames, self.max_frames_per_video * self.chunk_size):
                start_frame_arr.append(st)
                max_end = st + (self.max_frames_per_video * self.chunk_size)
                end_frame = max_end if max_end < total_frames else total_frames
                end_frame_arr.append(end_frame)

            for st_frame, end_frame in zip(start_frame_arr, end_frame_arr):

                ele_dict = {'st_frame': st_frame, 'end_frame': end_frame, 'video_id': video_id,
                            'tot_frames': (end_frame - st_frame)}
                
                ele_dict["labels"] = np.array(recog_content, dtype=int)
                if self.pseudo_data:
                    ele_dict["select_index"] = np.array(select_index, dtype=int)
                    ele_dict["nlmask"] = np.array(nlmask, dtype=int)


                data_arr.append(ele_dict)
            # data_arr.extend(ele_dict_arr)
        print("Number of videos logged in {} fold is {}".format(self.fold, len(data_arr)))
        print("Number of videos not found in {} fold is {}".format(self.fold, num_video_not_found))
        return data_arr

    def getitem(self, index):  # Try to use this for debugging purpose
        ele_dict = self.data[index]
        count = 0
        st_frame = ele_dict['st_frame']
        end_frame = ele_dict['end_frame']
        
        data_arr = torch.zeros((self.max_frames_per_video, self.feature_size)) #[960, 2048]
        label_arr = torch.ones(self.max_frames_per_video, dtype=torch.long) * -100 #[960]
        select_arr = torch.ones(self.max_frames_per_video, dtype=torch.long) * -100 #[960]

        nlmask_arr = torch.zeros((self.max_frames_per_video, self.num_class)) #[960, 19]


        image_path = os.path.join(self.base_dir_name, ele_dict['video_id'] + ".npy")
        elements = np.load(image_path) #[2048,9694]

        # elements = np.loadtxt(image_path)
        count = 0
        end_frame = min(end_frame, st_frame + (self.max_frames_per_video * self.chunk_size))
        len_video = end_frame - st_frame
        num_original_frames = np.ceil(len_video / self.chunk_size) # 9694/20=485
        
        if np.random.randint(low=0, high=2)==0 and (not self.validation):
            aug_start = np.random.uniform(low=0.0, high=1-self.smallest_cut)
            aug_len = np.random.uniform(low=self.smallest_cut, high=1-aug_start)
            aug_end = aug_start + aug_len
            min_possible_chunk_size = np.ceil(len_video/self.max_frames_per_video) #9694/960 11
            max_chunk_size = int(1.0*self.chunk_size/self.zoom_crop[0]) #40
            min_chunk_size = max(int(1.0*self.chunk_size/self.zoom_crop[1]), min_possible_chunk_size)  #11
            aug_chunk_size = int(np.exp(np.random.uniform(low=np.log(min_chunk_size), high=np.log(max_chunk_size)))) #30
            num_aug_frames = np.ceil(int(aug_len*len_video)/aug_chunk_size)  #324
            if num_aug_frames > self.max_frames_per_video:
                num_aug_frames = self.max_frames_per_video
                aug_chunk_size = int(np.ceil(aug_len*len_video/num_aug_frames))

            # semi-supervised loss_d breakfast
            # if(len_video==130):
            #     if aug_chunk_size > 18:
            #         print("aug_chunk_size > 18")
            #         aug_chunk_size = 10

            aug_translate = 0
            # if num_aug_frames<self.max_frames_per_video:
            #     aug_translate = np.random.randint(low=0.0, high=self.max_frames_per_video-num_aug_frames)
            aug_start_frame = st_frame + int(len_video*aug_start) #0
            aug_end_frame = st_frame + int(len_video*aug_end) #9694
        else:
            aug_translate, aug_start_frame, aug_end_frame, aug_chunk_size = 0, st_frame, end_frame, self.chunk_size
        
        labels_present_arr = torch.zeros(self.num_class, dtype=torch.float32)
        for i in range(aug_start_frame, aug_end_frame, aug_chunk_size):  #[0,9694, 30]
            end = min(aug_end_frame, i + aug_chunk_size)  #30
            key = elements[:, i: end]
            # key_nlmask = ele_dict["nl_mask"][i: end, :]
            values, counts = np.unique(ele_dict["labels"][i: end], return_counts=True) #[17] [30], 30 ge 17
            label_arr[count] = int(values[np.argmax(counts)])
            if self.pseudo_data:
                values_sel, counts_sel = np.unique(ele_dict["select_index"][i: end], return_counts=True) #[0,1]
                select_arr[count] = int(values_sel[np.argmax(counts_sel)])
                values_nlmask, counts_nlmask = np.unique(ele_dict["nlmask"][i: end], axis=0, return_counts=True)
                nlmask_arr[count] =  torch.tensor(values_nlmask[np.argmax(counts_nlmask)])


            labels_present_arr[label_arr[count]] = 1   #[19] class number
            data_arr[aug_translate+count, :] = torch.tensor(np.max(key, axis=-1), dtype=torch.float32)  #[960,2048]
            count += 1

        data_arr = data_arr # #[960,2048]
        # print(data_arr)
        count =count  #324
        label_arr = label_arr #[960]
        ele_tot =ele_dict['tot_frames']  #9694
        st_frame = st_frame  #0
        ele_d= ele_dict['video_id']
        labels_present_arr =labels_present_arr
        aug_chunk_size = aug_chunk_size  #30
        select_arr = select_arr
        v, c = np.unique(select_arr, return_counts=True)
        nlmask_arr = nlmask_arr #[960,19]

        if self.pseudo_data:
            return data_arr, count, label_arr, ele_dict['tot_frames'], st_frame, ele_dict['video_id'], \
                   labels_present_arr, aug_chunk_size, {"labels": label_arr}, select_arr, nlmask_arr
        else:
            return data_arr, count, label_arr, ele_dict['tot_frames'], st_frame, ele_dict['video_id'], \
                labels_present_arr, aug_chunk_size, {"labels": label_arr}


    def __getitem__(self, index):
        return self.getitem(index)

    def __len__(self):
        return len(self.data)


