import numpy as np
import pickle
import glob
from collections import defaultdict
import os
import sys

splits = {1, 2, 3, 4, 5}
data_name = ["d1"] # {"f1", "f2", "f3", "f4", "f5"}
total = 26
data_per = 0.65
base_dir = sys.argv[1]
ground_truth_dir = base_dir + 'groundTruth/' #sys.argv[1] # "/mnt/ssd/all_users/dipika/ms_tcn/data/50salads/groundTruth/"

for split in splits:
    traindataset = base_dir + '/splits/train.split{}.bundle'.format(split)
    all_train_dataset = open(traindataset).read().split("\n")[0:-1]

    for dn in data_name:
        activity_with_vid_dict = defaultdict(list)
        for filename in all_train_dataset:
            video_id = filename.split(".txt")[0]
            main_act = video_id.split("-")[-1]
            
            activity_with_vid_dict[main_act].append(filename)

        uniq_labels = [] 
        selected_vids = [] 
        total_data = 0
        count = 0

        while True:
            prev_selected = None
            for i, activity in enumerate(activity_with_vid_dict.keys()):
                if i == 0:
                    amt_data = np.random.choice([2,3])
                    prev_selected = amt_data
                else:
                    amt_data = total - prev_selected
                vids = np.random.choice(activity_with_vid_dict[activity], size=amt_data)
                total_data += amt_data 
                selected_vids.extend(vids)
                temp_labels = []
                for vid in vids:
                    labels = open(os.path.join(ground_truth_dir, vid)).read().split("\n")[0:-1]
                    temp_labels.extend(np.unique(labels))
                uniq_labels.extend(np.unique(temp_labels))
                uniq_labels = np.unique(uniq_labels).tolist()

            if len(uniq_labels) == 19:
                print(f"Completed selecting for {dn} and data per {data_per}, Number of videos selected = {total_data}")
                break
            else:
                if count % 10 == 0:
                    print(f"Completed tryng {count} times with unique labels = {len(uniq_labels)}")
                if count > 2000:
                    all_train_dataset = open(traindataset).read().split("\n")[0:-1]
                    activity_with_vid_dict = defaultdict(list)
                    for filename in all_train_dataset:
                        video_id = filename.split(".txt")[0]
                        main_act = video_id.split("_")[-1]
                        
                        activity_with_vid_dict[main_act].append(filename)
                total_data = 0
                selected_vids = []
                uniq_labels = [] 
                count = count + 1
                continue

        # semi_supervised_train_dataset = base_dir + "/error_bars/train.split{}_errn{}_amt{}.bundle".format(split, dn, data_per)
        semi_supervised_train_dataset = base_dir + "/semi_supervised/train.split{}_amt{}.bundle".format(split, data_per)
        with open(semi_supervised_train_dataset, "w") as wfp:
            wfp.write("\n".join(selected_vids))
            wfp.write("\n")
        print(f"Created {semi_supervised_train_dataset}")

        all_train_dataset = list(set(all_train_dataset) - set(selected_vids))
