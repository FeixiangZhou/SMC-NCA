import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as F
import os

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CriterionClass(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss()
        self.args = args


    def get_unsupervised_losses_SMC_ablation(self, count, outp1, activity_labels, input_f, anchor_f, label):
        print( "---------------get_unsupervised_losses_SMC_ablation------------------")
        labels_ori = []

        vid_ids = []
        f1 = []
        a1 = []
        t1 = []
        i3d_f = []

        feature_activity = []
        bsize = outp1.shape[0]
        for j in range(bsize):

            # Sampling of first K frames
            vidlen = count[j]
            sel_frames_current = torch.linspace(0, vidlen, self.args.num_samples_frames)
            sel_frames_current = [int(i) for i in sel_frames_current]

            idx = []
            for kkl in range(len(sel_frames_current) - 1):
                cur_start = sel_frames_current[kkl]
                cur_end = sel_frames_current[kkl + 1]
                list_frames = list(range(cur_start, cur_end + 1))
                idx.append(np.random.choice(list_frames, 1)[0])

            idx = torch.tensor(idx).type(torch.long).to(device)  # torch.Size([79])
            idx = torch.clamp(idx, 0, vidlen - 1)  # [79]
            # print("idx-----", idx)

            # Sampling of second set of frames from surroundings epsilon
            vlow = 1  # To prevent value 0 in variable low
            vhigh = int(np.ceil(self.args.epsilon * vidlen.item()))  # 0.05*

            if vhigh <= vlow:
                vhigh = vlow + 2
            offset = torch.randint(low=vlow,
                                   high=vhigh,
                                   size=(len(idx),)).type(torch.long).to(device)
            previdx = torch.clamp(idx - offset, 0, vidlen - 1)
            labels_ori.append(label[j][idx])
            labels_ori.append(label[j][previdx])


            # Now adding all frames togather
            f1.append(outp1[j].permute(1, 0)[idx, :])
            f1.append(outp1[j].permute(1, 0)[previdx, :])  # f1[0] torch.Size([79 frames, 1536])

            a1.append(anchor_f[j][idx, :])
            a1.append(anchor_f[j][previdx, :])

            if activity_labels is not None:
                feature_activity.extend([activity_labels[j]] * len(idx) * 2)
            else:
                feature_activity = None

            i3d_f.append(input_f[j][idx, :])
            i3d_f.append(input_f[j][previdx, :])

            vid_ids.extend([j] * len(idx))
            vid_ids.extend([j] * len(previdx))

            idx = idx / vidlen.to(dtype=torch.float32, device=vidlen.device)
            previdx = previdx / vidlen.to(dtype=torch.float32, device=vidlen.device)

            t1.extend(idx.detach().cpu().numpy().tolist())
            t1.extend(previdx.detach().cpu().numpy().tolist())

        vid_ids = torch.tensor(vid_ids).numpy()
        t1 = np.array(t1)

        labels_ori = torch.cat(labels_ori, dim=0)
        labels_ori = labels_ori.cpu().numpy()


        f1 = torch.cat(f1, dim=0)  # torch.Size([316, 1536]) [79*bs*2, 1536]
        print("f1-------", f1.shape)

        achor_out = torch.cat(a1, dim=0)
        print("achor_out-------", achor_out.shape)

        i3d_f = torch.cat(i3d_f, dim=0)  # torch.Size([316, 2048])
        f1_cpu = f1.cpu().detach()
        achor_out_cpu = achor_out.cpu().detach()


        # Getting label_info from inou, semnatic and temporal information
        clust = KMeans(n_clusters=self.args.clustCount)
        # label_info_f = clust.fit_predict(f1_cpu.numpy())
        label_info = clust.fit_predict(i3d_f.numpy())
        # label_info_anc = clust.fit_predict(achor_out_cpu.numpy())



        if feature_activity is not None:
            feature_activity = np.array(feature_activity)

        if feature_activity is None:  # For 50salads and GTEA where there is no high level activity defined
            if self.args.no_time:  # false
                pos_weight_mat = torch.tensor((label_info[:, None] == label_info[None, :]))
                negative_samples_minus = 0
            else:
                pos_weight_mat = torch.tensor((vid_ids[:, None] == vid_ids[None, :]) & \
                                              (np.abs(t1[:, None] - t1[None, :]) <= self.args.delta) & \
                                              (label_info[:, None] == label_info[None, :]))  # torch.Size([316, 316])
                negative_samples_minus = torch.tensor((vid_ids[:, None] == vid_ids[None, :]) & \
                                                      (np.abs(t1[:, None] - t1[None, :]) > self.args.delta) & \
                                                      (label_info[:, None] == label_info[None, :])).type(torch.float32).to(device)
                pos_weight_mat = pos_weight_mat | torch.tensor((vid_ids[:, None] != vid_ids[None, :]) & \
                                                               (label_info[:, None] == label_info[None,:]))  # 不同的视频帧， 但是聚类label相同
        else:  # For datasets like Breakfast where high level activity is known
            if self.args.no_time:
                pos_weight_mat = torch.tensor((feature_activity[:, None] == feature_activity[None, :]) & \
                                              (label_info[:, None] == label_info[None, :]))
                negative_samples_minus = torch.tensor((feature_activity[:, None] != feature_activity[None, :]) & \
                                                      (label_info[:, None] == label_info[None, :])).type(torch.float32).to(device)
            else:
                pos_weight_mat = torch.tensor((feature_activity[:, None] == feature_activity[None, :]) & \
                                              (np.abs(t1[:, None] - t1[None, :]) <= self.args.delta) & \
                                              (label_info[:, None] == label_info[None, :]))
                negative_samples_minus = torch.tensor((feature_activity[:, None] == feature_activity[None, :]) & \
                                                      (np.abs(t1[:, None] - t1[None, :]) > self.args.delta) & \
                                                      (label_info[:, None] == label_info[None, :])).type(torch.float32).to(device)


        pos_weight_mat = (pos_weight_mat).type(torch.float32).to(device)
        countpos = torch.sum(pos_weight_mat)
        I = torch.eye(pos_weight_mat.shape[1]).to(device)
        pos_weight_mat_an = pos_weight_mat - I
        not_same_activity_an = 1 - pos_weight_mat_an - I - negative_samples_minus
        scale_factor = 1



        #inter-information postive pairs--LP_ap
        anc_pos = torch.mm(achor_out, f1.T)
        anc_pos = anc_pos / scale_factor

       # M_in matrix
        anc_pos = pos_weight_mat * anc_pos
        anc_pos = torch.nn.functional.logsigmoid(anc_pos) * pos_weight_mat
        loss_pos = -torch.sum(anc_pos) / countpos


        #intra-information negative pairs--LN_aa
        countneg_an1 = torch.sum(not_same_activity_an)
        print("countneg_an1-----", countneg_an1)
        anc_neg1 = torch.mm(achor_out, achor_out.T)
        anc_neg1 = anc_neg1 / scale_factor
        anc_neg1 = not_same_activity_an * anc_neg1
        anc_neg1 = torch.nn.functional.logsigmoid(-anc_neg1) * not_same_activity_an
        loss_neg1 = -torch.sum(anc_neg1) / countneg_an1
        print("loss_neg1-----", loss_neg1.item())


        #inter-information negative pairs--LN_ap
        anc_neg3 = torch.mm(achor_out, f1.T)
        anc_neg3 = not_same_activity_an * anc_neg3
        anc_neg3 = torch.nn.functional.logsigmoid(-anc_neg3) * not_same_activity_an
        loss_neg3 = -torch.sum(anc_neg3) / countneg_an1
        print("loss_neg3------", loss_neg3.item())

        # intra-information negative pairs--LN_pp
        anc_neg4 = torch.mm(f1, f1.T)
        anc_neg4 = not_same_activity_an * anc_neg4
        anc_neg4 = torch.nn.functional.logsigmoid(-anc_neg4) * not_same_activity_an
        loss_neg4 = -torch.sum(anc_neg4) / countneg_an1
        print("loss_neg4------", loss_neg4.item())

        loss =  loss_pos + loss_neg1
        print("loss_pos/ loss_neg1/ loss -----", loss_pos.item(), loss_neg1.item(), loss.item())
        usupervised_dict_loss = {'contrastive_loss': loss}
        return usupervised_dict_loss

    def get_unsupervised_losses_SMC_breakfast(self, count, outp1, activity_labels, input_f, anchor_f, label):
        print("---------------get_unsupervised_losses_SMC_breakfast------------------")
        labels_ori = []

        vid_ids = []
        f1 = []
        a1 = []
        t1 = []
        i3d_f = []
        maxpool_features = []
        maxpool_anchor = []

        feature_activity = []
        bsize = outp1.shape[0]
        for j in range(bsize):

            # Sampling of first K frames
            vidlen = count[j]
            sel_frames_current = torch.linspace(0, vidlen, self.args.num_samples_frames)
            sel_frames_current = [int(i) for i in sel_frames_current]

            idx = []
            for kkl in range(len(sel_frames_current) - 1):
                cur_start = sel_frames_current[kkl]
                cur_end = sel_frames_current[kkl + 1]
                list_frames = list(range(cur_start, cur_end + 1))
                idx.append(np.random.choice(list_frames, 1)[0])

            idx = torch.tensor(idx).type(torch.long).to(device)  # torch.Size([79])
            idx = torch.clamp(idx, 0, vidlen - 1)  # [79]
            # print("idx-----", idx)

            # Sampling of second set of frames from surroundings epsilon
            vlow = 1  # To prevent value 0 in variable low
            vhigh = int(np.ceil(self.args.epsilon * vidlen.item()))  # 0.05*

            if vhigh <= vlow:
                vhigh = vlow + 2
            offset = torch.randint(low=vlow,
                                   high=vhigh,
                                   size=(len(idx),)).type(torch.long).to(device)
            previdx = torch.clamp(idx - offset, 0, vidlen - 1)
            labels_ori.append(label[j][idx])
            labels_ori.append(label[j][previdx])

            # Now adding all frames togather
            f1.append(outp1[j].permute(1, 0)[idx, :])
            f1.append(outp1[j].permute(1, 0)[previdx, :])  # f1[0] torch.Size([79 frames, 1536])

            a1.append(anchor_f[j][idx, :])
            a1.append(anchor_f[j][previdx, :])

            if activity_labels is not None:
                feature_activity.extend([activity_labels[j]] * len(idx) * 2)
            else:
                feature_activity = None

            i3d_f.append(input_f[j][idx, :])
            i3d_f.append(input_f[j][previdx, :])

            vid_ids.extend([j] * len(idx))
            vid_ids.extend([j] * len(previdx))

            idx = idx / vidlen.to(dtype=torch.float32, device=vidlen.device)
            previdx = previdx / vidlen.to(dtype=torch.float32, device=vidlen.device)

            t1.extend(idx.detach().cpu().numpy().tolist())
            t1.extend(previdx.detach().cpu().numpy().tolist())

            # video_level
            maxpool_features.append(torch.max(outp1[j, :, :vidlen], dim=-1)[0])
            maxpool_anchor.append(torch.max(anchor_f[j, :vidlen, :], dim=0)[0])

        vid_ids = torch.tensor(vid_ids).numpy()
        t1 = np.array(t1)

        labels_ori = torch.cat(labels_ori, dim=0)
        labels_ori = labels_ori.cpu().numpy()

        f1 = torch.cat(f1, dim=0)  # torch.Size([316, 1536]) [79*bs*2, 1536]
        print("f1-------", f1.shape)

        achor_out = torch.cat(a1, dim=0)
        print("achor_out-------", achor_out.shape)

        i3d_f = torch.cat(i3d_f, dim=0)  # torch.Size([316, 2048])
        f1_cpu = f1.cpu().detach()
        achor_out_cpu = achor_out.cpu().detach()

        # Getting label_info from inou, semnatic and temporal information
        clust = KMeans(n_clusters=self.args.clustCount)
        label_info_f = clust.fit_predict(f1_cpu.numpy())
        label_info = clust.fit_predict(i3d_f.numpy())
        label_info_anc = clust.fit_predict(achor_out_cpu.numpy())

        if feature_activity is not None:
            feature_activity = np.array(feature_activity)

        if feature_activity is None:  # For 50salads and GTEA where there is no high level activity defined
            if self.args.no_time:  # false
                pos_weight_mat = torch.tensor((label_info[:, None] == label_info[None, :]) & (
                            label_info_anc[:, None] == label_info_anc[None, :]) & \
                                              (label_info_f[:, None] == label_info_f[None, :]))
                negative_samples_minus = 0
            else:
                pos_weight_mat = torch.tensor((vid_ids[:, None] == vid_ids[None, :]) & \
                                              (np.abs(t1[:, None] - t1[None, :]) <= self.args.delta) & \
                                              (label_info[:, None] == label_info[None, :]) & (
                                                          label_info_anc[:, None] == label_info_anc[None, :]) & \
                                              (label_info_f[:, None] == label_info_f[None,:]))  # torch.Size([316, 316])
                negative_samples_minus = torch.tensor((vid_ids[:, None] == vid_ids[None, :]) & \
                                                      (np.abs(t1[:, None] - t1[None, :]) > self.args.delta) & \
                                                      (label_info[:, None] == label_info[None, :]) & (
                                                                  label_info_anc[:, None] == label_info_anc[None, :]) & \
                                                      (label_info_f[:, None] == label_info_f[None, :])).type( torch.float32).to(device)
                pos_weight_mat = pos_weight_mat | torch.tensor((vid_ids[:, None] != vid_ids[None, :]) & \
                                                               (label_info[:, None] == label_info[None, :]) & (label_info_anc[:, None] == label_info_anc[
                                                                                                      None, :]) & \
                                                               (label_info_f[:, None] == label_info_f[None,
                                                                                         :]))  # 不同的视频帧， 但是聚类label相同
        else:  # For datasets like Breakfast where high level activity is known
            if self.args.no_time:
                pos_weight_mat = torch.tensor((feature_activity[:, None] == feature_activity[None, :]) & \
                                              (label_info[:, None] == label_info[None, :]) & (label_info_anc[:, None] == label_info_anc[None, :]) & \
                                              (label_info_f[:, None] == label_info_f[None, :]))
                negative_samples_minus = torch.tensor((feature_activity[:, None] != feature_activity[None, :]) & \
                                                      (label_info[:, None] == label_info[None, :]) & (label_info_anc[:, None] == label_info_anc[None, :]) & \
                                                      (label_info_f[:, None] == label_info_f[None, :])).type(torch.float32).to(device)
            else:
                pos_weight_mat = torch.tensor((feature_activity[:, None] == feature_activity[None, :]) & \
                                              (np.abs(t1[:, None] - t1[None, :]) <= self.args.delta) & \
                                              (label_info[:, None] == label_info[None, :]) & (label_info_anc[:, None] == label_info_anc[None, :]) & \
                                              (label_info_f[:, None] == label_info_f[None, :]))
                negative_samples_minus = torch.tensor((feature_activity[:, None] == feature_activity[None, :]) & \
                                                      (np.abs(t1[:, None] - t1[None, :]) > self.args.delta) & \
                                                      (label_info[:, None] == label_info[None, :]) & ( label_info_anc[:, None] == label_info_anc[None, :]) & \
                                                      (label_info_f[:, None] == label_info_f[None, :])).type(torch.float32).to(device)

        pos_weight_mat = (pos_weight_mat).type(torch.float32).to(device)
        I = torch.eye(pos_weight_mat.shape[1]).to(device)
        pos_weight_mat_an = pos_weight_mat - I
        not_same_activity_an = 1 - pos_weight_mat_an - I - negative_samples_minus
        scale_factor = 1

        # inter-information postive pairs--LP_ap
        anc_pos = torch.mm(achor_out, f1.T)
        anc_pos = anc_pos / scale_factor
        anc_pos = I * anc_pos
        anc_pos = torch.nn.functional.logsigmoid(anc_pos) * I
        loss_pos = -torch.sum(anc_pos) / (anc_pos.shape[0])

        # intra-information negative pairs--LN_aa
        countneg_an1 = torch.sum(not_same_activity_an)
        print("countneg_an1-----", countneg_an1)
        anc_neg1 = torch.mm(achor_out, achor_out.T)
        anc_neg1 = anc_neg1 / scale_factor
        anc_neg1 = not_same_activity_an * anc_neg1
        anc_neg1 = torch.nn.functional.logsigmoid(-anc_neg1) * not_same_activity_an
        loss_neg1 = -torch.sum(anc_neg1) / countneg_an1
        print("loss_neg1-----", loss_neg1.item())

        # inter-information negative pairs--LN_ap
        anc_neg3 = torch.mm(achor_out, f1.T)
        anc_neg3 = not_same_activity_an * anc_neg3
        anc_neg3 = torch.nn.functional.logsigmoid(-anc_neg3) * not_same_activity_an
        loss_neg3 = -torch.sum(anc_neg3) / countneg_an1
        print("loss_neg3------", loss_neg3.item())

        # intra-information negative pairs--LN_pp
        anc_neg4 = torch.mm(f1, f1.T)
        anc_neg4 = not_same_activity_an * anc_neg4
        anc_neg4 = torch.nn.functional.logsigmoid(-anc_neg4) * not_same_activity_an
        loss_neg4 = -torch.sum(anc_neg4) / countneg_an1
        print("loss_neg4------", loss_neg4.item())

        loss = loss_pos + loss_neg1 + loss_neg3 + loss_neg4
        print("loss_pos/ loss_neg1/ loss -----", loss_pos.item(), loss_neg1.item(), loss.item())

        # original video-level
        if activity_labels is not None:
            print("activity_labels---", activity_labels)  # ['sandwich', 'salat', 'friedegg', 'juice', 'coffee']
            maxpool_features = torch.stack(maxpool_features)  # [5,1536]
            maxpool_anchor = torch.stack(maxpool_anchor)  # [5,1536]

            # ori---
            maxpool_features = maxpool_features / torch.norm(maxpool_features, dim=1, keepdim=True)
            maxpool_featsim = torch.exp(maxpool_features @ maxpool_features.T / 0.1)
            same_activity = torch.tensor(np.array(activity_labels)[:, None] == np.array(activity_labels)[None, :])
            I = torch.eye(len(same_activity)).to(device)
            same_activity = (same_activity).type(torch.float32).to(device) - I
            not_same_activity = 1 - same_activity - I
            countpos = torch.sum(same_activity)
            if countpos == 0:
                print("Video level contrast has no same pairs")
                video_level_contrast = 0
            else:
                maxpool_featsim_pos = same_activity * maxpool_featsim
                maxpool_featsim_negsum = torch.sum(not_same_activity * maxpool_featsim, dim=1)
                simprob = maxpool_featsim_pos / (maxpool_featsim_negsum + maxpool_featsim_pos) + not_same_activity
                video_level_contrast = torch.sum(-torch.log(simprob + I)) / countpos
                print("video_level_contrast -----", video_level_contrast.item())

            loss = loss + video_level_contrast
        usupervised_dict_loss = {'contrastive_loss': loss}
        return usupervised_dict_loss


    def get_unsupervised_losses_SMC(self, count, outp1, activity_labels, input_f, anchor_f, label):
        print( "---------------get_unsupervised_losses_SMC------------------")
        labels_ori = []

        vid_ids = []
        f1 = []
        a1 = []
        t1 = []
        i3d_f = []

        feature_activity = []
        bsize = outp1.shape[0]
        for j in range(bsize):

            # Sampling of first K frames
            vidlen = count[j]
            sel_frames_current = torch.linspace(0, vidlen, self.args.num_samples_frames)
            sel_frames_current = [int(i) for i in sel_frames_current]

            idx = []
            for kkl in range(len(sel_frames_current) - 1):
                cur_start = sel_frames_current[kkl]
                cur_end = sel_frames_current[kkl + 1]
                list_frames = list(range(cur_start, cur_end + 1))
                idx.append(np.random.choice(list_frames, 1)[0])

            idx = torch.tensor(idx).type(torch.long).to(device)  # torch.Size([79])
            idx = torch.clamp(idx, 0, vidlen - 1)  # [79]

            # Sampling of second set of frames from surroundings epsilon
            vlow = 1  # To prevent value 0 in variable low
            vhigh = int(np.ceil(self.args.epsilon * vidlen.item()))  # 0.05*

            if vhigh <= vlow:
                vhigh = vlow + 2
            offset = torch.randint(low=vlow,
                                   high=vhigh,
                                   size=(len(idx),)).type(torch.long).to(device)
            previdx = torch.clamp(idx - offset, 0, vidlen - 1)
            labels_ori.append(label[j][idx])
            labels_ori.append(label[j][previdx])


            # Now adding all frames togather
            f1.append(outp1[j].permute(1, 0)[idx, :])
            f1.append(outp1[j].permute(1, 0)[previdx, :])  # f1[0] torch.Size([79 frames, 1536])

            a1.append(anchor_f[j][idx, :])
            a1.append(anchor_f[j][previdx, :])

            if activity_labels is not None:
                feature_activity.extend([activity_labels[j]] * len(idx) * 2)
            else:
                feature_activity = None

            i3d_f.append(input_f[j][idx, :])
            i3d_f.append(input_f[j][previdx, :])

            vid_ids.extend([j] * len(idx))
            vid_ids.extend([j] * len(previdx))

            idx = idx / vidlen.to(dtype=torch.float32, device=vidlen.device)
            previdx = previdx / vidlen.to(dtype=torch.float32, device=vidlen.device)

            t1.extend(idx.detach().cpu().numpy().tolist())
            t1.extend(previdx.detach().cpu().numpy().tolist())

        vid_ids = torch.tensor(vid_ids).numpy()
        t1 = np.array(t1)

        labels_ori = torch.cat(labels_ori, dim=0)
        labels_ori = labels_ori.cpu().numpy()


        f1 = torch.cat(f1, dim=0)  # torch.Size([316, 1536]) [79*bs*2, 1536]
        print("f1-------", f1.shape)

        achor_out = torch.cat(a1, dim=0)
        print("achor_out-------", achor_out.shape)

        i3d_f = torch.cat(i3d_f, dim=0)  # torch.Size([316, 2048])
        f1_cpu = f1.cpu().detach()
        achor_out_cpu = achor_out.cpu().detach()


        # Getting label_info from inou, semnatic and temporal information
        clust = KMeans(n_clusters=self.args.clustCount)
        label_info_f = clust.fit_predict(f1_cpu.numpy())
        label_info = clust.fit_predict(i3d_f.numpy())
        label_info_anc = clust.fit_predict(achor_out_cpu.numpy())



        if feature_activity is not None:
            feature_activity = np.array(feature_activity)

        if feature_activity is None:  # For 50salads and GTEA where there is no high level activity defined
            if self.args.no_time:  # false
                pos_weight_mat = torch.tensor((label_info[:, None] == label_info[None, :]) & (label_info_anc[:, None] == label_info_anc[None, :]) & \
                                              (label_info_f[:, None] == label_info_f[None, :]))
                negative_samples_minus = 0
            else:
                pos_weight_mat = torch.tensor((vid_ids[:, None] == vid_ids[None, :]) & \
                                              (np.abs(t1[:, None] - t1[None, :]) <= self.args.delta) & \
                                              (label_info[:, None] == label_info[None, :]) & (label_info_anc[:, None] == label_info_anc[None, :]) & \
                                              (label_info_f[:, None] == label_info_f[None, :]))  # torch.Size([316, 316])
                negative_samples_minus = torch.tensor((vid_ids[:, None] == vid_ids[None, :]) & \
                                                      (np.abs(t1[:, None] - t1[None, :]) > self.args.delta) & \
                                                      (label_info[:, None] == label_info[None, :]) & (label_info_anc[:, None] == label_info_anc[None, :]) & \
                                                      (label_info_f[:, None] == label_info_f[None, :])).type(torch.float32).to(device)
                pos_weight_mat = pos_weight_mat | torch.tensor((vid_ids[:, None] != vid_ids[None, :]) & \
                                                               (label_info[:, None] == label_info[None,:]) & (label_info_anc[:, None] == label_info_anc[None, :]) & \
                                                               (label_info_f[:, None] == label_info_f[None, :]))  # 不同的视频帧， 但是聚类label相同
        else:  # For datasets like Breakfast where high level activity is known
            if self.args.no_time:
                pos_weight_mat = torch.tensor((feature_activity[:, None] == feature_activity[None, :]) & \
                                              (label_info[:, None] == label_info[None, :]) & (label_info_anc[:, None] == label_info_anc[None, :]) & \
                                              (label_info_f[:, None] == label_info_f[None, :]))
                negative_samples_minus = torch.tensor((feature_activity[:, None] != feature_activity[None, :]) & \
                                                      (label_info[:, None] == label_info[None, :]) & (label_info_anc[:, None] == label_info_anc[None, :]) & \
                                                      (label_info_f[:, None] == label_info_f[None, :])).type(torch.float32).to(device)
            else:
                pos_weight_mat = torch.tensor((feature_activity[:, None] == feature_activity[None, :]) & \
                                              (np.abs(t1[:, None] - t1[None, :]) <= self.args.delta) & \
                                              (label_info[:, None] == label_info[None, :]) & (label_info_anc[:, None] == label_info_anc[None, :]) & \
                                              (label_info_f[:, None] == label_info_f[None, :]))
                negative_samples_minus = torch.tensor((feature_activity[:, None] == feature_activity[None, :]) & \
                                                      (np.abs(t1[:, None] - t1[None, :]) > self.args.delta) & \
                                                      (label_info[:, None] == label_info[None, :]) & (label_info_anc[:, None] == label_info_anc[None, :]) & \
                                                      (label_info_f[:, None] == label_info_f[None, :])).type(torch.float32).to(device)


        pos_weight_mat = (pos_weight_mat).type(torch.float32).to(device)
        I = torch.eye(pos_weight_mat.shape[1]).to(device)
        pos_weight_mat_an = pos_weight_mat - I
        not_same_activity_an = 1 - pos_weight_mat_an - I - negative_samples_minus
        scale_factor =1



        #inter-information postive pairs--LP_ap
        anc_pos = torch.mm(achor_out, f1.T)
        anc_pos = anc_pos / scale_factor
        anc_pos = I * anc_pos
        anc_pos = torch.nn.functional.logsigmoid(anc_pos) * I
        loss_pos = -torch.sum(anc_pos) / (anc_pos.shape[0])



        #intra-information negative pairs--LN_aa
        countneg_an1 = torch.sum(not_same_activity_an)
        print("countneg_an1-----", countneg_an1)
        anc_neg1 = torch.mm(achor_out, achor_out.T)
        anc_neg1 = anc_neg1 / scale_factor
        anc_neg1 = not_same_activity_an * anc_neg1
        anc_neg1 = torch.nn.functional.logsigmoid(-anc_neg1) * not_same_activity_an
        loss_neg1 = -torch.sum(anc_neg1) / countneg_an1
        print("loss_neg1-----", loss_neg1.item())


        #inter-information negative pairs--LN_ap
        anc_neg3 = torch.mm(achor_out, f1.T)
        anc_neg3 = not_same_activity_an * anc_neg3
        anc_neg3 = torch.nn.functional.logsigmoid(-anc_neg3) * not_same_activity_an
        loss_neg3 = -torch.sum(anc_neg3) / countneg_an1
        print("loss_neg3------", loss_neg3.item())

        # intra-information negative pairs--LN_pp
        anc_neg4 = torch.mm(f1, f1.T)
        anc_neg4 = not_same_activity_an * anc_neg4
        anc_neg4 = torch.nn.functional.logsigmoid(-anc_neg4) * not_same_activity_an
        loss_neg4 = -torch.sum(anc_neg4) / countneg_an1
        print("loss_neg4------", loss_neg4.item())

        loss =  loss_pos + loss_neg1 + loss_neg3 + loss_neg4
        print("loss_pos/ loss_neg1/ loss -----", loss_pos.item(), loss_neg1.item(), loss.item())
        usupervised_dict_loss = {'contrastive_loss': loss}
        return usupervised_dict_loss


    def forward(self, count, projection, prediction, labels, activity_labels, input_f, achor_f):
        '''
         loss_dict = criterion(count, projection_out, pred_out, labels, activity_labels, item[0])
        :param count:
        :param projection:
        :param prediction:
        :param labels:
        :param activity_labels:
        :param input_f:
        :return:
        '''
        unsupervised_loss_dict = self.get_unsupervised_losses_SMC(count, projection, activity_labels, input_f, achor_f, labels)
        loss = unsupervised_loss_dict['contrastive_loss']
        loss_dict = {'full_loss': loss}
        loss_dict.update(unsupervised_loss_dict)
        return loss_dict
        
