import torch
import torch.nn as nn
import numpy as np
from sklearn.cluster import KMeans
import torch.nn.functional as F

import os

# Device argsuration

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_t = 1
window_size =8
mc_sample_size = 10


loss_fn = torch.nn.BCEWithLogitsLoss()
# loss_fn = torch.nn.BCELoss()


# generate neighbourhoods centered at frames
def generatesegments(fea, label, select_idx): #(1536, 158)
    label = label.cpu().numpy()
    select_idx = select_idx.cpu().numpy()
    pos_label_idx = np.where(select_idx == 1)[0]
    t_list = np.random.choice(pos_label_idx, 20)   # breakfast=20, other=10
    remove_idx = np.where((t_list < (window_size // 2)) | (t_list > (fea.shape[1] - window_size // 2)) )[0]

    t_list = np.delete(t_list, remove_idx )
    t_list = np.random.choice(t_list, num_t)

    x_t_all = []
    x_close_all = []
    x_distant_all = []

    for t in t_list:
        x_t = fea[:, t - window_size // 2:t + window_size // 2]  # torch.Size([561, 4])
        x_t_list = []
        for i in range(mc_sample_size):
            x_t_list.append(x_t)
        x_t = torch.stack(x_t_list)

        x_t_all.append(x_t)
        x_close = find_segments_samelabel(fea, t, label, pos_label_idx)  # torch.Size([20, 561, 4])
        x_close_all.append(x_close)
        x_distant = find_segments_differentlabel(fea, t, label, pos_label_idx)  # torch.Size([10, 1536, 8])
        x_distant_all.append(x_distant)

    x_t_all = torch.stack(x_t_all)
    x_close_all = torch.stack(x_close_all)
    x_distant_all = torch.stack(x_distant_all)
    return x_t_all, x_close_all, x_distant_all


# find segments with same label (centre frame)
def find_segments_samelabel(x, t, label, pos_label_idx):
    # print("x before choose------", x.shape)
    x = x[:, pos_label_idx]
    # print("x after choose------", x.shape)
    T = x.shape[-1]
    # print("T------", T)
    select_neig_idx = np.where(label[pos_label_idx] == label[t])[0]
    t_p =  np.random.choice(select_neig_idx, mc_sample_size)
    t_p = [max(window_size // 2 + 1, min(t_pp, T - window_size // 2)) for t_pp in t_p]
    x_p = torch.stack([x[:, t_ind - window_size // 2:t_ind + window_size // 2] for t_ind in t_p])
    return x_p


# find segments with different labels (ccentre frame)
def find_segments_differentlabel(x, t, label, pos_label_idx): #(1536, 158)

    x = x[:, pos_label_idx]
    T = x.shape[-1]
    l = label[t]
    select_non_neig_idx = np.where(label[pos_label_idx] != l)[0]
    if len(select_non_neig_idx)==0:
        print("**********select_non_neig_idx=0********")
        return torch.zeros((mc_sample_size, 1536, window_size)).to(device)
    else:
        t_n = np.random.choice(select_non_neig_idx, mc_sample_size)
        t_n = [max(window_size // 2 + 1, min(t_nn, T - window_size // 2)) for t_nn in t_n]
        x_n = torch.stack([x[:, t_ind - window_size // 2:t_ind + window_size // 2] for t_ind in t_n])
        return x_n


class CriterionClass(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100) # remove label=-100
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCELoss()
        self.args = args
        self.mse = nn.MSELoss(reduction='none')


    def get_mse_loss(self, count, outp1):
        src_mask = torch.arange(outp1.shape[2], device=outp1.device)[None, :] < count[:, None]
        src_mask = src_mask.to(torch.float32).to(outp1.device).unsqueeze(1)
        mse_loss = 0.15 * torch.mean(torch.clamp(self.mse(outp1[:, :, 1:],outp1.detach()[:, :, :-1]), min=0, max=16) * src_mask[:, :, 1:])
        return {'mse_loss': mse_loss}



    def get_unsupervised_losses_SMC_nca(self, count, outp1, activity_labels, input_f, anchor_f, label, select_index, model_d, onlyunsupervised):
        print( "---------------get_unsupervised_losses_SMC_nca------------------")
        labels_ori = []
        select_info = []

        vid_ids = []
        f1 = []
        a1 = []
        t1 = []
        i3d_f = []

        f1_discri = []
        f1_t_list = []
        f1_t_close_list = []
        f1_t_notclose_list = []

        feature_activity = []
        maxpool_features = []
        maxpool_anchor = []

        bsize = outp1.shape[0]  # bs
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
            offset = torch.randint(low=vlow,  # torch.Size([79])
                                   high=vhigh,
                                   size=(len(idx),)).type(torch.long).to(device)
            previdx = torch.clamp(idx - offset, 0, vidlen - 1)

            labels_ori.append(label[j][idx])
            labels_ori.append(label[j][previdx])

            select_info.append(select_index[j][idx])
            select_info.append(select_index[j][previdx])

            # Now adding all frames togather
            f1.append(outp1[j].permute(1, 0)[idx, :])
            f1.append(outp1[j].permute(1, 0)[previdx, :])

            a1.append(anchor_f[j][idx, :])
            a1.append(anchor_f[j][previdx, :])


            #sample action segmets
            f1_discri.append(outp1[j].permute(1, 0)[0:vidlen, :]) #[len, 1536]
            f1_t, f1_t_close, f1_t_notclose = generatesegments(f1_discri[j].permute(1, 0), label[j][0:vidlen], select_index[j][0:vidlen])  # [1536, 4],[20, 1536, 4]
            f1_t_list.append(f1_t)
            f1_t_close_list.append(f1_t_close)
            f1_t_notclose_list.append(f1_t_notclose)


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

        select_info = torch.cat(select_info, dim=0)
        select_info = select_info.cpu().numpy()

        select_valid = torch.tensor((select_info[:, None] == select_info[None, :]) & ((select_info[:, None] == [[1] * len(select_info)]))).type(torch.float32).to(outp1.device)
        label_info = labels_ori

        # nca  loss
        f1_t_batch = torch.stack(f1_t_list)  # [5, 1536, 4] [5, num_t, 1536, 4]
        print("f1_t_batch0----",f1_t_batch.shape)
        f1_t_close_batch = torch.stack(f1_t_close_list)  # [5,num_t,20,1536,4]
        print("f1_t_close_batch----",f1_t_close_batch.shape)

        f1_t_notclose_batch = torch.stack(f1_t_notclose_list)  # [5,num_t,20,1536,4]

        batch_size, num_t, mc_sample,  f_dim, t_len = f1_t_batch.shape  # 3, 4, 1536, 8
        f1_t_close_batch = f1_t_close_batch.reshape((-1, f_dim, t_len))  # (20*5,1536,4)
        f1_t_notclose_batch = f1_t_notclose_batch.reshape((-1, f_dim, t_len))
        f1_t_batch = f1_t_batch.reshape((-1, f_dim, t_len))
        same_label = torch.ones((len(f1_t_close_batch) )).to(device)
        diff_label = torch.zeros((len(f1_t_notclose_batch))).to(device)

        # f1_t_batch = f1_t_batch.meann(-1)  # [30, 1536]
        # f1_t_close_batch = f1_t_close_batch.mean(-1)  # [30, 1536]
        # f1_t_notclose_batch = f1_t_notclose_batch.mean(-1)  # [30, 1536]

        f1_t_batch = torch.max(f1_t_batch, -1)[0]  # [30, 1536]
        print("f1_t_batch----",f1_t_batch.shape)
        f1_t_close_batch = torch.max(f1_t_close_batch, -1)[0] # [30, 1536]
        f1_t_notclose_batch = torch.max(f1_t_notclose_batch, -1)[0] # [30, 1536]

        d_p = model_d(f1_t_batch, f1_t_close_batch)
        d_n = model_d(f1_t_batch, f1_t_notclose_batch)

        p_loss = loss_fn(d_p, same_label)
        n_loss = loss_fn(d_n, diff_label)
        loss_nca = (p_loss + n_loss) / 2
        print("loss_d---------", loss_nca.item())

        if feature_activity is not None:
            feature_activity = np.array(feature_activity)

        if feature_activity is None:  # For 50salads and GTEA where there is no high level activity defined
                pos_weight_mat = torch.tensor((vid_ids[:, None] == vid_ids[None, :]) & \
                                              (np.abs(t1[:, None] - t1[None, :]) <= self.args.delta) & \
                                              (label_info[:, None] == label_info[None, :]))  # torch.Size([316, 316])
                negative_samples_minus = torch.tensor((vid_ids[:, None] == vid_ids[None, :]) & \
                                                      (np.abs(t1[:, None] - t1[None, :]) > self.args.delta) & \
                                                      (label_info[:, None] == label_info[None, :])).type(torch.float32).to(device)
                pos_weight_mat = pos_weight_mat | torch.tensor((vid_ids[:, None] != vid_ids[None, :]) & \
                                                               (label_info[:, None] == label_info[None,:]))
        else:  # For datasets like Breakfast where high level activity is known
                pos_weight_mat = torch.tensor((feature_activity[:, None] == feature_activity[None, :]) & \
                                              (np.abs(t1[:, None] - t1[None, :]) <= self.args.delta) & \
                                              (label_info[:, None] == label_info[None, :]))
                negative_samples_minus = torch.tensor((feature_activity[:, None] == feature_activity[None, :]) & \
                                                      (np.abs(t1[:, None] - t1[None, :]) > self.args.delta) & \
                                                      (label_info[:, None] == label_info[None, :])).type(torch.float32).to(device)


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

        # if (activity_labels is not None):
        #     loss = loss_pos + loss_neg1 + loss_neg3 + loss_neg4 +  loss_nca
        # else:
        #     loss = loss_pos + loss_neg1 + loss_neg3 + loss_neg4 + loss_nca

        loss = loss_pos + loss_neg1 + loss_neg3 + loss_neg4 +  loss_nca
        print("loss_pos/ loss_neg1/ loss -----", loss_pos.item(), loss_neg1.item(), loss.item())


        # original video-level
        if activity_labels is not None:
            print("activity_labels---", activity_labels)  # ['sandwich', 'salat', 'friedegg', 'juice', 'coffee']
            maxpool_features = torch.stack(maxpool_features)  # [5,1536]
            maxpool_anchor = torch.stack(maxpool_anchor)  # [5,1536]

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


    def forward(self, count, projection, prediction, labels, activity_labels, input_f, unsupervised, onlyunsupervised, achor_f, select_index, nl_mask, model_d):
        '''
         loss_dict = criterion(count, projection_out, pred_out, labels, activity_labels, item[0], unsupervised, achor_out, select_index ) #cross en loss
        '''
        if unsupervised is False:
            ce_loss = self.ce(prediction, labels)

            loss = ce_loss
            print("ce_loss----", loss.item())
            loss_dict = {'ce_loss': ce_loss}
            if self.args.mse:  # False
                mse_loss_dict = self.get_mse_loss(count, prediction)
                print("mse_loss----",  mse_loss_dict['mse_loss'].item())
                loss = loss + mse_loss_dict['mse_loss']
                loss_dict.update(mse_loss_dict)
        else:
            loss = 0
            loss_dict = {}

        if unsupervised is True and onlyunsupervised is False:
            v, c = np.unique(labels.cpu().numpy(), return_counts=True)
            print("before pesu_labels value, count----", v[0], c[0] ) #-100
            for i in range(labels.shape[0]):
                idx = np.where(select_index.cpu().numpy()[i]==0)[0]
                labels[i][idx] = -100
            v, c = np.unique(labels.cpu().numpy(), return_counts=True)
            print("after pesu_labels value, count----", v[0], c[0] )
            ce_loss = self.ce(prediction, labels)
            print("ce_loss----", ce_loss.item())

            #NL learning
            select_index_nl = select_index.view(-1)
            print("negtive select_index==0----", torch.sum(select_index_nl==0))
            print("positive select_index==1----", torch.sum(select_index_nl==1))
            nl_mask = nl_mask.view(nl_mask.shape[0]* nl_mask.shape[1], nl_mask.shape[2]) #[5*960, 19]
            prediction = prediction.permute(0,2,1).contiguous().view(-1, prediction.shape[1] )
            num_classes = nl_mask.shape[1]
            nl_idx = (nl_mask.sum(dim=1) != num_classes) * (nl_mask.sum(dim=1) > 0) * (select_index_nl==0)  #True/False delete positive sample

            loss_nl = 0
            if sum(nl_idx * 1) > 0:
                nl_prediction = prediction[nl_idx]
                pred_nl = F.softmax(nl_prediction, dim=1)
                pred_nl = 1 - pred_nl
                pred_nl = torch.clamp(pred_nl, 1e-7, 1.0)
                nl_mask = nl_mask[nl_idx]
                y_nl = torch.ones((nl_prediction.shape)).to(device, dtype=prediction.dtype)
                loss_nl = torch.mean((-torch.sum((y_nl * torch.log(pred_nl)) * nl_mask, dim=-1)) / (torch.sum(nl_mask, dim=-1) + 1e-7))
                print("loss_nl----", loss_nl.item())

            loss = ce_loss + loss_nl
            print("PL_NL_loss----", loss.item())
            loss_dict = {'ce_loss': loss}

        unsupervised_loss_dict = self.get_unsupervised_losses_SMC_nca(count, projection, activity_labels, input_f, achor_f, labels, select_index, model_d, onlyunsupervised )

        loss_dict.update(unsupervised_loss_dict)
        loss = loss + unsupervised_loss_dict['contrastive_loss']
        # loss = loss + 0    #supervised
        loss_dict.update({'full_loss': loss})
        return loss_dict
        
