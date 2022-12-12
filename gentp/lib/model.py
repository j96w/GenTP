import math
import torch
import copy
import random
import numpy as np
from typing import Optional, Any
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn import ModuleList
from sklearn.decomposition import PCA
import torchvision.models as models
import matplotlib.pyplot as plt
from torch.autograd import Variable
import time

color_list = []
import seaborn as sns
alllist = sns.color_palette("hls", 15)
random.shuffle(alllist)
for item in alllist:
    new = list(item)
    color_list.append([new[0]*255.0, new[1]*255.0, new[2]*255.0])

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class ConvDecoder(nn.Module):
    def __init__(self, input_size, output_size, output_dim, output_dim_affordance):
        super(ConvDecoder, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.output_dim = output_dim
        self.output_dim_affordance = output_dim_affordance

        self.fc_1 = nn.Linear(256, 512)
        self.fc_2 = nn.Linear(512, 1024)

        self.t_conv1 = nn.ConvTranspose2d(4, 64, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(64, self.output_dim, 2, stride=2)

        self.affordance_fc_1 = nn.Linear(512, 512)
        self.affordance_fc_2 = nn.Linear(512, 1024)

        self.affordance_t_conv1 = nn.ConvTranspose2d(4, 128, 2, stride=2)
        self.affordance_t_conv2 = nn.ConvTranspose2d(128, self.output_dim_affordance, 2, stride=2)


    def forward(self, x):

        bs = x.size()[0]
        num_obj = x.size()[1]

        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)

        x = x.view(bs, num_obj, -1, self.input_size, self.input_size).contiguous()
        x = x.view(bs * num_obj, -1, self.input_size, self.input_size).contiguous()

        x = F.relu(self.t_conv1(x))
        x = self.t_conv2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True).contiguous()

        x = x.view(bs, num_obj, self.output_dim, self.output_size, self.output_size).contiguous().permute(0, 1, 3, 4, 2).contiguous()

        return x


class ConvDecoderAct(nn.Module):
    def __init__(self, input_size, output_size, output_dim, middle_dim):
        super(ConvDecoderAct, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.output_dim = output_dim
        self.middle_dim = middle_dim

        self.fc_1 = nn.Linear(512, 512)
        self.fc_2 = nn.Linear(512, 1024)

        self.t_conv1 = nn.ConvTranspose2d(4, 128, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(128, self.output_dim, 2, stride=2)

    def forward(self, x, x_act):
        bs = 1

        x = torch.cat((x, x_act), dim=0)
        x = F.relu(self.fc_1(x))
        x = self.fc_2(x)

        x = x.view(1, -1, self.input_size, self.input_size)

        x = F.relu(self.t_conv1(x))
        x = self.t_conv2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True).contiguous()

        x = x.view(bs, self.output_dim, self.output_size, self.output_size).permute(0, 2, 3, 1).contiguous()

        # x = torch.cat((x, pos_encode), dim=3).contiguous()

        return x


class ObservationEncoder(nn.Module):
    def __init__(self, d_model: int, act_num: int, image_size: int, object_num: int, dropout: float):
        super().__init__()
        self.image_size = image_size
        self.object_num = None
        self.frame_num = None
        self.img_encoder = models.resnet18(num_classes=d_model)

    def forward(self, input_obs, segmentation):
        bs = len(input_obs)

        self.object_num = segmentation.size()[1]
        self.frame_num = segmentation.size()[2]

        vori_obs = input_obs.view(bs, 1, self.frame_num, self.image_size, self.image_size, 3).repeat(1, self.object_num, 1, 1, 1, 1)
        segmentation = segmentation.view(bs, self.object_num, self.frame_num, self.image_size, self.image_size, 1)
        whole_seg = segmentation.clone()
        for b in range(bs):
            for obj in range(self.object_num):
                for frame in range(self.frame_num):
                    if len(whole_seg[b][obj][frame].nonzero()) > 0:
                        whole_seg[b][obj][frame] = 1

        segmentation = segmentation.repeat(1, 1, 1, 1, 1, 3)

        ori_obs = (vori_obs * segmentation).contiguous()
        obs = ori_obs[:, :, 0].contiguous()
        for ii in range(1, self.frame_num):
            obs = obs + ori_obs[:, :, ii]


        whole_ori_obs = (vori_obs * whole_seg).contiguous()
        whole_obs = whole_ori_obs[:, :, 0].contiguous()
        for ii in range(1, self.frame_num):
            whole_obs = whole_obs + whole_ori_obs[:, :, ii]

        obs = torch.cat((obs, whole_obs), dim=4).contiguous()

        x_img = obs.view(bs * self.object_num, self.image_size, self.image_size, 6).permute(0, 3, 1, 2).contiguous()
        x_img = x_img.view(bs * self.object_num, 6, self.image_size, self.image_size).contiguous()

        x_img, x_early_img = self.img_encoder(x_img)

        x_img = x_img.view(bs, self.object_num, -1)
        x_early_img = x_early_img.view(bs, self.object_num, 64, 32, 32)

        return x_img, x_early_img

class ViewSelector(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.classifier = nn.Sequential(
                          nn.Linear(d_model, d_model),
                          nn.ReLU(),
                          nn.Linear(d_model, 2))

    def forward(self, feat):
        bs = feat.size()[0]
        num_obj = feat.size()[1]

        feat = feat.view(bs, num_obj, -1).contiguous()

        x_pred = self.classifier(feat)

        return x_pred


class SymbolicDecoder(nn.Module):
    def __init__(self, d_model: int, num_syb: int):
        super().__init__()

        self.num_syb = num_syb

        self.single_net = nn.ModuleDict({str(_): nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2)) for _ in range(4)})
        self.double_net = nn.ModuleDict({str(_): nn.Sequential(nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Linear(d_model, 2)) for _ in range(1)})

        self.default_syb = [
                            [0, 0, -1, '0'],
                            [0, 1, -1, '0'],
                            [0, 2, -1, '0'],
                            [0, 6, -1, '1'],
                            [0, 6, -1, '2'],
                            [1, 0, 3, '0'],
                            [1, 0, 4, '0'],
                            [1, 0, 5, '0'],
                            [1, 0, 6, '0'],
                            [1, 1, 3, '0'],
                            [1, 1, 4, '0'],
                            [1, 1, 5, '0'],
                            [1, 1, 6, '0'],
                            [1, 2, 3, '0'],
                            [1, 2, 4, '0'],
                            [1, 2, 5, '0'],
                            [1, 2, 6, '0'],
                            [1, 4, 5, '0'],
                            [1, 4, 6, '0'],
                            [0, 0, -1, '3'],
                            [0, 1, -1, '3'],
                            [0, 2, -1, '3'],
                            ]

    def forward(self, feat, syb=None):

        if syb == None:
            syb = self.default_syb
            syb = [syb for _ in range(len(feat))]
            pre_train = False
        else:
            pre_train = True

        bs = len(feat)

        all_syb = []

        for b in range(bs):
            first = True
            b_syb = None
            for idx in range(len(syb[b])):
                net_id = syb[b][idx][-1] if not pre_train else '0'
                if pre_train and syb[b][idx][2] < -1.5:
                    net_id = '3'

                if syb[b][idx][0] == 0:
                    if first:
                        b_syb = self.single_net[net_id](feat[b, syb[b][idx][1]].unsqueeze(0))
                        first = False
                    else:
                        b_syb = torch.cat((b_syb, self.single_net[net_id](feat[b, syb[b][idx][1]].unsqueeze(0))), dim=0).contiguous()
                if syb[b][idx][0] == 1:
                    if first:
                        b_syb = self.double_net[net_id](torch.cat((feat[b, syb[b][idx][1]].unsqueeze(0), feat[b, syb[b][idx][2]].unsqueeze(0)), dim=1).contiguous())
                        first = False
                    else:
                        b_syb = torch.cat((b_syb, self.double_net[net_id](torch.cat((feat[b, syb[b][idx][1]].unsqueeze(0), feat[b, syb[b][idx][2]].unsqueeze(0)), dim=1).contiguous())), dim=0).contiguous()

            all_syb.append(b_syb.clone().unsqueeze(0).contiguous())

        all_syb = torch.cat((all_syb), dim=0).contiguous()

        return all_syb

    def dual(self, feat, rad_feat, syb=None):
        if syb == None:
            syb = self.default_syb
            pre_train = False
        else:
            pre_train = True

        bs = len(feat)

        all_syb = []

        for b in range(bs):
            first = True
            b_syb = None
            for idx in range(len(syb[b])):
                net_id = syb[b][idx][-1] if not pre_train else '0'
                if syb[b][idx][0] == 0:
                    if first:
                        b_syb = self.single_net[net_id](feat[b, syb[b][idx][1]].unsqueeze(0))
                        first = False
                    else:
                        b_syb = torch.cat((b_syb, self.single_net[net_id](feat[b, syb[b][idx][1]].unsqueeze(0))), dim=0).contiguous()
                if syb[b][idx][0] == 1:
                    if first:
                        b_syb = self.double_net[net_id](torch.cat((feat[b, syb[b][idx][1]].unsqueeze(0), rad_feat[b, syb[b][idx][2]].unsqueeze(0)), dim=1).contiguous())
                        first = False
                    else:
                        b_syb = torch.cat((b_syb, self.double_net[net_id](torch.cat((feat[b, syb[b][idx][1]].unsqueeze(0), rad_feat[b, syb[b][idx][2]].unsqueeze(0)), dim=1).contiguous())), dim=0).contiguous()

            all_syb.append(b_syb.clone().unsqueeze(0).contiguous())

        all_syb = torch.cat((all_syb), dim=0).contiguous()

        return all_syb


class FutureTransModel(nn.Module):
    __constants__ = ['norm']

    def __init__(self, layers, num_layers, num_act, norm=None):
        super(FutureTransModel, self).__init__()

        self.num_act = num_act

        self.layers = ModuleList([copy.deepcopy(layers) for _ in range(num_layers)])

        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):

        bs = len(src)

        act = src[:, (-self.num_act):].contiguous()
        output = src[:, :(-self.num_act+1)].contiguous()
        ans = src[:, :(-self.num_act+1)].unsqueeze(1).contiguous()

        for idx in range(self.num_act):
            output = torch.cat((output[:, :-1].clone(), act[:, idx:(idx + 1)]), dim=1).contiguous()

            for mod in self.layers:
                output = mod(output, src_mask=None, src_key_padding_mask=None)

            if self.norm is not None:
                output = self.norm(output)

            if idx == 0:
                ans = output.clone().unsqueeze(1)
            else:
                ans = torch.cat((ans, output.clone().unsqueeze(1)), dim=1).contiguous()

        return ans



class TransModel(nn.Module):
    __constants__ = ['norm']

    def __init__(self, trans_layers, num_layers, num_act, num_cluster, image_size, object_id_list, norm=None):
        super(TransModel, self).__init__()

        self.num_act = num_act
        self.image_size = image_size

        self.trans_layers = ModuleList([copy.deepcopy(trans_layers) for _ in range(num_layers)])

        self.affordance_decoder = ConvDecoderAct(16, self.image_size, 256, 254)
        self.segmentation_decoder = ConvDecoder(16, self.image_size, 2, 256)
        self.viewselector = ViewSelector(256)

        self.image_idx = [0 for _ in range(self.image_size * self.image_size)] + [1 for _ in range(self.image_size * self.image_size)]
        self.select_idx = [[u, v] for u in range(self.image_size) for v in range(self.image_size)] + [[u, v] for u in range(self.image_size) for v in range(self.image_size)]

        self.select_pos_inputs = [[u, v] for u in range(self.image_size) for v in range(self.image_size)]
        self.select_pos_inputs = Variable(torch.from_numpy(np.array(self.select_pos_inputs).astype(np.float32))).cuda()
        self.select_pos_inputs = self.select_pos_inputs.view(1, 128, 128, 2) / 128.0

        self.object_id_list = object_id_list

        self.num_layers = num_layers
        self.num_cluster = num_cluster
        self.norm = norm

    def forward(self, input: Tensor):
        raise Exception("Not implemented yet")

    def forward_seq(self, src, affordance_index, affordance_select, seg_id, act, init_obs):

        bs = len(src)

        x_act = src[:, -self.num_act:]
        output_trans = src[:, :(-self.num_act+1)]

        ans = output_trans.unsqueeze(1).contiguous()
        output_segmentation = None
        output_view = None

        for idx in range(self.num_act):

            output_trans = torch.cat((output_trans[:, :-1].clone(), x_act[:, idx:(idx+1)]), dim=1).contiguous()

            output = output_trans[:, :-1].clone()
            output_segmentation_sub = None
            output_view_sub = None

            for b in range(bs):
                target_obj = self.object_id_list.index(int(seg_id[b][idx].item()))
                output_affordance = self.affordance_decoder(output[b][target_obj], x_act[b][idx])

                if b == 0:
                    output_segmentation_sub = self.segmentation_decoder(output[b][target_obj].unsqueeze(0).unsqueeze(0).contiguous()).unsqueeze(1).clone()
                    output_view_sub = self.viewselector(output[b][target_obj].unsqueeze(0).unsqueeze(0).contiguous()).unsqueeze(1).clone()
                else:
                    output_segmentation_sub = torch.cat((output_segmentation_sub, self.segmentation_decoder(output[b][target_obj].unsqueeze(0).unsqueeze(0).contiguous()).unsqueeze(1).clone()), dim=1).contiguous()
                    output_view_sub = torch.cat((output_view_sub, self.viewselector(output[b][target_obj].unsqueeze(0).unsqueeze(0).contiguous()).unsqueeze(1).clone()), dim=1).contiguous()

                select = affordance_select[b][idx]
                output_trans[b][target_obj] = output_affordance[0][select[0]][select[1]]

            output_segmentation_sub = output_segmentation_sub.unsqueeze(2).contiguous()
            output_view_sub = output_view_sub.unsqueeze(2).contiguous()
            if idx == 0:
                output_segmentation = output_segmentation_sub.clone()
                output_view = output_view_sub.clone()
            else:
                output_segmentation = torch.cat((output_segmentation, output_segmentation_sub.clone()), dim=2).contiguous()
                output_view = torch.cat((output_view, output_view_sub.clone()), dim=2).contiguous()

            output_trans = output_trans.contiguous()

            for mod in self.trans_layers:
                output_trans = mod(output_trans, src_mask=None, src_key_padding_mask=None)

            if self.norm is not None:
                output_trans = self.norm(output_trans)

            ans = torch.cat((ans, output_trans.clone().unsqueeze(1)), dim=1).contiguous()

        return ans, output_segmentation, output_view


    def forward_single_action_seg(self, src, seg_id):
        self.num_act = 1

        bs = len(src)

        output = src

        for idx in range(self.num_act):

            target_obj = self.object_id_list.index(int(seg_id[0][0].item()))
            output_segmentation = self.segmentation_decoder(output[0][target_obj].unsqueeze(0).unsqueeze(0))

        return output_segmentation[0], target_obj

    def forward_single_action_group(self, src, affordance_info, seg_id):

        num_act = 1

        bs = len(src)

        x_act = src[:, -num_act:].contiguous()
        output_trans = src.clone()

        for idx in range(num_act):
            output_trans = torch.cat((output_trans[:, :-1].clone(), x_act[:, idx:(idx + 1)]), dim=1).contiguous()

            output = output_trans[:, :-1].clone()
            target_obj = self.object_id_list.index(int(seg_id[0][0].item()))

            output_affordance = self.affordance_decoder(output[0][target_obj], x_act[0][idx])

            output_trans = output_trans.repeat(len(affordance_info), 1, 1)

            for b in range(bs):
                for idx2 in range(len(affordance_info)):
                    image_idx = affordance_info[idx2][0]
                    select = affordance_info[idx2][1:3]

                    output_trans[idx2][target_obj] = output_affordance[b][select[0]][select[1]]

            output_trans = output_trans.contiguous()

            for mod in self.trans_layers:
                output_trans = mod(output_trans, src_mask=None, src_key_padding_mask=None)

            if self.norm is not None:
                output_trans = self.norm(output_trans)

        return output_trans



class SybTransModel(nn.Module):

    def __init__(self, obj_num: int, act_num: int, act_num_future: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, nclusters: int, image_size: int, num_syb: int, object_num: int, object_id_list: list, dropout: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.obj_num = obj_num
        self.act_num = act_num
        self.act_num_future = act_num_future
        self.obj_syb_num = obj_num + act_num
        self.nclusters = nclusters
        self.image_size = image_size
        self.num_syb = num_syb
        self.object_num = object_num

        self.encoder = ObservationEncoder(d_model, self.act_num, self.image_size, self.object_num, dropout)

        self.action_encoder = nn.Sequential(
                              nn.Linear(1, d_model),
                              nn.ReLU(),
                              nn.Linear(d_model, d_model))

        self.base_encoder = nn.Sequential(
                            nn.Linear(2, d_model),
                            nn.ReLU(),
                            nn.Linear(d_model, d_model))


        trans_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.trans_model = TransModel(trans_layers, nlayers, self.act_num, self.nclusters, self.image_size, object_id_list)

        self.symbol_decoder = SymbolicDecoder(d_model, num_syb)

        self.act_decoder = nn.Sequential(
                            nn.Linear(d_model, d_model),
                            nn.ReLU(),
                            nn.Linear(d_model, 2))


        self.color_map = {}
        self.color_map_id = 1


    def forward(self, input):
        raise Exception("Not implemented yet")

    def forward_traj(self, obs, base, act, affordance_index, affordance_select, pre_syb, seg_id, future_act, future_succ, future_seg_id, segmentation):

        init_obs = obs[:, 0].contiguous()
        init_seg = segmentation[:, 0].contiguous()
        init_base = base[:, 0].unsqueeze(1).contiguous()

        feat, _ = self.encoder(init_obs, init_seg)

        x_act = self.action_encoder(act)

        x_base = self.base_encoder(init_base)

        x = torch.cat((feat, x_base, x_act), dim=1)
        all_feat = torch.cat((feat, x_base), dim=1).contiguous().unsqueeze(1).contiguous()

        for fr in range(1, obs.size()[1]):
            tmp_feat = torch.cat((self.encoder(obs[:, fr].contiguous(), segmentation[:, fr].contiguous())[0].clone(), self.base_encoder(base[:, fr].unsqueeze(1).contiguous())), dim=1).contiguous().unsqueeze(1).contiguous()
            all_feat = torch.cat((all_feat, tmp_feat), dim=1).contiguous()

        output_feat, output_seg, output_view_select = self.trans_model.forward_seq(x, affordance_index, affordance_select, seg_id, act, init_obs)

        bs = output_feat.size()[0]
        output_feat = output_feat.view(bs, (self.act_num+1), (self.object_num+2), self.d_model).contiguous()
        syb_feat = output_feat[:, :, :-1].contiguous().view(bs * (self.act_num+1), (self.object_num+1), self.d_model).contiguous()
        act_feat = output_feat[:, 1:, -1:].contiguous().view(bs * (self.act_num), self.d_model).contiguous()

        all_feat = all_feat.view(bs * self.act_num, (self.object_num+1), self.d_model).contiguous()
        syb_feat_compare = syb_feat.view(bs, (self.act_num+1), (self.object_num+1), self.d_model).contiguous()
        syb_feat_compare = syb_feat_compare[:, :self.act_num, :, :].contiguous()
        syb_feat_compare = syb_feat_compare.view(bs * self.act_num, (self.object_num+1), self.d_model)

        dis_feat = torch.norm(syb_feat_compare - all_feat, dim=2).view(-1).mean() * 0.01

        all_syb_pred = self.symbol_decoder(all_feat)
        all_syb_pred = all_syb_pred.view(bs, self.act_num, all_syb_pred.size()[1], 2).contiguous()

        syb_pred = self.symbol_decoder(syb_feat)
        act_pred = self.act_decoder(act_feat)

        num_syb = syb_pred.size()[1]
        syb_pred = syb_pred.view(bs, (self.act_num+1), num_syb, 2).contiguous()

        syb_pred_pre = syb_pred[:, :1].contiguous()

        syb_pred_post = syb_pred[:, 1:].contiguous()


        return syb_pred_pre, syb_pred_post, output_view_select, act_pred, output_seg, all_syb_pred, dis_feat

    def forward_pre(self, rgb, seg, syb):

        rgb = rgb.unsqueeze(1).contiguous()
        seg = seg.unsqueeze(2).contiguous()
        feat, _ = self.encoder(rgb, seg)

        output_seg = self.trans_model.segmentation_decoder(feat)

        view_pred = self.trans_model.viewselector(feat)

        syb_pred = self.symbol_decoder(feat, syb)

        return syb_pred, view_pred, output_seg

    def show_affordance(self, obs, base, act, seg_id, seg_init, first=True, feat=None):
        if first:
            feat, _ = self.encoder(obs, seg_init)
            init_base = base[:, 0:1].contiguous()
            x_base = self.base_encoder(init_base)
            feat = torch.cat((feat, x_base), dim=1)

        x_act = self.action_encoder(act)

        x = torch.cat((feat, x_act), dim=1)

        output = np.zeros((1, 128, 128, 3))
        scoremap = np.zeros((1, 128, 128))
        output_seg, target_obj = self.trans_model.forward_single_action_seg(x, seg_id)
        output_view = torch.argmax(self.trans_model.viewselector(x)[0][target_obj])

        output_seg = torch.softmax(output_seg, dim=3)

        mask = (output_seg[:, :, :, 1] - output_seg[:, :, :, 0])
        for threshold in [0.9999999, 0.99999, 0.999, 0.97, 0.93, 0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.0]:
            num = len(torch.nonzero(mask > threshold))
            if num > 200:
                break
        mask = mask > threshold
        out_mask = output_seg[:, :, :, 1] > output_seg[:, :, :, 0]

        nonzero_idx = torch.nonzero(mask)
        nonzero_idx[:, 0] = output_view

        perm = torch.randperm(nonzero_idx.size(0))
        idx = perm[:50]
        nonzero_idx = nonzero_idx[idx]

        if len(nonzero_idx) == 0:
            return None, None, None, out_mask

        output_affordance = self.trans_model.forward_single_action_group(x, nonzero_idx, seg_id)
        output_syb_ori = self.symbol_decoder(output_affordance)
        output_syb = torch.argmax(output_syb_ori, dim=2)
        criterion = nn.CrossEntropyLoss()

        output_confidence = [criterion(output_syb_ori[i], output_syb[i]).item() for i in range(len(nonzero_idx))]
        output_confidence = np.array(output_confidence)
        output_confidence = (max(output_confidence) - output_confidence) / max(output_confidence)

        final_output = {}

        for count in range(len(nonzero_idx)):
            i, j, k = nonzero_idx[count][0], nonzero_idx[count][1], nonzero_idx[count][2]
            str = np.array2string(output_syb[count][..., :13].detach().cpu().numpy())

            scoremap[i][j][k] = (output_seg[0][j][k][1] - output_seg[0][j][k][0]) * output_confidence[count]# * succ_conf
            if not str in final_output.keys():

                if self.color_map_id > len(color_list):
                    self.color_map_id = 0

                final_output[str] = {'feat': [], 'score': [], 'act': []}
                final_output[str]['feat'].append(output_affordance[count][:-1].detach())
                final_output[str]['score'].append(scoremap[i][j][k].item())
                final_output[str]['act'].append(
                    [act.view(-1).long().item(), seg_id.view(-1).long().item(), i.item(), j.item(), k.item()])
            else:

                final_output[str]['feat'].append(output_affordance[count][:-1].detach())
                final_output[str]['score'].append(scoremap[i][j][k].item())
                final_output[str]['act'].append(
                    [act.view(-1).long().item(), seg_id.view(-1).long().item(), i.item(), j.item(), k.item()])

        output = output.astype(np.int16)

        return output, scoremap, final_output, out_mask


