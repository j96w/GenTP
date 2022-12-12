import copy
import torch
from torch.utils.data import Dataset
import numpy as np
import h5py

from gentp.configs.base_config import Configs


class PreDataset(Dataset):
    def __init__(self, buffer_size, num_act, num_future_act, mode):
        self.configs = Configs()
        self.data = {}
        self.data_max_idx = 0
        self.num_act = num_act
        self.num_future_act = num_future_act
        self.mode = mode

        self.buffer_size = buffer_size
        self.test_size = 20

        self.scale = 4

        if self.mode == 'train':
            self.data_list = [self.configs.pretrain_train_data_path]
            self.single_len = 5000
        else:
            self.data_list = [self.configs.pretrain_test_data_path]
            self.single_len = 500

        self.data_file = []
        self.keys = ['Cooked', 'Cookable', 'OnTop']

        self.empty_image = [[[0] for _ in range(128)] for __ in range(128)]
        self.empty_image = torch.from_numpy(np.array(self.empty_image).astype(np.int16)).long()

        for name in self.data_list:
            self.data_file.append(h5py.File('{}'.format(name), 'r'))

        print(mode, self.data_list)

    def __len__(self):
        return self.single_len * len(self.data_list)

    def __getitem__(self, idx_input):
        idx = idx_input // self.single_len
        start_id = idx_input % self.single_len

        rgb = self.data_file[idx]['frame_{0}'.format(start_id)]['rgb']

        seg = self.data_file[idx]['frame_{0}'.format(start_id)]['seg']
        cook = self.data_file[idx]['frame_{0}'.format(start_id)]['Cooked']
        cookable = self.data_file[idx]['frame_{0}'.format(start_id)]['Cookable']
        ontop = self.data_file[idx]['frame_{0}'.format(start_id)]['OnTop']

        rgb = torch.from_numpy(np.array(rgb).astype(np.float32))
        seg = torch.from_numpy(np.array(seg).astype(np.int16)).long()
        cook = torch.from_numpy(np.array(cook).astype(np.int16)).long()
        cookable = torch.from_numpy(np.array(cookable).astype(np.int16)).long()
        ontop = torch.from_numpy(np.array(ontop).astype(np.int16)).long()

        obj_id_list = []
        ans_seg = []
        ans_syb = []

        for item in cook:
            if item[0] not in obj_id_list:
                obj_id_list.append(item[0].item())
                ans_seg.append((seg == item[0]).unsqueeze(0))
            ans_syb.append([0, obj_id_list.index(item[0].item()), -1, item[1].item()])

        for item in cookable:
            if item[0] not in obj_id_list:
                obj_id_list.append(item[0].item())
                ans_seg.append((seg == item[0]).unsqueeze(0))
            ans_syb.append([0, obj_id_list.index(item[0].item()), -2, item[1].item()])

        for item in ontop:
            if item[0].item() not in obj_id_list:
                obj_id_list.append(item[0].item())
                ans_seg.append((seg == item[0]).unsqueeze(0))
            if item[1].item() not in obj_id_list:
                obj_id_list.append(item[1].item())
                ans_seg.append((seg == item[1]).unsqueeze(0))
            ans_syb.append([1, obj_id_list.index(item[0].item()), obj_id_list.index(item[1].item()), item[2].item()])

        ans_seg = torch.cat(ans_seg, dim=0).long()
        ans_syb = torch.from_numpy(np.array(ans_syb).astype(np.int16)).long().view(-1, 4)

        view_label = torch.from_numpy(np.array([0]).astype(np.int16)).long()

        ans_seg = ans_seg.repeat(20 // ans_seg.size()[0] + 1, 1, 1, 1)[:20]
        ans_syb = ans_syb.repeat(80 // ans_syb.size()[0] + 1, 1)[:80]

        view_label = view_label.repeat(20 // view_label.size()[0] + 1, 1)[:20].view(-1)

        return rgb, ans_seg, ans_syb, view_label

class TransDataset(Dataset):
    def __init__(self, buffer_size, num_act, num_future_act):
        self.configs = Configs()

        self.data_max_idx = 0
        self.num_act = num_act
        self.num_future_act = num_future_act
        self.mode = 'train'

        self.data = {'train': [h5py.File(self.configs.trans_train_data_path, 'r', driver='core')],
                     'test': [h5py.File(self.configs.trans_test_data_path, 'r', driver='core')]}

        self.obj_set_num_train = 1
        self.obj_set_num_test = 1
        self.train_size = 1000
        self.test_size = 100

        self.scale = 4

    def __len__(self):
        if self.mode == 'train':
            return self.obj_set_num_train * self.train_size * self.scale
        else:
            return self.obj_set_num_test * self.test_size * self.scale

    def switch_mode(self, mode):
        self.mode = mode

    def __getitem__(self, idx_input):
        if self.mode == 'train':
            set_id = idx_input // (self.train_size * self.scale)
            idx_set = idx_input % (self.train_size * self.scale)
        else:
            set_id = idx_input // (self.test_size * self.scale)
            idx_set = idx_input % (self.test_size * self.scale)
        idx = idx_set // self.scale
        start_id = idx_set % self.scale

        data_idx_str = 'traj_{0}'.format(idx)

        obs = self.data[self.mode][set_id][data_idx_str]['obs'][start_id:(start_id+self.num_act)]
        base = self.data[self.mode][set_id][data_idx_str]['base'][start_id:(start_id + self.num_act)]
        act = self.data[self.mode][set_id][data_idx_str]['act'][start_id:(start_id + self.num_act)]
        succ = self.data[self.mode][set_id][data_idx_str]['succ'][start_id:(start_id + self.num_act)]
        score = self.data[self.mode][set_id][data_idx_str]['score'][start_id:(start_id + self.num_act)]
        affordance_index = self.data[self.mode][set_id][data_idx_str]['affordance_index'][start_id:(start_id + self.num_act)]
        affordance_select = self.data[self.mode][set_id][data_idx_str]['affordance_select'][start_id:(start_id + self.num_act)]
        seg_id = self.data[self.mode][set_id][data_idx_str]['seg_id'][start_id:(start_id + self.num_act)]
        segmentation = self.data[self.mode][set_id][data_idx_str]['segmentation'][start_id:(start_id + self.num_act)]
        segmentation_init = self.data[self.mode][set_id][data_idx_str]['segmentation_init'][start_id:(start_id + self.num_act)]

        pre_syb = self.data[self.mode][set_id][data_idx_str]['pre_syb'][start_id:(start_id+self.num_act)]
        post_syb = self.data[self.mode][set_id][data_idx_str]['post_syb'][start_id:(start_id + self.num_act)]

        future_post = self.data[self.mode][set_id][data_idx_str]['post_syb'][(start_id + self.num_act):(start_id + self.num_act + self.num_future_act)]
        future_act = self.data[self.mode][set_id][data_idx_str]['act'][(start_id + self.num_act):(start_id + self.num_act + self.num_future_act)]
        future_succ = self.data[self.mode][set_id][data_idx_str]['succ'][(start_id + self.num_act):(start_id + self.num_act + self.num_future_act)]
        future_seg_id = self.data[self.mode][set_id][data_idx_str]['seg_id'][(start_id + self.num_act):(start_id + self.num_act + self.num_future_act)]


        obs = torch.from_numpy(np.array(obs).astype(np.float32))
        base = torch.from_numpy(np.array(base).astype(np.float32))
        act = torch.from_numpy(np.array(act).astype(np.float32))
        succ = torch.from_numpy(np.array(succ).astype(np.int16)).long()
        score = torch.from_numpy(np.array(score).astype(np.float32))
        affordance_index = torch.from_numpy(np.array(affordance_index).astype(np.int16)).long()
        affordance_select = torch.from_numpy(np.array(affordance_select).astype(np.int16)).long()
        seg_id = torch.from_numpy(np.array(seg_id).astype(np.float32))
        segmentation = torch.from_numpy(np.array(segmentation).astype(np.int16)).long()

        view_label = []
        for item in range(len(affordance_index)):
            tmp_append = [0, 0]
            tmp_append[affordance_index[item]] = 1
            view_label.append(copy.deepcopy(tmp_append))
        view_label = torch.from_numpy(np.array(view_label).astype(np.int16)).long()

        segmentation_init = torch.from_numpy(np.array(segmentation_init).astype(np.int16)).long()

        pre_syb = torch.from_numpy(np.array(pre_syb).astype(np.int16)).long()
        post_syb = torch.from_numpy(np.array(post_syb).astype(np.int16)).long()

        act = torch.from_numpy(np.array(act).astype(np.float32))

        future_syb = torch.from_numpy(np.array(future_post).astype(np.int16)).long()
        future_act = torch.from_numpy(np.array(future_act).astype(np.float32))
        future_succ = torch.from_numpy(np.array(future_succ).astype(np.int16)).long()
        future_seg_id = torch.from_numpy(np.array(future_seg_id).astype(np.float32))

        cookable_syb = torch.from_numpy(np.array([1, 0, 0]).astype(np.int16)).long()
        cookable_syb = cookable_syb.view(1, 3).repeat(self.num_act, 1)
        pre_syb = torch.cat((pre_syb, cookable_syb), dim=1)
        post_syb = torch.cat((post_syb, cookable_syb), dim=1)

        return obs, base, act, succ, score, affordance_index, affordance_select, seg_id, pre_syb, post_syb, segmentation, segmentation_init, future_syb, future_act, future_succ, future_seg_id, view_label



class SybDataset(Dataset):
    def __init__(self, buffer_size, num_act, num_future_act):
        self.configs = Configs()
        self.data = {}
        self.data_max_idx = 0
        self.num_act = num_act
        self.num_future_act = num_future_act
        self.mode = 'train'

        self.buffer_size = buffer_size
        self.test_size = 20

        self.scale = 4

    def __len__(self):
        if self.mode == 'train':
            return (self.buffer_size - self.test_size) * self.scale
        else:
            return self.test_size * self.scale

    def switch_mode(self, mode):
        self.mode = mode

    def generate_future_data(self, goal, act, id):
        while len(act) < self.num_future_act:
            act.append(-1)
            id.append(-1)
        return np.concatenate((goal, act, id), axis=0)

    def __getitem__(self, idx_input):
        idx = idx_input // self.scale
        start_id = idx_input % self.scale

        if self.mode == 'test':
            idx += 100

        obs = self.data[idx]['obs'][start_id:(start_id+self.num_act)]
        base = self.data[idx]['base'][start_id:(start_id + self.num_act)]
        act = self.data[idx]['act'][start_id:(start_id + self.num_act)]
        succ = self.data[idx]['succ'][start_id:(start_id + self.num_act)]
        score = self.data[idx]['score'][start_id:(start_id + self.num_act)]
        affordance_index = self.data[idx]['affordance_index'][start_id:(start_id + self.num_act)]
        affordance_select = self.data[idx]['affordance_select'][start_id:(start_id + self.num_act)]
        seg_id = self.data[idx]['seg_id'][start_id:(start_id + self.num_act)]
        segmentation = self.data[idx]['segmentation'][start_id:(start_id + self.num_act)]
        segmentation_init = self.data[idx]['segmentation_init'][start_id:(start_id + self.num_act)]

        pre_syb = self.data[idx]['pre_syb'][start_id:(start_id+self.num_act)]
        post_syb = self.data[idx]['post_syb'][start_id:(start_id + self.num_act)]

        future_post = self.data[idx]['post_syb'][(start_id + self.num_act):(start_id + self.num_act + self.num_future_act)]
        future_act = self.data[idx]['act'][(start_id + self.num_act):(start_id + self.num_act + self.num_future_act)]
        future_succ = self.data[idx]['succ'][(start_id + self.num_act):(start_id + self.num_act + self.num_future_act)]
        future_seg_id = self.data[idx]['seg_id'][(start_id + self.num_act):(start_id + self.num_act + self.num_future_act)]


        obs = torch.from_numpy(np.array(obs).astype(np.float32))
        base = torch.from_numpy(np.array(base).astype(np.float32))
        act = torch.from_numpy(np.array(act).astype(np.float32))
        succ = torch.from_numpy(np.array(succ).astype(np.int16)).long()
        score = torch.from_numpy(np.array(score).astype(np.float32))
        affordance_index = torch.from_numpy(np.array(affordance_index).astype(np.int16)).long()
        affordance_select = torch.from_numpy(np.array(affordance_select).astype(np.int16)).long()
        seg_id = torch.from_numpy(np.array(seg_id).astype(np.float32))
        segmentation = torch.from_numpy(np.array(segmentation).astype(np.int16)).long()

        view_label = []
        for item in range(len(affordance_index)):
            tmp_append = [0, 0]
            tmp_append[affordance_index[item]] = 1
            view_label.append(copy.deepcopy(tmp_append))
        view_label = torch.from_numpy(np.array(view_label).astype(np.int16)).long()

        segmentation_init = torch.from_numpy(np.array(segmentation_init).astype(np.int16)).long()

        pre_syb = torch.from_numpy(np.array(pre_syb).astype(np.int16)).long()
        post_syb = torch.from_numpy(np.array(post_syb).astype(np.int16)).long()

        post_syb = torch.from_numpy(np.array(post_syb).astype(np.int16)).long()
        act = torch.from_numpy(np.array(act).astype(np.float32))

        future_syb = torch.from_numpy(np.array(future_post).astype(np.int16)).long()
        future_act = torch.from_numpy(np.array(future_act).astype(np.float32))
        future_succ = torch.from_numpy(np.array(future_succ).astype(np.int16)).long()
        future_seg_id = torch.from_numpy(np.array(future_seg_id).astype(np.float32))

        return obs, base, act, succ, score, affordance_index, affordance_select, seg_id, pre_syb, post_syb, segmentation, segmentation_init, future_syb, future_act, future_succ, future_seg_id, view_label


    def add_trajectory(self, traj):
        self.data[self.data_max_idx] = traj.copy()
        self.data_max_idx = (self.data_max_idx + 1) % self.buffer_size

    def start_dataset(self, num, obj_name, layout_name):
        self.dataset_f = h5py.File('{}/datasets/generated_data_trans_{}.hdf5'.format(self.configs.base_path, num), 'w')

    def end_dataset(self):
        self.dataset_f.close()

    def add_to_dataset(self, traj, frame):
        self.dataset_f.create_group('traj_{0}'.format(frame))
        for key in traj.keys():
            print(frame, key)
            self.dataset_f['traj_{0}'.format(frame)].create_dataset(key, data=np.array(traj[key]))

