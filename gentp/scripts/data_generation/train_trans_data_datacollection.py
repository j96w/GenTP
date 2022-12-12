import os
import igibson
import copy
import random
import numpy as np

from gentp.lib.env import IndoorEnv
from gentp.lib.dataset import SybDataset
from gentp.configs.base_config import Configs

def get_e_greedy_action(configs, env, syb, seg):
    select_idx = [[u, v] for u in range(configs.image_size) for v in range(configs.image_size)] + [[u, v] for u in range(configs.image_size) for v in range(configs.image_size)]

    num_pt = 10

    if syb[3] == 1:
        if syb[4] == 1 and (syb[6] == 1 or syb[10] == 1 or syb[14] == 1):
            act = 2
        else:
            act = 1
    else:
        act = 0

    seg = np.array(seg)
    seg = seg.reshape(-1)
    seg_cand = [env.graspable_obj_id,
                env.placeable_obj_id,
                env.pourable_obj_id,
                env.pan_placeable_obj_id]

    obj_cand = env.target_obj_id_list

    valid_count = 0

    additional_info = []

    if syb[4] == 1 and act == 1:
        for seg_id in seg_cand[-1]:
            idx_set = np.where(seg == seg_id)[0]
            if len(idx_set) > 0:
                valid_count += 1
                target_set = idx_set[np.random.choice(len(idx_set), size=num_pt)]
                for idx in target_set:
                    additional_info.append([act, 0, select_idx[idx][0], select_idx[idx][1], seg_id])
    else:
        if act == 0:
            grasp_pan_coin = random.randint(0, 4)
            if grasp_pan_coin <= 1:
                candidate_list = seg_cand[act][:1] + seg_cand[act][-1:]
            else:
                candidate_list = seg_cand[act]
        else:
            candidate_list = seg_cand[act]

        for seg_id in candidate_list:
            idx_set = np.where(seg == seg_id)[0]
            if len(idx_set) > 0:
                valid_count += 1
                target_set = idx_set[np.random.choice(len(idx_set), size=num_pt)]
                for idx in target_set:
                    additional_info.append([act, 0, select_idx[idx][0], select_idx[idx][1], seg_id])

    if len(additional_info) == 0:
        return True, None, None, None, None, None, None

    best_idx = random.randint(0, len(additional_info) - 1)

    output_seg = seg.reshape(1, 1, 128, 128)
    answer_seg = None
    first = True
    for i in range(len(obj_cand)):
        if first:
            answer_seg = copy.deepcopy(output_seg == obj_cand[i])
            first = False
        else:
            answer_seg = np.concatenate((answer_seg, output_seg == obj_cand[i]), axis=0)

    output_seg = (seg.reshape(1, 128, 128) == additional_info[best_idx][4])

    return False, additional_info[best_idx][0], additional_info[best_idx][1], [additional_info[best_idx][2],
                                                                               additional_info[best_idx][3]], \
           additional_info[best_idx][4], answer_seg, output_seg

def data_collection(configs, env, buffer):
    traj_id = 0

    while traj_id < configs.datagen_trans_num_epochs:
        print(traj_id)
        tmp_traj = {}
        tmp_traj['obs'] = []
        tmp_traj['base'] = []
        tmp_traj['act'] = []
        tmp_traj['score'] = []
        tmp_traj['affordance_index'] = []
        tmp_traj['affordance_select'] = []
        tmp_traj['seg_id'] = []
        tmp_traj['segmentation'] = []
        tmp_traj['segmentation_init'] = []
        tmp_traj['pre_syb'] = []
        tmp_traj['post_syb'] = []
        tmp_traj['succ'] = []

        visited = []

        env.reset()

        while sum(env.get_syb_state()[-14:-2]) != 3 or sum(env.get_syb_state()[-2:]) != 1:
            env.reset()

        env.random_coin += 1
        if env.random_coin > 10:
            env.random_coin = 0

        empty = False

        for step in range(configs.datagen_trans_max_steps):
            tmp_traj['pre_syb'].append(np.array(env.get_syb_state()).astype(np.int16).copy())
            tmp_traj['obs'].append(np.array(env.view_rgb).astype(np.float32).copy())
            tmp_traj['base'].append(np.array([int(env.in_hand == -1), int(env.in_hand == env.utils_id[0])]).astype(np.float32).copy())

            if step == 0:
                visited.append(np.array2string(tmp_traj['pre_syb'][-1]))

            empty, act, affordance_index, affordance_select, seg_id, segmentation_init, segmentation = get_e_greedy_action(configs, env, env.get_syb_state(), env.view_seg)
            if empty:
                break

            tmp_traj['act'].append(np.array([act]).astype(np.float32).copy())
            tmp_traj['affordance_index'].append(np.array([affordance_index]).astype(np.float32).copy())
            tmp_traj['affordance_select'].append(np.array(affordance_select).astype(np.float32).copy())
            tmp_traj['seg_id'].append(np.array([seg_id]).astype(np.float32).copy())
            tmp_traj['segmentation'].append(np.array(segmentation).astype(np.int16).copy())
            tmp_traj['segmentation_init'].append(np.array(segmentation_init).astype(np.int16).copy())

            tmp_traj['score'].append(np.array([float(len(np.unique(visited))) / 10.0]).astype(np.float32).copy())

            if tmp_traj['act'][-1] == 0:
                env.arm_pick(affordance_index, affordance_select[0], affordance_select[1])
            elif tmp_traj['act'][-1] == 1:
                env.arm_place(affordance_index, affordance_select[0], affordance_select[1])
            elif tmp_traj['act'][-1] == 2:
                env.arm_pour(affordance_index, affordance_select[0], affordance_select[1])

            tmp_traj['post_syb'].append(np.array(env.get_syb_state()).astype(np.int16).copy())

            tmp_traj['succ'].append(np.array([int(np.array2string(tmp_traj['post_syb'][-1]) != np.array2string(tmp_traj['pre_syb'][-1]))]).astype(np.int16).copy())

            visited.append(np.array2string(tmp_traj['post_syb'][-1]))

        if not empty:
            print(tmp_traj['pre_syb'][0])
            print(tmp_traj['post_syb'])
            print(tmp_traj['succ'])
            print(tmp_traj['score'])
            buffer.add_to_dataset(tmp_traj.copy(), traj_id)
            traj_id += 1

def main():
    configs = Configs()

    env = IndoorEnv(os.path.join(igibson.example_config_path, configs.env_config_file), 'train', "iggui")
    obj_id_list = env.target_obj_id_list

    print(obj_id_list)

    data_buffer = SybDataset(configs.buffer_size, configs.num_act, configs.num_future_act)
    data_buffer.start_dataset(configs.datagen_trans_num_epochs, 'for_layout_obj-0', 'lay-2-fixpouring')

    data_collection(configs, env, data_buffer)

    data_buffer.end_dataset()


if __name__ == "__main__":
    main()
