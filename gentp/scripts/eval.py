import os
import numpy as np

import torch
from torch.autograd import Variable

import igibson
import matplotlib.pyplot as plt

from gentp.lib.env import IndoorEnv
from gentp.lib.model import SybTransModel
from gentp.configs.base_config import Configs

import copy


fig_size = [None, None]
view_select = []
click = []
qual_id = 0

def show_affordance_map(env, model, act_idx, seg_id):
    obs = torch.from_numpy(np.array([env.view_rgb]).astype(np.float32))
    base = torch.from_numpy(np.array([int(env.in_hand == -1), int(env.in_hand == env.utils_id[0])]).astype(np.float32))
    act = torch.from_numpy(np.array([act_idx]).astype(np.float32))
    seg_id = torch.from_numpy(np.array([seg_id]).astype(np.float32))

    obj_cand = env.target_obj_id_list
    seg = np.array(env.view_seg)
    seg = seg.reshape(-1)
    seg = np.array(seg).astype(np.int16).copy()
    output_seg = seg.reshape(1, 1, 128, 128)
    answer_seg = None
    first = True

    for i in range(len(obj_cand)):
        if first:
            answer_seg = copy.deepcopy(output_seg == obj_cand[i])
            first = False
        else:
            answer_seg = np.concatenate((answer_seg, output_seg == obj_cand[i]), axis=0)

    seg_init = torch.from_numpy(np.array(answer_seg).astype(np.int16))

    obs, base, act, seg_id, seg_init = Variable(obs).cuda(), Variable(base).cuda().unsqueeze(0).unsqueeze(0), Variable(act).cuda().unsqueeze(0).unsqueeze(0), Variable(seg_id).cuda().unsqueeze(0), Variable(seg_init).cuda().unsqueeze(0)
    output, scoremap, final_output, mask = model.show_affordance(obs, base, act, seg_id, seg_init)

    f, axarr = plt.subplots(3, 2)

    axarr[0][0].imshow(env.high_rgb[0], interpolation='none')
    axarr[1][0].imshow(output[0], interpolation='none')
    axarr[1][1].imshow(mask.detach().cpu().numpy()[0], interpolation='none')
    axarr[2][0].imshow(scoremap[0], interpolation='none')

    plt.show()

def check_syb(current, goal):
    n_current = [int(item) for item in current[1:-1].split(' ')]
    n_goal = [int(item) for item in goal[1:-1].split(' ')]

    return (n_current[1] == n_goal[1]) and (n_current[9:13] == n_goal[9:13])

def bfs(env, model, goal):
    goal_str = np.array2string(np.array(goal))
    max_step = 4
    num_pixel = 1
    seg_list = [env.graspable_obj_id, env.placeable_obj_id, env.pourable_obj_id, env.pan_placeable_obj_id]

    state = []
    count = [0, 0, 0, 0, 0, 0]
    answer = []

    st = 0
    ed = 0

    obs = torch.from_numpy(np.array([env.view_rgb]).astype(np.float32))
    base = torch.from_numpy(np.array([int(env.in_hand == -1), int(env.in_hand == env.utils_id[0])]).astype(np.float32))
    obs, base = Variable(obs).cuda(), Variable(base).cuda().unsqueeze(0)

    obj_cand = env.target_obj_id_list
    seg = np.array(env.view_seg)
    seg = seg.reshape(-1)
    seg = np.array(seg).astype(np.int16).copy()
    output_seg = seg.reshape(1, 1, 128, 128)
    answer_seg = None
    first = True

    for i in range(len(obj_cand)):
        if first:
            answer_seg = copy.deepcopy(output_seg == obj_cand[i])
            first = False
        else:
            answer_seg = np.concatenate((answer_seg, output_seg == obj_cand[i]), axis=0)

    seg_init = torch.from_numpy(np.array(answer_seg).astype(np.int16))
    seg_init = Variable(seg_init).cuda().unsqueeze(0)

    feat, _ = model.encoder(obs, seg_init)
    init_base = base.unsqueeze(0).contiguous()
    x_base = model.base_encoder(init_base)
    feat = torch.cat((feat, x_base), dim=1)

    syb = model.symbol_decoder(feat)
    syb = torch.argmax(syb, dim=2).view(-1)

    state.append({'feat': feat.detach().clone(), 'syb': np.array2string(syb.detach().cpu().numpy()), 'score': 1, 'depth': 0, 'his_act': []})
    ed += 1

    while st < ed:
        current_state = state[st]
        current_feat = current_state['feat'].cuda()

        current_syb = current_state['syb']
        current_depth = current_state['depth']

        count[current_depth] += 1

        current_his_act = current_state['his_act']
        current_score = current_state['score']

        if current_depth >= max_step:
            break

        if check_syb(current_syb, goal_str) and len(current_state['his_act']) == 4:
            answer.append([current_state['his_act'], current_state['score']])

        if current_syb.split(' ')[3] == '1' and current_depth != 0:
            if current_syb.split(' ')[4] == '1' and (current_syb.split(' ')[6] == '1' or current_syb.split(' ')[10] == '1'):# or current_syb.split(' ')[14] == '1'):
                act_idx = 2
            else:
                act_idx = 1
        else:
            act_idx = 0

        seg_act_idx = act_idx
        if current_syb.split(' ')[4] == '1' and act_idx == 1:
            seg_act_idx = -1

        for target in seg_list[seg_act_idx]:

            act = torch.from_numpy(np.array([act_idx]).astype(np.float32))
            seg_id = torch.from_numpy(np.array([target]).astype(np.float32))
            act, seg_id = Variable(act).cuda().unsqueeze(0).unsqueeze(0), Variable(seg_id).cuda().unsqueeze(0)

            output_syb, scoremap, outputs, _ = model.show_affordance(None, None, act, seg_id, seg_init, False, current_feat)

            if outputs == None:
                continue

            candidates = list(outputs.keys())
            print(candidates)
            for i in range(len(candidates)):
                candidate_score = outputs[candidates[i]]['score']
                candidate_feat = outputs[candidates[i]]['feat']
                candidate_act = outputs[candidates[i]]['act']

                sort_idx = np.argsort(candidate_score)[-num_pixel:]

                if check_syb(candidates[i], goal_str) and len(current_his_act) == 3:
                    answer.append([current_his_act + [candidate_act[sort_idx[-1]]], current_score * candidate_score[-1]])

                for j in sort_idx:
                    sum_item = 0
                    for item in candidates[i][1:-1].split(' ')[-8:]:
                        sum_item += int(item)
                    if int(candidates[i][1:-1].split(' ')[3]) == 1 and int(candidates[i][1:-1].split(' ')[4]) != 1:
                        sum_item += 1

                    skill_sense = False
                    if len(current_his_act) > 0 and (candidate_act[j][0] == current_his_act[-1][0]):
                        skill_sense = True

                    if skill_sense or sum_item != 2:
                        continue

                    state.append({'feat': candidate_feat[j].unsqueeze(0).detach().cpu(), 'syb': candidates[i], 'score': current_score * candidate_score[j], 'depth': (current_depth + 1), 'his_act': (current_his_act + [candidate_act[j]])})

                    ed += 1
        st += 1

    best = -1.0
    best_traj = None
    for item in answer:
        if item[1] > best and item[0][-1][0] == 2:
            best_traj = item[0]
            best = item[1]

    return best_traj

def main():
    configs = Configs()
    env = IndoorEnv(os.path.join(igibson.example_config_path, configs.env_config_file), "eval", "iggui")

    eval_model_list = configs.eval_model_list

    success = [[] for _ in range(len(eval_model_list))]

    for j in range(30):

        for model_path in range(len(eval_model_list)):
            model = SybTransModel(configs.num_item, configs.num_act, configs.num_future_act, configs.emsize,
                                  configs.nhead, configs.d_hid, configs.nlayers, configs.num_candidate,
                                  configs.image_size, configs.num_syb, configs.object_num, configs.obj_id_list, configs.dropout).to(configs.device)

            model.load_state_dict(torch.load(eval_model_list[model_path]))

            model.eval()

            env.reset()

            global fig_size
            global view_select
            global click
            fig_size = [None, None]
            view_select = []
            click = []

            while sum(env.get_syb_state()[7:9]) + env.get_syb_state()[9] + sum(env.get_syb_state()[11:13]) + sum(env.get_syb_state()[15:17]) != 3:
                env.reset()
            goal = [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]

            best_traj = bfs(env, model, goal)
            if best_traj == None:
                success[model_path].append(0)
                print("fail!", model_path, success[model_path], best_traj)
                continue

            env.update_pcd = False

            for i in range(len(best_traj)):
                if best_traj[i][0] == 0:
                    env.arm_pick(best_traj[i][2], best_traj[i][3], best_traj[i][4], skip_motion=False)
                elif best_traj[i][0] == 1:
                    env.arm_place(best_traj[i][2], best_traj[i][3], best_traj[i][4], skip_motion=False)
                else:
                    env.arm_pour(best_traj[i][2], best_traj[i][3], best_traj[i][4], skip_motion=False)

                env.get_syb_state()

            if np.array(goal).astype(np.int16)[9] == np.array(env.get_syb_state()).astype(np.int16)[9] and np.array(goal).astype(np.int16)[1] == np.array(env.get_syb_state()).astype(np.int16)[1]:
                success[model_path].append(1)
                print("success!", model_path, success[model_path], best_traj)
            else:
                success[model_path].append(0)
                print("fail!", model_path, success[model_path], best_traj)

    env.close()

if __name__ == "__main__":
    main()
