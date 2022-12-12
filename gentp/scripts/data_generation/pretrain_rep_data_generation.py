import os
import random

import igibson
import cv2
import copy
import time
import torch
import numpy as np

import pybullet as p

from igibson.object_states.soaked import Soaked
from igibson.object_states.on_top import OnTop
from igibson.object_states.cooked import Cooked

from gentp.lib.env_datacollection import IndoorEnv
from gentp.configs.base_config import Configs

import h5py


def main():
    configs = Configs()

    env = IndoorEnv(os.path.join(igibson.example_config_path, configs.env_datagen_config_file), "iggui")

    f = h5py.File('{}/datasets/generated_data_pretrain_{}.hdf5'.format(configs.base_path, configs.datagen_num_frames), 'w')

    for frame in range(configs.datagen_num_frames):
        print("Trial {0}".format(frame))

        while True:
            for _ in range(5):
                env.reset()

            exist_object_list = env.env.scene.objects_by_name.keys()
            for item in exist_object_list:
                if Soaked in env.env.scene.objects_by_name[item].states.keys():
                    env.env.scene.objects_by_name[item].states[Soaked].set_value(random.randint(0, 1))

            env.update_view(True)
            predicates = {'Cooked': [], 'Cookable':[], 'OnTop': []}
            seg = env.view_seg[0]
            seg = np.array(seg)
            unique_id = list(np.unique(seg))
            unique_id_name = []
            for item in unique_id:
                if item in env.food_id:
                    unique_id_name.append(env.food_name[env.food_id.index(item)])
                elif item in env.ycb_id:
                    unique_id_name.append(env.ycb_name[env.ycb_id.index(item)])
                else:
                    unique_id_name.append(str(p.getBodyInfo(item)[0], encoding='utf-8'))

            env.reset_obj()

            env.update_view(False)
            seg = env.view_seg[0]
            seg = np.array(seg)
            unique_id = list(np.unique(seg))
            unique_id_name = []
            for item in unique_id:
                if item in env.food_id:
                    unique_id_name.append(env.food_name[env.food_id.index(item)])
                elif item in env.ycb_id:
                    unique_id_name.append(env.ycb_name[env.ycb_id.index(item)])
                else:
                    unique_id_name.append(str(p.getBodyInfo(item)[0], encoding='utf-8'))
            unique_pixel_count = [len((seg == item).nonzero()[0]) for item in unique_id]


            for i in range(len(unique_id)):
                if not(unique_id_name[i] in exist_object_list) or (unique_id_name[i] in configs.neglect_obj) or (unique_pixel_count[i] < 10):
                    continue
                if (not unique_id_name[i] in env.food_name) and (not unique_id_name[i] in env.bowl_name) and (not unique_id_name[i] in env.utils_name) and (not unique_id_name[i] in env.ycb_name):
                    continue
                if Cooked in env.env.scene.objects_by_name[unique_id_name[i]].states.keys():
                    predicates['Cooked'].append([copy.deepcopy(unique_id[i]), int(env.env.scene.objects_by_name[unique_id_name[i]].states[Cooked].get_value())])
                    print("Cooked, ", unique_id_name[i], unique_pixel_count[i], predicates['Cooked'][-1])
                    predicates['Cookable'].append([copy.deepcopy(unique_id[i]), int(True)])
                    print("Cookable, ", unique_id_name[i], unique_pixel_count[i], predicates['Cookable'][-1])
                else:
                    predicates['Cookable'].append([copy.deepcopy(unique_id[i]), int(False)])
                    print("Cookable, ", unique_id_name[i], unique_pixel_count[i], predicates['Cookable'][-1])
                for j in range(len(unique_id)):
                    if i == j or not(unique_id_name[j] in exist_object_list) or (unique_id_name[j] in configs.neglect_obj) or (unique_pixel_count[j] < 10):
                        continue
                    if unique_id_name[j] in configs.surface_list or ('bowl' in unique_id_name[j] and len(unique_id_name[j]) < 10):
                        if (OnTop in env.env.scene.objects_by_name[unique_id_name[i]].states.keys()) and (env.env.scene.objects_by_name[unique_id_name[i]].states[OnTop].get_value(env.env.scene.objects_by_name[unique_id_name[j]])):
                            predicates['OnTop'].append([copy.deepcopy(unique_id[i]), copy.deepcopy(unique_id[j]), int(True)])
                            print("OnTop, ", unique_id_name[i], unique_id_name[j], unique_pixel_count[i], unique_pixel_count[j], predicates['OnTop'][-1])
                        elif OnTop in env.env.scene.objects_by_name[unique_id_name[i]].states.keys():
                            predicates['OnTop'].append([copy.deepcopy(unique_id[i]), copy.deepcopy(unique_id[j]), int(False)])
                            print("OnTop, ", unique_id_name[i], unique_id_name[j], unique_pixel_count[i], unique_pixel_count[j], predicates['OnTop'][-1])

            if (len(predicates['Cooked']) + len(predicates['Cookable']) + len(predicates['OnTop'])) > 5: # keep the data diverse

                tmp_rgb = np.array(env.view_rgb[0])[:, :, ::-1]

                if configs.datagen_vis:
                    cv2.imwrite('{}/data_generation_vis/{}_rgb.png'.format(configs.base_path, frame), tmp_rgb * 255.0)
                    unique = np.unique(env.view_seg[0])
                    for id in unique:
                        cv2.imwrite('{}/data_generation_vis/{}_seg_{}.png'.format(configs.base_path, frame, id), np.array(env.view_seg[0] == id).astype(np.float16) * tmp_rgb * 255.0)
                        cv2.imwrite('{}/data_generation_vis/{}_mask_{}.png'.format(configs.base_path, frame, id), np.array(env.view_seg[0] == id).astype(np.int16) * 255.0)

                env.view_rgb[0] = cv2.resize(env.view_rgb[0], dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
                env.view_seg[0] = cv2.resize(env.view_seg[0], dsize=(128, 128), interpolation=cv2.INTER_NEAREST)

                f.create_group('frame_{0}'.format(frame))
                for key in predicates.keys():
                    f['frame_{0}'.format(frame)].create_dataset(key, data=np.array(predicates[key]))
                f['frame_{0}'.format(frame)].create_dataset('rgb', data=np.array(env.view_rgb[0]).reshape((128, 128, 3)))
                f['frame_{0}'.format(frame)].create_dataset('seg', data=np.array(env.view_seg[0]).reshape((128, 128, 1)))
                f['frame_{0}'.format(frame)].create_dataset('frame_id', data=np.array([env.base_close_to-1]))

                break

    f.close()
    env.close()

if __name__ == "__main__":
    main()
