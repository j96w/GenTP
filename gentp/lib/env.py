import os
import logging
import random
import time
import copy

import pybullet as p
import numpy as np
import igibson
import matplotlib.pyplot as plt
from igibson.envs.behavior_env import BehaviorEnv
from igibson.render.profiler import Profiler
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
from igibson.utils.utils import l2_distance, quatToXYZW, rotate_vector_2d
from igibson.utils.mesh_util import lookat, mat2xyz, ortho, perspective, quat2rotmat, safemat2quat, xyz2mat, xyzw2wxyz
from transforms3d.euler import euler2quat
from igibson.objects.ycb_object import YCBObject
from igibson.objects.articulated_object import ArticulatedObject, URDFObject
from igibson.utils.assets_utils import get_ig_avg_category_specs
from igibson.object_states.cooked import Cooked
import cv2

from igibson.external.pybullet_tools.utils import (
    get_pose,
    set_pose,
    set_quat,
    get_joint_info,
    get_joints,
    get_velocity,
    set_velocity,
    link_from_name,
    matrix_from_quat,
    quat_from_matrix,
    set_joint_position,
    get_joint_position,
    get_joint_positions,
    set_joint_positions,
    set_base_values_with_z,
)

from igibson.object_states.soaked import Soaked
from igibson.object_states.dirty import Stained, Dusty
from igibson.object_states.on_top import OnTop
from igibson.object_states.inside import Inside

from gentp.configs.base_config import Configs

fig_size = [None, None]
view_select = -1
click_x = 0.0
click_y = 0.0

def onclick(event):
    global fig_size
    global view_select
    global click_x
    global click_y

    if event.x < (fig_size[0] / 2):
        view_select = 0
    else:
        view_select = 1

    click_x = event.xdata
    click_y = event.ydata



def create_primitive_shape(mass, shape, dim, color=(0.6, 0, 0, 1), collidable=False, init_xyz=(0, 0, 0.5),
                           init_quat=(0, 0, 0, 1)):
    # shape: p.GEOM_SPHERE or p.GEOM_BOX or p.GEOM_CYLINDER
    # dim: halfExtents (vec3) for box, (radius, length)vec2 for cylinder, (radius) for sphere
    # init_xyz vec3 being initial obj location, init_quat being initial obj orientation
    visual_shape_id = None
    collision_shape_id = -1
    if shape == p.GEOM_BOX:
        visual_shape_id = p.createVisualShape(shapeType=shape, halfExtents=dim, rgbaColor=color)
        if collidable:
            collision_shape_id = p.createCollisionShape(shapeType=shape, halfExtents=dim)
    elif shape == p.GEOM_CYLINDER:
        visual_shape_id = p.createVisualShape(shape, dim[0], [1, 1, 1], dim[1], rgbaColor=color)
        if collidable:
            collision_shape_id = p.createCollisionShape(shape, dim[0], [1, 1, 1], dim[1])
    elif shape == p.GEOM_SPHERE:
        visual_shape_id = p.createVisualShape(shape, radius=dim[0], rgbaColor=color)
        if collidable:
            collision_shape_id = p.createCollisionShape(shape, radius=dim[0])

    sid = p.createMultiBody(baseMass=mass, baseInertialFramePosition=[0, 0, 0],
                            baseCollisionShapeIndex=collision_shape_id,
                            baseVisualShapeIndex=visual_shape_id,
                            basePosition=init_xyz, baseOrientation=init_quat)
    return sid


class IndoorEnv():
    def __init__(self, config_filename, mode='train', viewer='pbgui'):

        self.ig_config_filename = config_filename
        self.configs = Configs()
        self.mode = mode
        self.env = BehaviorEnv(
            config_file=os.path.join(igibson.example_config_path, self.ig_config_filename),
            mode=viewer,
            action_timestep=1 / 30.0,
            physics_timestep=1 / 300.0,
        )

        exist_obj_list = self.env.scene.objects_by_name.keys()
        for item in exist_obj_list:
            if 'towel' in item:
                target_switch_pos = [random.uniform(10, 20), random.uniform(10, 20), random.uniform(10, 20)]
                self.env.scene.objects_by_name[item].set_position(target_switch_pos)

        self.robot = self.env.robots[0]
        self.robot_id = self.robot.robot_ids[0]

        avg_category_spec = get_ig_avg_category_specs()

        if mode == 'train':
            # object for training
            self.food = [
                '{}/igibson/data/ig_dataset/objects/rib/rib_000/rib_000.urdf'.format(self.configs.igibson_path),
                '009_gelatin_box',
                '005_tomato_soup_can']
            self.food_name = ['rib', 'object_0', 'object_1']
            self.food_scale = [0.8, 1.0, 1.0]
        else:
            # object unseen eval set 1
            self.food = ['035_power_drill',
                         '{}/igibson/data/ig_dataset/objects/crab/crab_000/crab_000.urdf'.format(self.configs.igibson_path),
                         '008_pudding_box']
            self.food_name = ['object_0', 'crab', 'object_1']
            self.food_scale = [1.0, 0.7, 1.0]

            # # object unseen eval set 2
            # self.food = ['061_foam_brick',
            #              '{}/igibson/data/ig_dataset/objects/broccoli/28_0/28_0.urdf'.format(self.configs.igibson_path),
            #              '025_mug']
            # self.food_name = ['object_0', 'broccoli', 'object_1']
            # self.food_scale = [1.0, 4.0, 1.0]

            # object unseen eval set 3
            # self.food = ['025_mug',
            #              '{}/igibson/data/ig_dataset/objects/steak/steak_000/steak_000.urdf'.format(self.configs.igibson_path),
            #              '008_pudding_box']
            # self.food_name = ['object_0', 'steak', 'object_1']
            # self.food_scale = [1.0, 0.8, 1.0]


        self.food_id = []

        for idx in range(len(self.food)):
            if 'object' in self.food_name[idx]:
                self.food_id.append(self.env.simulator.import_object(YCBObject(self.food[idx])))
            else:
                for size_id in range(len(avg_category_spec.get(self.food_name[idx])['size'])):
                    avg_category_spec.get(self.food_name[idx])['size'][size_id] *= self.food_scale[idx]
                obj_load = URDFObject(
                    self.food[idx],
                    name=self.food_name[idx],
                    category=self.food_name[idx],
                    avg_obj_dims=avg_category_spec.get(self.food_name[idx]),
                    fit_avg_dim_volume=True,
                    texture_randomization=False,
                    overwrite_inertial=True,
                )
                self.food_id.append(self.env.simulator.import_object(obj_load)[0])
            p.resetBasePositionAndOrientation(self.food_id[-1], (random.uniform(-100, 100), random.uniform(-100, 100), random.uniform(-105, -100)), (0, 0, 0, 1))

        self.bowl = ['{}/igibson/data/ig_dataset/objects/bowl/a1393437aac09108d627bfab5d10d45d/a1393437aac09108d627bfab5d10d45d.urdf'.format(self.configs.igibson_path)]
        self.bowl_name = ['bowl']
        self.bowl_scale = [2.2]
        self.bowl_id = []

        for idx in range(len(self.bowl)):
            item_size = copy.deepcopy(avg_category_spec.get('bowl'))
            for size_id in range(len(item_size['size'])):
                item_size['size'][size_id] *= self.bowl_scale[idx]

            obj_load = URDFObject(
                self.bowl[idx],
                name=self.bowl_name[idx],
                category='bowl',
                avg_obj_dims=item_size,
                fit_avg_dim_volume=True,
                texture_randomization=False,
                overwrite_inertial=True,
            )
            self.bowl_id.append(self.env.simulator.import_object(obj_load)[0])
            p.resetBasePositionAndOrientation(self.bowl_id[-1], (random.uniform(-100, 100), random.uniform(-100, 100), random.uniform(-105, -100)), (0, 0, 0, 1))
            p.changeDynamics(self.bowl_id[-1], -1, mass=1000.0)

        self.utils = ['{}/igibson/data/ig_dataset/objects/frying_pan/36_0/36_0.urdf'.format(self.configs.igibson_path)]
        self.utils_name = ['frying_pan']
        self.utils_scale = [1.5]
        self.utils_id = []

        for idx in range(len(self.utils)):
            for size_id in range(len(avg_category_spec.get(self.utils_name[idx])['size'])):
                avg_category_spec.get(self.utils_name[idx])['size'][size_id] *= self.utils_scale[idx]
            obj_load = URDFObject(
                self.utils[idx],
                name=self.utils_name[idx],
                category=self.utils_name[idx],
                avg_obj_dims=avg_category_spec.get(self.utils_name[idx]),
                fit_avg_dim_volume=True,
                texture_randomization=False,
                overwrite_inertial=True,
            )
            self.utils_id.append(self.env.simulator.import_object(obj_load)[0])
            p.resetBasePositionAndOrientation(self.utils_id[-1], (random.uniform(-100, 100), random.uniform(-100, 100), random.uniform(-105, -100)), (0, 0, 0, 1))
            p.changeDynamics(self.utils_id[-1], -1, mass=0.1)

        self.motion_planner = MotionPlanningWrapper(self.env, fine_motion_plan=False, collision_obj=[])

        self.graspable_obj_list = self.food_name + self.utils_name

        self.graspable_obj_id = [self.food_id[0], self.food_id[1], self.food_id[2], self.env.scene.objects_by_name['frying_pan'].body_ids[0]]
        self.placeable_obj_id = [self.env.scene.objects_by_name['countertop_76'].body_ids[0], self.env.scene.objects_by_name['burner_77'].body_ids[0],
                                 self.env.scene.objects_by_name['frying_pan'].body_ids[0], self.env.scene.objects_by_name['bowl'].body_ids[0]]
        self.pan_placeable_obj_id = [self.env.scene.objects_by_name['burner_77'].body_ids[0]]

        self.pourable_obj_id = [self.env.scene.objects_by_name['countertop_76'].body_ids[0], self.env.scene.objects_by_name['burner_77'].body_ids[0], self.env.scene.objects_by_name['bowl'].body_ids[0]]

        self.surface = ['countertop_76', 'burner_77']

        self.target_obj_list = self.food_name + self.bowl_name + self.utils_name + self.surface
        self.target_obj_id_list = [self.food_id[0], self.food_id[1], self.food_id[2],
                                   self.env.scene.objects_by_name['bowl'].body_ids[0], self.env.scene.objects_by_name['frying_pan'].body_ids[0],
                                   self.env.scene.objects_by_name['countertop_76'].body_ids[0], self.env.scene.objects_by_name['burner_77'].body_ids[0]]

        self.env.task.initial_state = self.env.task.save_scene()

        self.in_hand = -1
        self.pan_cons = -1
        self.grasp_pose = None
        self.previous_traj = None
        self.current_state = None

        self.base_close_to = 0
        self.view_rgb = []
        self.view_pcd = []
        self.view_seg = []

        self.high_rgb = []

        self.food_on_pan = []
        self.food_on_pan_grasp_pose = {}

        self.reset_buffer = {}
        self.random_coin = 0

        self.first = True
        self.towel_init_pos, self.towel_init_ori = None, None
        self.update_pcd = True

        for j in get_joints(self.robot_id):
            info = get_joint_info(self.robot_id, j)
            if info.jointName.decode() == 'torso_lift_joint':
                self.lift_joint = j

    def reset(self, layout=0):

        if self.pan_cons != -1:
            p.removeConstraint(self.pan_cons)
            self.pan_cons = -1
        self.update_pcd = True

        self.env.reset()
        self.in_hand = -1
        self.grasp_pose = None
        self.previous_traj = None
        self.current_state = None

        self.base_close_to = 0
        self.view_rgb = []
        self.view_pcd = []
        self.view_seg = []

        self.food_on_pan = []
        self.food_on_pan_grasp_pose = {}

        self.base_reset(True)

        utils_list = self.utils_name.copy()
        random_sample_list = self.utils_name.copy() + self.bowl_name.copy() + self.food_name.copy()  # + self.towel_list

        if layout == 0:
            # # layout 0
            pan_pos = copy.deepcopy(list(self.env.scene.objects_by_name[self.surface[1]].get_position()))
            pan_pos[0] += 0.15
            pan_pos[1] += -0.1
            pan_pos[2] += 0.05

            bowl_pos = copy.deepcopy(list(self.env.scene.objects_by_name[self.surface[0]].get_position()))
            bowl_pos[0] += random.uniform(0.4, 0.5)
            bowl_pos[1] += random.uniform(-0.15, -0.1)
            bowl_pos[2] += 0.1
        elif layout == 1:
            # layout 1
            pos = copy.deepcopy(list(self.env.scene.objects_by_name[self.surface[1]].get_position()))
            pos[0] += 0.7
            self.env.scene.objects_by_name[self.surface[1]].set_position(pos, force=True)

            pan_pos = copy.deepcopy(list(self.env.scene.objects_by_name[self.surface[1]].get_position()))
            pan_pos[0] -= 0.15
            pan_pos[1] += -0.1
            pan_pos[2] += 0.05

            bowl_pos = copy.deepcopy(list(self.env.scene.objects_by_name[self.surface[0]].get_position()))
            bowl_pos[0] += random.uniform(-0.2, 0.1)
            bowl_pos[1] += random.uniform(-0.25, -0.15)
            bowl_pos[2] += 0.1
        elif layout == 2:
            # # layout 2
            pos = copy.deepcopy(list(self.env.scene.objects_by_name[self.surface[1]].get_position()))
            pos[0] += 0.35
            pos[1] += 0.3
            self.env.scene.objects_by_name[self.surface[1]].set_position(pos, force=True)

            pan_pos = copy.deepcopy(list(self.env.scene.objects_by_name[self.surface[1]].get_position()))
            pan_pos[0] += 0.15
            pan_pos[1] += -0.1
            pan_pos[2] += 0.05

            bowl_pos = copy.deepcopy(list(self.env.scene.objects_by_name[self.surface[0]].get_position()))
            bowl_pos[0] += random.uniform(0.0, 0.2)
            bowl_pos[1] += random.uniform(-0.3, -0.2)
            bowl_pos[2] += 0.1


        for i in range(len(random_sample_list)):
            if random_sample_list[i] in utils_list:

                self.env.scene.objects_by_name[random_sample_list[i]].set_position(pan_pos)
                self.env.scene.objects_by_name[random_sample_list[i]].rotate_by(0.0, 0.0, np.pi)
            elif 'bowl' in random_sample_list[i]:

                self.env.scene.objects_by_name[random_sample_list[i]].set_position(bowl_pos)
                self.env.scene.objects_by_name[random_sample_list[i]].rotate_by(0.0, 0.0, random.uniform(-1.0, 1.0))
            else:
                surface_select = self.surface[0]
                pos = copy.deepcopy(list(self.env.scene.objects_by_name[surface_select].get_position()))
                pos[0] += random.uniform(-0.2, 0.6)
                pos[1] += random.uniform(-0.2, 0.1)
                pos[2] += random.uniform(0.2, 0.3)

                if self.mode == 'eval':
                    if random_sample_list[i] == self.food_name[1]:
                        pos = copy.deepcopy(list(self.env.scene.objects_by_name['bowl'].get_position()))
                        pos[2] += random.uniform(0.2, 0.3)
                    else:
                        pos = copy.deepcopy(list(self.env.scene.objects_by_name['bowl'].get_position()))
                        pos[0] += random.uniform(-0.2, 0.0)
                        pos[1] += random.uniform(0.25, 0.35)

                self.env.scene.objects_by_name[random_sample_list[i]].set_position(pos)
                if Cooked not in self.env.scene.objects_by_name[random_sample_list[i]].states.keys():
                    self.env.scene.objects_by_name[random_sample_list[i]].rotate_by(0.0, 0.0, random.uniform(-1.0, 1.0))
                else:
                    self.env.scene.objects_by_name[random_sample_list[i]].rotate_by(0.0, 0.0, random.uniform(-1.0, 1.0))
                if Cooked in self.env.scene.objects_by_name[random_sample_list[i]].states.keys():
                    self.env.scene.objects_by_name[random_sample_list[i]].states[Cooked].set_value(False)


        self.random_step(15, hold=(self.in_hand != -1), real_step=True)

        self.bowl_initial_pos, self.bowl_initial_ori = self.env.scene.objects_by_name[self.bowl_name[0]].get_position_orientation()
        self.utils_initial_z = self.env.scene.objects_by_name[self.utils_name[0]].get_position()[2]

        self.base_reset()

        self.current_state = self.random_step(1, hold=(self.in_hand != -1), real_step=True)

    def get_syb_state(self):

        for i in range(3):
            if self.env.scene.objects_by_name[self.target_obj_list[i]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[4]]) \
                    and Cooked in self.env.scene.objects_by_name[self.target_obj_list[i]].states.keys() \
                    and self.env.scene.objects_by_name[self.target_obj_list[i]].states[Cooked].get_value() == False:
                self.env.scene.objects_by_name[self.target_obj_list[i]].states[Cooked].set_value(True)
                self.random_step(3, hold=(self.in_hand != -1), real_step=True)
                self.base_reset()

        syb_state = [self.env.scene.objects_by_name[self.target_obj_list[0]].states[Cooked].get_value() if Cooked in self.env.scene.objects_by_name[self.target_obj_list[0]].states.keys() else False,
                     self.env.scene.objects_by_name[self.target_obj_list[1]].states[Cooked].get_value() if Cooked in self.env.scene.objects_by_name[self.target_obj_list[1]].states.keys() else False,
                     self.env.scene.objects_by_name[self.target_obj_list[2]].states[Cooked].get_value() if Cooked in self.env.scene.objects_by_name[self.target_obj_list[2]].states.keys() else False,
                     self.in_hand != -1,
                     self.in_hand == self.utils_id[0],
                     self.env.scene.objects_by_name[self.target_obj_list[0]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[3]]),
                     self.env.scene.objects_by_name[self.target_obj_list[0]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[4]]),
                     self.env.scene.objects_by_name[self.target_obj_list[0]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[5]]),
                     self.env.scene.objects_by_name[self.target_obj_list[0]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[6]]),
                     self.env.scene.objects_by_name[self.target_obj_list[1]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[3]]),
                     self.env.scene.objects_by_name[self.target_obj_list[1]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[4]]),
                     self.env.scene.objects_by_name[self.target_obj_list[1]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[5]]),
                     self.env.scene.objects_by_name[self.target_obj_list[1]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[6]]),
                     self.env.scene.objects_by_name[self.target_obj_list[2]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[3]]),
                     self.env.scene.objects_by_name[self.target_obj_list[2]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[4]]),
                     self.env.scene.objects_by_name[self.target_obj_list[2]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[5]]),
                     self.env.scene.objects_by_name[self.target_obj_list[2]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[6]]),
                     self.env.scene.objects_by_name[self.target_obj_list[4]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[5]]),
                     self.env.scene.objects_by_name[self.target_obj_list[4]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[6]])]

        if syb_state[5] or syb_state[6]:
            syb_state[7], syb_state[8] = False, False
        if syb_state[9] or syb_state[10]:
            syb_state[11], syb_state[12] = False, False
        if syb_state[13] or syb_state[14]:
            syb_state[15], syb_state[16] = False, False

        food_on_pan_syb_idx = [6, 10, 14]
        if (self.in_hand == self.utils_id[0]) and (len(self.food_on_pan) != 0):
            for food_idx in range(len(self.food_id)):
                if self.food_id[food_idx] in self.food_on_pan:
                    syb_state[food_on_pan_syb_idx[food_idx]] = True

        if self.in_hand != self.utils_id[0]:
            self.food_on_pan = []
            for food in self.food_id:
                if self.env.scene.objects_by_id[food].states[OnTop].get_value(self.env.scene.objects_by_name[self.utils_name[0]]):
                    self.food_on_pan.append(copy.deepcopy(food))
        # print(self.food_on_pan)

        syb_string = np.array2string(np.array(syb_state).astype(np.int16))
        if (not syb_string in self.reset_buffer.keys()) and (sum(syb_state[-14:-2]) == 3) and (sum(syb_state[:3]) == 0): #and (sum(syb_state[2:4]) == 2)
            self.reset_buffer[syb_string] = {}
            self.reset_buffer[syb_string]['p_state'] = p.saveState()
            self.reset_buffer[syb_string]['in_hand'] = copy.deepcopy(self.in_hand)
            self.reset_buffer[syb_string]['grasp_pose'] = copy.deepcopy(self.grasp_pose)
            self.reset_buffer[syb_string]['food_on_pan'] = copy.deepcopy(self.food_on_pan)
            self.reset_buffer[syb_string]['food_on_pan_grasp_pose'] = copy.deepcopy(self.food_on_pan_grasp_pose)
            self.reset_buffer[syb_string]['previous_traj'] = copy.deepcopy(self.previous_traj)
            self.reset_buffer[syb_string]['current_state'] = copy.deepcopy(self.current_state)
            self.reset_buffer[syb_string]['base_close_to'] = copy.deepcopy(self.base_close_to)
            self.reset_buffer[syb_string]['view_rgb'] = copy.deepcopy(self.view_rgb)
            self.reset_buffer[syb_string]['view_pcd'] = copy.deepcopy(self.view_pcd)
            self.reset_buffer[syb_string]['view_seg'] = copy.deepcopy(self.view_seg)
        elif (syb_string in self.reset_buffer.keys()) and (sum(syb_state[-14:-2]) == 3) and (sum(syb_state[:3]) == 0): #and (sum(syb_state[2:4]) == 2)
            p.removeState(self.reset_buffer[syb_string]['p_state'])
            self.reset_buffer[syb_string] = {}
            self.reset_buffer[syb_string]['p_state'] = p.saveState()
            self.reset_buffer[syb_string]['in_hand'] = copy.deepcopy(self.in_hand)
            self.reset_buffer[syb_string]['grasp_pose'] = copy.deepcopy(self.grasp_pose)
            self.reset_buffer[syb_string]['food_on_pan'] = copy.deepcopy(self.food_on_pan)
            self.reset_buffer[syb_string]['food_on_pan_grasp_pose'] = copy.deepcopy(self.food_on_pan_grasp_pose)
            self.reset_buffer[syb_string]['previous_traj'] = copy.deepcopy(self.previous_traj)
            self.reset_buffer[syb_string]['current_state'] = copy.deepcopy(self.current_state)
            self.reset_buffer[syb_string]['base_close_to'] = copy.deepcopy(self.base_close_to)
            self.reset_buffer[syb_string]['view_rgb'] = copy.deepcopy(self.view_rgb)
            self.reset_buffer[syb_string]['view_pcd'] = copy.deepcopy(self.view_pcd)
            self.reset_buffer[syb_string]['view_seg'] = copy.deepcopy(self.view_seg)

        return syb_state

    def process_pc_to_world(self, td_image, u, v):
        # Pose of the camera of the simulated robot in world frame
        eye_pos, eye_orn = self.robot.parts["eyes"].get_position(), self.robot.parts["eyes"].get_orientation()
        camera_in_wf = quat2rotmat(xyzw2wxyz(eye_orn))
        camera_in_wf[:3, 3] = eye_pos

        # Transforming coordinates of points from opengl frame to camera frame
        camera_in_openglf = quat2rotmat(euler2quat(np.pi / 2.0, 0, -np.pi / 2.0))

        # Pose of the simulated robot in world frame
        robot_pos, robot_orn = self.robot.get_position(), self.robot.get_orientation()
        robot_in_wf = quat2rotmat(xyzw2wxyz(robot_orn))
        robot_in_wf[:3, 3] = robot_pos

        # Pose of the camera in robot frame
        cam_in_robot_frame = np.dot(np.linalg.inv(robot_in_wf), camera_in_wf)

        x, y, _ = td_image.shape
        ones = np.ones((x, y, 1))
        td_image = np.concatenate((td_image, ones), axis=2)

        point_in_openglf = td_image[u, v]
        point_in_cf = np.dot(camera_in_openglf, point_in_openglf)
        point_in_rf = np.dot(cam_in_robot_frame, point_in_cf)
        point_in_wf = np.dot(robot_in_wf, point_in_rf)

        return point_in_wf

    def update_view(self, reset=False):
        states = self.random_step(1, hold = (self.in_hand != -1), real_step=True)

        if reset:
            self.view_rgb = []
            self.view_pcd = []
            self.view_seg = []
            self.high_rgb = []
            self.high_rgb.append(states['rgb'].copy())

            states['rgb'] = cv2.resize(states['rgb'], dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
            states['pc'] = cv2.resize(states['pc'], dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
            states['ins_seg'] = cv2.resize(states['ins_seg'], dsize=(128, 128), interpolation=cv2.INTER_NEAREST)

            self.view_rgb.append(states['rgb'].copy())
            self.view_pcd.append(states['pc'].copy())
            self.view_seg.append(states['ins_seg'].copy())
        else:
            self.high_rgb[0] = states['rgb'].copy()

            states['rgb'] = cv2.resize(states['rgb'], dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
            states['pc'] = cv2.resize(states['pc'], dsize=(128, 128), interpolation=cv2.INTER_NEAREST)
            states['ins_seg'] = cv2.resize(states['ins_seg'], dsize=(128, 128), interpolation=cv2.INTER_NEAREST)

            self.view_rgb[0] = states['rgb'].copy()
            if self.update_pcd:
                self.view_pcd[0] = states['pc'].copy()
            self.view_seg[0] = states['ins_seg'].copy()

    def show_view(self):
        fig, axarr = plt.subplots(1, 2)
        axarr[0].imshow(self.view_rgb[0], interpolation='none')
        axarr[1].imshow(self.view_seg[0], interpolation='none')

        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        global fig_size
        fig_size = fig.get_size_inches() * fig.dpi

        plt.show()


    def arm_move_to(self, target_pos, skip_motion=True):
        arm_joints = self.motion_planner.get_arm_joint_positions(target_pos)
        if arm_joints == None:
            return False

        if not skip_motion:
            arm_path = self.motion_planner.plan_arm_motion(arm_joints)
            if arm_path == None:
                return False
            self.previous_traj = arm_path.copy()
            self.motion_planner.dry_run_arm_plan(arm_path, self.in_hand, self.grasp_pose, self.in_hand == self.utils_id[0], self.food_on_pan, self.food_on_pan_grasp_pose, True)
        else:
            start_conf = get_joint_positions(self.motion_planner.robot_id, self.motion_planner.arm_joint_ids)
            self.previous_traj = list([start_conf]).copy()

            self.motion_planner.dry_run_arm_plan(arm_joints, self.in_hand, self.grasp_pose, self.in_hand == self.utils_id[0], self.food_on_pan, self.food_on_pan_grasp_pose)
        return True

    def arm_move_to_pour(self, target_pos, skip_motion=True):
        arm_joints = self.motion_planner.get_arm_joint_positions(target_pos, pour=True)
        if arm_joints == None:
            return False

        if not skip_motion:
            arm_path = self.motion_planner.plan_arm_motion(arm_joints)
            if arm_path == None:
                return False
            self.previous_traj_pour = arm_path.copy()
            self.motion_planner.dry_run_arm_plan(arm_path, self.in_hand, self.grasp_pose, self.in_hand == self.utils_id[0], self.food_on_pan, self.food_on_pan_grasp_pose, True)

            self.pour_joints = arm_joints
        else:
            start_conf = get_joint_positions(self.motion_planner.robot_id, self.motion_planner.arm_joint_ids)
            self.previous_traj_pour = list([start_conf]).copy()
            self.pour_joints = arm_joints

            self.motion_planner.dry_run_arm_plan(arm_joints, self.in_hand, self.grasp_pose, self.in_hand == self.utils_id[0], self.food_on_pan, self.food_on_pan_grasp_pose)
        return True

    def arm_return(self):
        if self.previous_traj == None:
            return

        arm_path = self.previous_traj.copy()[::-1]

        self.motion_planner.dry_run_arm_plan(arm_path, self.in_hand, self.grasp_pose, self.in_hand == self.utils_id[0], self.food_on_pan, self.food_on_pan_grasp_pose, True)


    def arm_return_pour(self):
        if self.previous_traj == None:
            return

        arm_path = self.previous_traj_pour.copy()[::-1]

        self.motion_planner.dry_run_arm_plan(arm_path, self.in_hand, self.grasp_pose, self.in_hand == self.utils_id[0], self.food_on_pan, self.food_on_pan_grasp_pose, True)



    def base_move_to(self, target_pos):
        self.motion_planner.dry_run_base_plan(target_pos, self.in_hand, self.grasp_pose)

    def robot_lift(self, target_height):
        current_pose = list(get_joint_positions(self.robot_id, self.motion_planner.arm_joint_ids))
        for target in np.arange(current_pose[0], target_height, 0.01):
            if self.in_hand != -1:
                gripper_pos = p.getLinkState(self.robot_id, self.robot.end_effector_part_index())[0]
                gripper_orn = p.getLinkState(self.robot_id, self.robot.end_effector_part_index())[1]
                object_pose = p.multiplyTransforms(gripper_pos, gripper_orn, self.grasp_pose[0], self.grasp_pose[1])
                set_pose(self.in_hand, object_pose)

            current_pose[0] = target
            set_joint_positions(self.robot_id, self.motion_planner.arm_joint_ids, current_pose)
            self.env.simulator.sync()

    def check_grasping(self, gripper):
        initial_in_hand = copy.deepcopy(self.in_hand)
        if self.in_hand == -1 and gripper < 0:
            gripper_pos = self.robot.get_end_effector_position()
            minimal = 0.15
            for item in self.graspable_obj_list:
                item_pos = list(self.env.scene.objects_by_name[item].get_position())

                if item == 'frying_pan':
                    item_pos[1] -= 0.1
                    item_pos[2] += 0.05
                dist = l2_distance(gripper_pos, item_pos)

                if item == 'frying_pan' and dist < 0.2:
                    dist = 0.0

                if dist < minimal:
                    minimal = dist
                    self.in_hand = self.env.scene.objects_by_name[item].get_body_id()
                    obj_pos, obj_orn = p.getBasePositionAndOrientation(self.in_hand)
                    gripper_pos = p.getLinkState(self.robot_id, self.robot.end_effector_part_index())[0]
                    gripper_orn = p.getLinkState(self.robot_id, self.robot.end_effector_part_index())[1]
                    self.grasp_pose = p.multiplyTransforms(*p.invertTransform(gripper_pos, gripper_orn), obj_pos, obj_orn)
                    self.grasp_pose = list(self.grasp_pose)

                    delta = np.array([0.0, 0.0, 0.0])

                    if self.in_hand == self.utils_id[0]:
                        delta = np.array([0.1 - self.grasp_pose[0][0], 0.0, 0.1 - self.grasp_pose[0][2]])
                        self.grasp_pose[0] = (self.grasp_pose[0][0] + delta[0], self.grasp_pose[0][1], self.grasp_pose[0][2] + delta[2])

                    if self.in_hand == self.utils_id[0] and len(self.food_on_pan) != 0:
                        self.food_on_pan_grasp_pose = {}
                        for food in self.food_on_pan:
                            obj_pos, obj_orn = p.getBasePositionAndOrientation(food)
                            self.food_on_pan_grasp_pose[food] = list(p.multiplyTransforms(*p.invertTransform(gripper_pos, gripper_orn), obj_pos, obj_orn))
                            self.food_on_pan_grasp_pose[food][0] = (self.food_on_pan_grasp_pose[food][0][0] + delta[0], self.food_on_pan_grasp_pose[food][0][1], self.food_on_pan_grasp_pose[food][0][2] + delta[2])

            if self.in_hand == self.utils_id[0] and self.pan_cons == -1:
                self.pan_cons = p.createConstraint(
                    parentBodyUniqueId=self.robot_id,
                    parentLinkIndex=self.robot.end_effector_part_index(),
                    childBodyUniqueId=self.in_hand,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=(0, 0, 0),
                    parentFramePosition=self.grasp_pose[0],
                    childFramePosition=[0, 0, 0],
                    parentFrameOrientation=self.grasp_pose[1],
                    childFrameOrientation=(0, 0, 0, 1),
                )
                p.changeConstraint(self.pan_cons, maxForce=100000.0)

        elif self.in_hand != -1 and gripper >= 0:
            if self.in_hand == self.utils_id[0]:
                p.removeConstraint(self.pan_cons)
                self.pan_cons = -1

            self.in_hand = -1
            self.grasp_pose = None

    def update_in_hand(self):
        if self.in_hand != -1:
            gripper_pos = p.getLinkState(self.robot_id, self.robot.end_effector_part_index())[0]
            gripper_orn = p.getLinkState(self.robot_id, self.robot.end_effector_part_index())[1]
            object_pose = p.multiplyTransforms(gripper_pos, gripper_orn, self.grasp_pose[0], self.grasp_pose[1])
            set_pose(self.in_hand, object_pose)
            self.env.simulator.sync()

    def arm_pick_click(self):
        global view_select
        global click_x
        global click_y

        pos = self.process_pc_to_world(self.view_pcd[view_select], int(click_y), int(click_x))[:3]
        pos[2] += 0.06

        succ = self.arm_move_to(pos, skip_motion=False)
        # print(succ)
        if not succ:
            self.base_reset()
            return
        self.random_step(1, hold=True, real_step=False)

        self.arm_return()

        self.base_reset()

    def arm_place_click(self):
        global view_select
        global click_x
        global click_y

        pos = self.process_pc_to_world(self.view_pcd[view_select], int(click_y), int(click_x))[:3]
        if self.in_hand != -1:
            obj_pos, _ = p.getBasePositionAndOrientation(self.in_hand)
            gripper_pos = p.getLinkState(self.robot_id, self.robot.end_effector_part_index())[0]
            pos[0] += gripper_pos[0] - obj_pos[0]
            pos[1] += gripper_pos[1] - obj_pos[1]
            pos[2] += gripper_pos[2] - obj_pos[2]
            if self.in_hand == self.utils_id[0]:
                pos[2] += 0.12
            else:
                pos[2] += 0.07
        else:
            pos[2] += 0.07

        succ = self.arm_move_to(pos, skip_motion=False)
        # print(succ)
        if not succ:
            self.base_reset()
            return
        if self.in_hand == self.utils_id[0]:
            self.fix_pan_z()

        self.random_step(5, hold=False, real_step=True)
        self.arm_return()
        self.base_reset()

    def arm_pour_click(self):
        global view_select
        global click_x
        global click_y

        if self.in_hand != self.utils_id[0] or len(self.food_on_pan) == 0:
            return

        pos = self.process_pc_to_world(self.view_pcd[view_select], int(click_y), int(click_x))[:3]
        obj_pos, _ = p.getBasePositionAndOrientation(self.in_hand)
        gripper_pos = p.getLinkState(self.robot_id, self.robot.end_effector_part_index())[0]
        pos[0] += gripper_pos[0] - obj_pos[0]
        pos[1] += gripper_pos[1] - obj_pos[1]
        pos[2] += 0.45
        pos[0] -= 0.15


        succ = self.arm_move_to_pour(pos, skip_motion=False)
        if not succ:
            self.base_reset()
            return

        for _ in range(10):
            self.base_reset(pour=True)
            self.random_step(1, hold=(self.in_hand != -1), real_step=True)

        if len(self.food_on_pan) != 0:
            self.food_on_pan = []
            self.food_on_pan_grasp_pose = {}

        self.arm_return_pour()
        self.base_reset()


    def arm_pick(self, view_select, click_x, click_y, skip_motion=True):

        pos = self.process_pc_to_world(self.view_pcd[view_select], click_x, click_y)[:3]
        pos[2] += 0.06

        succ = self.arm_move_to(pos, skip_motion=skip_motion)
        # print(succ)
        if not succ:
            self.base_reset()
            return
        self.random_step(1, hold=True, real_step=False)

        self.arm_return()

        self.base_reset()


    def arm_place(self, view_select, click_x, click_y, skip_motion=True):

        pos = self.process_pc_to_world(self.view_pcd[view_select], click_x, click_y)[:3]
        if self.in_hand != -1:
            obj_pos, _ = p.getBasePositionAndOrientation(self.in_hand)
            gripper_pos = p.getLinkState(self.robot_id, self.robot.end_effector_part_index())[0]
            pos[0] += gripper_pos[0] - obj_pos[0]
            pos[1] += gripper_pos[1] - obj_pos[1]
            pos[2] += gripper_pos[2] - obj_pos[2]
            if self.in_hand == self.utils_id[0]:
                pos[2] += 0.12
            else:
                pos[2] += 0.07
        else:
            pos[2] += 0.07

        succ = self.arm_move_to(pos, skip_motion=skip_motion)
        if not succ:
            self.base_reset()
            return
        if self.in_hand == self.utils_id[0]:
            self.fix_pan_z()

        self.random_step(5, hold=False, real_step=True)
        self.arm_return()
        self.base_reset()


    def arm_pour(self, view_select, click_x, click_y, skip_motion=True):

        if self.in_hand != self.utils_id[0] or len(self.food_on_pan) == 0:
            return

        pos = self.process_pc_to_world(self.view_pcd[view_select], click_x, click_y)[:3]
        obj_pos, _ = p.getBasePositionAndOrientation(self.in_hand)
        gripper_pos = p.getLinkState(self.robot_id, self.robot.end_effector_part_index())[0]
        pos[0] += gripper_pos[0] - obj_pos[0]
        pos[1] += gripper_pos[1] - obj_pos[1]
        pos[2] += 0.45
        pos[0] -= 0.15

        succ = self.arm_move_to_pour(pos, skip_motion=skip_motion)
        if not succ:
            self.base_reset()
            return

        for _ in range(15):
            self.base_reset(pour=True)
            self.random_step(1, hold=(self.in_hand != -1), real_step=True)

        if len(self.food_on_pan) != 0:
            self.food_on_pan = []
            self.food_on_pan_grasp_pose = {}

        self.arm_return_pour()
        self.base_reset()


    def fix_pan_z(self):
        utils_pos = self.env.scene.objects_by_name[self.utils_name[0]].get_position()
        utils_pos = list(utils_pos)
        utils_pos[2] = self.utils_initial_z
        set_pose(self.utils_id[0], (utils_pos, (0.0, 0.0, 1.0, 0.0)))
        set_velocity(self.utils_id[0], (0, 0, 0), (0, 0, 0))


    def base_reset(self, reset=False, pour=False):
        set_base_values_with_z(self.robot_id, (-5.3, -4.5, np.pi / 2), z=self.env.initial_pos_z_offset)
        if pour:
            set_joint_positions(self.robot_id, self.motion_planner.arm_joint_ids, self.pour_joints)
        set_velocity(self.robot_id, (0, 0, 0), (0, 0, 0))

        self.base_close_to = 0

        if not pour:
            set_quat(self.utils_id[0], (0.0, 0.0, 1.0, 0.0))
            set_velocity(self.utils_id[0], (0, 0, 0), (0, 0, 0))

        if self.in_hand != -1 and not pour:
            gripper_pos = p.getLinkState(self.robot_id, self.robot.end_effector_part_index())[0]
            gripper_orn = p.getLinkState(self.robot_id, self.robot.end_effector_part_index())[1]
            object_pose = p.multiplyTransforms(gripper_pos, gripper_orn, self.grasp_pose[0], self.grasp_pose[1])
            set_pose(self.in_hand, object_pose)
            set_velocity(self.in_hand, (0, 0, 0), (0, 0, 0))

        self.update_view(reset)


    def random_step(self, steps, hold=False, real_step=False):
        for i in range(steps):
            action = [0.0 for _ in range(11)]
            if hold:
                gripper = -0.5
            else:
                gripper = 0.5

            self.check_grasping(gripper)
            if real_step:
                self.current_state, reward, done, info = self.env.step(action)
            else:
                self.current_state = None

        return self.current_state

    def close(self):
        self.env.close()