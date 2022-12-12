import os
import logging
import random
import time
import copy
import cv2
import pybullet as p
import numpy as np
import igibson
from igibson.objects.ycb_object import YCBObject
import matplotlib.pyplot as plt
from igibson.envs.behavior_env import BehaviorEnv
from igibson.utils.motion_planning_wrapper import MotionPlanningWrapper
from igibson.utils.utils import l2_distance, quatToXYZW, rotate_vector_2d
from igibson.utils.mesh_util import lookat, mat2xyz, ortho, perspective, quat2rotmat, safemat2quat, xyz2mat, xyzw2wxyz
from transforms3d.euler import euler2quat
from igibson.objects.articulated_object import URDFObject
from igibson.utils.assets_utils import get_ig_avg_category_specs
from igibson.object_states.cooked import Cooked
from igibson.object_states.toggle import ToggledOn

from igibson.external.pybullet_tools.utils import (
    get_pose,
    set_pose,
    get_joint_info,
    get_joints,
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
    def __init__(self, config_filename, viewer = 'pbgui', instance_id=0):

        self.config_filename = config_filename
        self.env = BehaviorEnv(
            config_file=os.path.join(igibson.example_config_path, self.config_filename),
            mode=viewer,
            action_timestep=1 / 30.0,
            physics_timestep=1 / 300.0,
            instance_id=instance_id
        )
        self.configs = Configs()

        exist_obj_list = self.env.scene.objects_by_name.keys()
        for item in exist_obj_list:
            if 'towel' in item:
                target_switch_pos = [random.uniform(10, 20), random.uniform(10, 20), random.uniform(10, 20)]
                self.env.scene.objects_by_name[item].set_position(target_switch_pos)


        avg_category_spec = get_ig_avg_category_specs()

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
            p.changeDynamics(self.utils_id[-1], -1, mass=1000.0)

        self.bowl = [
            '{}/igibson/data/ig_dataset/objects/bowl/5aad71b5e6cb3967674684c50f1db165/5aad71b5e6cb3967674684c50f1db165.urdf'.format(self.configs.igibson_path),
            '{}/igibson/data/ig_dataset/objects/bowl/7d7bdea515818eb844638317e9e4ff18/7d7bdea515818eb844638317e9e4ff18.urdf'.format(self.configs.igibson_path),
            '{}/igibson/data/ig_dataset/objects/bowl/8b90aa9f4418c75452dd9cc5bac31c96/8b90aa9f4418c75452dd9cc5bac31c96.urdf'.format(self.configs.igibson_path),
            '{}/igibson/data/ig_dataset/objects/bowl/68_0/68_0.urdf'.format(self.configs.igibson_path),
            '{}/igibson/data/ig_dataset/objects/bowl/68_1/68_1.urdf'.format(self.configs.igibson_path),
            '{}/igibson/data/ig_dataset/objects/bowl/68_2/68_2.urdf'.format(self.configs.igibson_path),
            '{}/igibson/data/ig_dataset/objects/bowl/68_3/68_3.urdf'.format(self.configs.igibson_path),
            '{}/igibson/data/ig_dataset/objects/bowl/80_0/80_0.urdf'.format(self.configs.igibson_path),
            '{}/igibson/data/ig_dataset/objects/bowl/56803af65db53307467ca2ad6571afff/56803af65db53307467ca2ad6571afff.urdf'.format(self.configs.igibson_path),
            '{}/igibson/data/ig_dataset/objects/bowl/6494761a8a0461777cba8364368aa1d/6494761a8a0461777cba8364368aa1d.urdf'.format(self.configs.igibson_path)
            ]

        self.bowl_name = ['bowl_1',
                          'bowl_2',
                          'bowl_3',
                          'bowl_4',
                          'bowl_5',
                          'bowl_6',
                          'bowl_7',
                          'bowl_8',
                          'bowl_9',
                          'bowl_10',
                          ]

        self.bowl_scale = [1.8,
                           1.8,
                           1.8,
                           1.8,
                           1.8,
                           1.8,
                           1.8,
                           1.8,
                           1.8,
                           1.8,
                           ]

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


        self.ycb = ["002_master_chef_can", "003_cracker_box", "004_sugar_box", "005_tomato_soup_can", "006_mustard_bottle", "007_tuna_fish_can",
                    "009_gelatin_box", "019_pitcher_base", "021_bleach_cleanser", "024_bowl", "036_wood_block", "040_large_marker", "061_foam_brick"]
        self.ycb_name = ['object_{0}'.format(item) for item in range(len(self.ycb))]
        self.ycb_id = []

        for ycb in self.ycb:
            self.ycb_id.append(self.env.simulator.import_object(YCBObject(ycb)))
            p.resetBasePositionAndOrientation(self.ycb_id[-1], (random.uniform(-100, 100), random.uniform(-100, 100), random.uniform(-105, -100)), (0, 0, 0, 1))


        self.food = ['{}/igibson/data/ig_dataset/objects/banana/09_0/09_0.urdf'.format(self.configs.igibson_path),
                     '{}/igibson/data/ig_dataset/objects/carrot/47_0/47_0.urdf'.format(self.configs.igibson_path),
                     '{}/igibson/data/ig_dataset/objects/chicken/chicken_000/chicken_000.urdf'.format(self.configs.igibson_path),
                     '{}/igibson/data/ig_dataset/objects/strawberry/36_1/36_1.urdf'.format(self.configs.igibson_path),
                     '{}/igibson/data/ig_dataset/objects/spinach/43_0/43_0.urdf'.format(self.configs.igibson_path),
                     '{}/igibson/data/ig_dataset/objects/sausage/sausage_000/sausage_000.urdf'.format(self.configs.igibson_path),
                     '{}/igibson/data/ig_dataset/objects/pomelo/12_0/12_0.urdf'.format(self.configs.igibson_path),
                     '{}/igibson/data/ig_dataset/objects/butter/butter_000/butter_000.urdf'.format(self.configs.igibson_path),
                     '{}/igibson/data/ig_dataset/objects/chocolate_box/chocolate_box_000/chocolate_box_000.urdf'.format(self.configs.igibson_path),
                     '{}/igibson/data/ig_dataset/objects/pretzel/pretzel_000/pretzel_000.urdf'.format(self.configs.igibson_path),
                     '{}/igibson/data/ig_dataset/objects/rib/rib_000/rib_000.urdf'.format(self.configs.igibson_path),
                     '{}/igibson/data/ig_dataset/objects/pork/pork_000/pork_000.urdf'.format(self.configs.igibson_path),
                     '{}/igibson/data/ig_dataset/objects/fish/fish_000/fish_000.urdf'.format(self.configs.igibson_path),
                     '{}/igibson/data/ig_dataset/objects/basil/basil_000/basil_000.urdf'.format(self.configs.igibson_path),
                     '{}/igibson/data/ig_dataset/objects/parsley/32_0/32_0.urdf'.format(self.configs.igibson_path),
                     '{}/igibson/data/ig_dataset/objects/clove/20_1/20_1.urdf'.format(self.configs.igibson_path),
                     '{}/igibson/data/ig_dataset/objects/potato/44_3/44_3.urdf'.format(self.configs.igibson_path),
                     '{}/igibson/data/ig_dataset/objects/sweet_corn/24_0/24_0.urdf'.format(self.configs.igibson_path),
                     '{}/igibson/data/ig_dataset/objects/sushi/sushi_000/sushi_000.urdf'.format(self.configs.igibson_path),
                     '{}/igibson/data/ig_dataset/objects/lasagna/lasagna_000/lasagna_000.urdf'.format(self.configs.igibson_path),
                     ]

        self.food_name = ['banana',
                          'carrot',
                          'chicken',
                          'strawberry',
                          'spinach',
                          'sausage',
                          'pomelo',
                          'butter',
                          'chocolate_box',
                          'pretzel',
                          'rib',
                          'pork',
                          'fish',
                          'basil',
                          'parsley',
                          'clove',
                          'potato',
                          'sweet_corn',
                          'sushi',
                          'lasagna'
                          ]

        self.food_scale = [1.0,
                           1.0,
                           0.8,
                           3.0,
                           1.0,
                           1.0,
                           1.0,
                           0.7,
                           0.3,
                           0.4,
                           0.8,
                           0.8,
                           0.8,
                           1.5,
                           2.0,
                           2.0,
                           1.0,
                           1.0,
                           1.0,
                           1.0
                           ]

        self.food_id = []

        for idx in range(len(self.food)):
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

        self.surface = ['countertop_76', 'burner_77']

        self.env.task.initial_state = self.env.task.save_scene()

        self.initial_height = self.env.initial_pos_z_offset
        self.motion_planner = MotionPlanningWrapper(self.env, fine_motion_plan=False)

        self.robot = self.env.robots[0]
        self.robot_id = self.robot.robot_ids[0]

        self.graspable_obj_list = []
        self.target_obj_list = ['dishtowel-Tag_Dishtowel_Dobby_Stripe_Blue-0', 'microwave_36', 'sink_19', 'countertop_18', 'oven_24', 'hand_towel-Threshold_Hand_Towel_Light_Yellow-0', 'countertop_23']

        self.in_hand = -1
        self.grasp_pose = None
        self.previous_traj = None
        self.current_state = None

        self.base_close_to = -1
        self.view_rgb = []
        self.view_pcd = []
        self.view_seg = []

        self.small_object_list = []

        self.reset_buffer = {}

        self.random_coin = 0

        self.first = True
        self.towel_init_pos, self.towel_init_ori = None, None

        for j in get_joints(self.robot_id):
            info = get_joint_info(self.robot_id, j)
            if info.jointName.decode() == 'torso_lift_joint':
                self.lift_joint = j

    def reset(self):

        self.in_hand = -1
        self.grasp_pose = None

        self.previous_traj = None

        self.current_state = None

        self.base_close_to = -1
        self.view_rgb = []
        self.view_pcd = []
        self.view_seg = []

        self.env.reset()

        self.towel_list = []

        for _, obj in self.env.task.object_scope.items():
            if obj.category in ["agent", "room_floor"]:
                continue
            if obj.name != 'microwave_36':
                continue
            for j in get_joints(obj.body_ids[0]):
                info = get_joint_info(obj.body_ids[0], j)
                jointType = info.jointType
                if jointType in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
                    set_joint_position(obj.body_ids[0], j, np.pi * 0.6)

        for key in self.env.scene.objects_by_name:
            if 'towel' in key:
                self.towel_list.append(copy.deepcopy(key))

        self.robot_pos = (-5.3, -4.5, np.pi / 2)
        set_base_values_with_z(self.robot_id, self.robot_pos, z=self.initial_height)


    def reset_obj(self, layout=0):
        must_list = self.utils_name.copy()
        switch_list = self.utils_name.copy() + random.sample(self.bowl_name.copy(), 1) + random.sample(self.food_name.copy() + self.ycb_name.copy(), random.randint(2, 5))

        if layout == 0:
            pan_pos = copy.deepcopy(list(self.env.scene.objects_by_name[self.surface[1]].get_position()))
            pan_pos[0] += random.uniform(-0.2, 0.2)
            pan_pos[1] += random.uniform(-0.2, 0.2)
            pan_pos[2] += 0.05

            bowl_pos = copy.deepcopy(list(self.env.scene.objects_by_name[self.surface[0]].get_position()))
            bowl_pos[0] += random.uniform(0.4, 0.5)
            bowl_pos[1] += random.uniform(-0.15, -0.05)
            bowl_pos[2] += 0.1

        elif layout == 1:
            pos = copy.deepcopy(list(self.env.scene.objects_by_name[self.surface[1]].get_position()))
            pos[0] += 0.7
            self.env.scene.objects_by_name[self.surface[1]].set_position(pos, force=True)

            pan_pos = copy.deepcopy(list(self.env.scene.objects_by_name[self.surface[1]].get_position()))
            pan_pos[0] += random.uniform(-0.2, 0.2)
            pan_pos[1] += random.uniform(-0.2, 0.2)
            pan_pos[2] += 0.05

            bowl_pos = copy.deepcopy(list(self.env.scene.objects_by_name[self.surface[0]].get_position()))
            bowl_pos[0] += random.uniform(-0.2, 0.1)
            bowl_pos[1] += random.uniform(-0.25, -0.15)
            bowl_pos[2] += 0.1
        else:
            pos = copy.deepcopy(list(self.env.scene.objects_by_name[self.surface[1]].get_position()))
            pos[0] += 0.35
            pos[1] += 0.3
            self.env.scene.objects_by_name[self.surface[1]].set_position(pos, force=True)

            pan_pos = copy.deepcopy(list(self.env.scene.objects_by_name[self.surface[1]].get_position()))
            pan_pos[0] += random.uniform(-0.2, 0.2)
            pan_pos[1] += random.uniform(-0.1, -0.05)
            pan_pos[2] += 0.05

            bowl_pos = copy.deepcopy(list(self.env.scene.objects_by_name[self.surface[0]].get_position()))
            bowl_pos[0] += random.uniform(0.0, 0.2)
            bowl_pos[1] += random.uniform(-0.3, -0.2)
            bowl_pos[2] += 0.1


        for i in range(len(switch_list)):
            if switch_list[i] in must_list:
                in_air_coin = random.randint(0, 3)
                if in_air_coin > 0:
                    pos = pan_pos
                    self.pan_in_air = False
                else:
                    surface_select = self.surface[0]
                    pos = copy.deepcopy(list(self.env.scene.objects_by_name[surface_select].get_position()))
                    pos[0] += random.uniform(0.4, 0.6)
                    pos[1] += random.uniform(-0.3, 0.1)
                    pos[2] += 0.3
                    self.pan_in_air = True
                    self.pan_in_air_pos = copy.deepcopy(pos)

                self.env.scene.objects_by_name[switch_list[i]].set_position(pos)
                self.env.scene.objects_by_name[switch_list[i]].rotate_by(0.0, 0.0, np.pi)
            elif 'bowl' in switch_list[i]:

                self.env.scene.objects_by_name[switch_list[i]].set_position(bowl_pos)
                self.env.scene.objects_by_name[switch_list[i]].rotate_by(0.0, 0.0, random.uniform(-1.0, 1.0))
            else:
                pos = copy.deepcopy(list(self.env.scene.objects_by_name[self.surface[0]].get_position()))
                pos[0] += random.uniform(-0.2, 0.6)
                pos[1] += random.uniform(-0.4, 0.2)
                pos[2] += random.uniform(0.2, 0.3)

                self.env.scene.objects_by_name[switch_list[i]].set_position(pos)
                self.env.scene.objects_by_name[switch_list[i]].rotate_by(0.0, 0.0, random.uniform(-1.0, 1.0))
                if not 'object' in switch_list[i]:
                    cooked_coin = random.randint(0, 1)
                    if cooked_coin == 0:
                        self.env.scene.objects_by_name[switch_list[i]].states[Cooked].set_value(False)
                    else:
                        self.env.scene.objects_by_name[switch_list[i]].states[Cooked].set_value(True)

        for i in range(12):
            self.env.scene.objects_by_name['burner_77'].states[ToggledOn].set_value(False)
            self.env.simulator.step()
            set_base_values_with_z(self.robot_id, self.robot_pos, z=self.initial_height)
            if self.pan_in_air:
                self.env.scene.objects_by_name[must_list[0]].set_position(self.pan_in_air_pos)

    def get_syb_state(self):

        if (self.env.scene.objects_by_name[self.target_obj_list[0]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[2]])) and self.env.scene.objects_by_name[self.target_obj_list[0]].get_position()[2] < 0.63:
            self.env.scene.objects_by_name[self.target_obj_list[0]].states[Soaked].set_value(True)
        if (self.env.scene.objects_by_name[self.target_obj_list[5]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[2]])) and self.env.scene.objects_by_name[self.target_obj_list[5]].get_position()[2] < 0.63:
            self.env.scene.objects_by_name[self.target_obj_list[5]].states[Soaked].set_value(True)

        syb_state = [self.env.scene.objects_by_name[self.target_obj_list[0]].states[Soaked].get_value(),
                     self.env.scene.objects_by_name[self.target_obj_list[5]].states[Soaked].get_value(),
                     self.env.scene.objects_by_name[self.target_obj_list[1]].states[Dusty].get_value(),
                     self.env.scene.objects_by_name[self.target_obj_list[1]].states[Stained].get_value(),
                     self.in_hand == self.env.scene.objects_by_name[self.graspable_obj_list[0]].get_body_id(),
                     self.in_hand == self.env.scene.objects_by_name[self.graspable_obj_list[1]].get_body_id(),
                     self.base_close_to == 2,
                     self.in_hand != -1,
                     self.env.scene.objects_by_name[self.target_obj_list[0]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[2]]),
                     self.env.scene.objects_by_name[self.target_obj_list[0]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[4]]),
                     self.env.scene.objects_by_name[self.target_obj_list[0]].states[Inside].get_value(self.env.scene.objects_by_name[self.target_obj_list[1]]),
                     self.env.scene.objects_by_name[self.target_obj_list[5]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[2]]),
                     self.env.scene.objects_by_name[self.target_obj_list[5]].states[OnTop].get_value(self.env.scene.objects_by_name[self.target_obj_list[4]]),
                     self.env.scene.objects_by_name[self.target_obj_list[5]].states[Inside].get_value(self.env.scene.objects_by_name[self.target_obj_list[1]])]

        syb_string = np.array2string(np.array(syb_state).astype(np.int16))
        if (not syb_string in self.reset_buffer.keys()) and (sum(syb_state[-7:]) == 2) and (sum(syb_state[2:4]) == 2) and (sum(syb_state[:2]) == 0):
            self.reset_buffer[syb_string] = {}
            self.reset_buffer[syb_string]['p_state'] = p.saveState()
            self.reset_buffer[syb_string]['in_hand'] = copy.deepcopy(self.in_hand)
            self.reset_buffer[syb_string]['grasp_pose'] = copy.deepcopy(self.grasp_pose)
            self.reset_buffer[syb_string]['previous_traj'] = copy.deepcopy(self.previous_traj)
            self.reset_buffer[syb_string]['current_state'] = copy.deepcopy(self.current_state)
            self.reset_buffer[syb_string]['base_close_to'] = copy.deepcopy(self.base_close_to)
            self.reset_buffer[syb_string]['view_rgb'] = copy.deepcopy(self.view_rgb)
            self.reset_buffer[syb_string]['view_pcd'] = copy.deepcopy(self.view_pcd)
            self.reset_buffer[syb_string]['view_seg'] = copy.deepcopy(self.view_seg)

        elif (syb_string in self.reset_buffer.keys()) and (sum(syb_state[-7:]) == 2) and (sum(syb_state[2:4]) == 2) and (sum(syb_state[:2]) == 0):
            p.removeState(self.reset_buffer[syb_string]['p_state'])
            self.reset_buffer[syb_string] = {}
            self.reset_buffer[syb_string]['p_state'] = p.saveState()
            self.reset_buffer[syb_string]['in_hand'] = copy.deepcopy(self.in_hand)
            self.reset_buffer[syb_string]['grasp_pose'] = copy.deepcopy(self.grasp_pose)
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

        rgb = states['rgb']
        ins_seg = states['ins_seg']

        if reset or len(self.view_rgb) == 0:
            self.view_rgb.append(copy.deepcopy(rgb))
            self.view_pcd.append(states['pc'].copy())
            self.view_seg.append(copy.deepcopy(ins_seg))
        else:
            self.view_rgb[0] = copy.deepcopy(rgb)
            self.view_pcd[0] = states['pc'].copy()
            self.view_seg[0] = copy.deepcopy(ins_seg)


    def show_view(self):
        fig, axarr = plt.subplots(1, 2)
        axarr[0].imshow(self.view_rgb[0], interpolation='none')
        axarr[1].imshow(self.view_rgb[1], interpolation='none')

        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        global fig_size
        fig_size = fig.get_size_inches() * fig.dpi

        plt.show()

    def check_grasping(self, gripper):
        if self.in_hand == -1 and gripper < 0:
            gripper_pos = self.robot.get_end_effector_position()
            minimal = 0.25
            for item in self.graspable_obj_list:
                item_pos = list(self.env.scene.objects_by_name[item].get_position())
                dist = l2_distance(gripper_pos, item_pos)
                if dist < minimal:
                    minimal = dist
                    self.in_hand = self.env.scene.objects_by_name[item].get_body_id()
                    obj_pos, obj_orn = p.getBasePositionAndOrientation(self.in_hand)
                    gripper_pos = p.getLinkState(self.robot_id, self.robot.end_effector_part_index())[0]
                    gripper_orn = p.getLinkState(self.robot_id, self.robot.end_effector_part_index())[1]
                    self.grasp_pose = p.multiplyTransforms(*p.invertTransform(gripper_pos, gripper_orn), obj_pos, obj_orn)
                    self.grasp_pose = list(self.grasp_pose)
                    self.grasp_pose[0] = (self.grasp_pose[0][0], self.grasp_pose[0][1], -0.08)
                    self.grasp_pose[1] = (0.0, 0.0, 0.0, 1.0)

        elif self.in_hand != -1 and gripper >= 0:
            self.in_hand = -1
            self.grasp_pose = None

    def update_in_hand(self):
        if self.in_hand != -1:
            gripper_pos = p.getLinkState(self.robot_id, self.robot.end_effector_part_index())[0]
            gripper_orn = p.getLinkState(self.robot_id, self.robot.end_effector_part_index())[1]
            object_pose = p.multiplyTransforms(gripper_pos, gripper_orn, self.grasp_pose[0], self.grasp_pose[1])
            set_pose(self.in_hand, object_pose)
            self.env.simulator.sync()


    def random_step(self, steps, hold=False, real_step=False):
        for i in range(steps):
            action = [0.0 for _ in range(11)]
            if hold:
                action[-1] = -0.5
            else:
                action[-1] = 0.5

            self.check_grasping(action[-1])
            if real_step:
                self.current_state, reward, done, info = self.env.step(action)
            else:
                self.current_state = None

        return self.current_state

    def close(self):
        self.env.close()