import os

import igibson
from gentp.lib.env import IndoorEnv
from gentp.configs.base_config import Configs


def main():
    configs = Configs()
    env = IndoorEnv(os.path.join(igibson.example_config_path, configs.env_config_file), 'train', 'iggui')

    for j in range(10):

        env.reset()
        print("SYB", env.get_syb_state())

        env.show_view()
        env.arm_pick_click()
        print("SYB", env.get_syb_state())

        env.show_view()
        env.arm_place_click()
        print("SYB", env.get_syb_state())

        env.show_view()
        env.arm_pick_click()
        print("SYB", env.get_syb_state())

        env.show_view()
        env.arm_pour_click()
        print("SYB", env.get_syb_state())

    env.close()

if __name__ == "__main__":
    main()
