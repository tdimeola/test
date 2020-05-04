

from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines.common import tf_util as U
from baselines import logger
from baselines.common.cmd_util import make_robotics_env, robotics_arg_parser
import gym
import hybrid3D_envx
import random
import time
import os
import argparse


def train(num_timesteps, copter_model, model_directory, save_model_interval, save_model_with_prefix, restore_model1_from_file, restore_model2_from_file, play, visualize):

    os.makedirs(model_directory, exist_ok=True) # Create model directory if it doesn't exist

    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)

    train_env = hybrid3D_envx.Hybrid3DEnv(data_folder = "../data/", config_file = copter_model + ".xml", play = False)
    print('Im in train_env after play=false')
    play_env = hybrid3D_envx.Hybrid3DEnv(data_folder = "../data/", config_file = copter_model + ".xml", play = visualize or play)

    pposgd_simple.learn(train_env, play_env, policy_fn,
            max_timesteps=num_timesteps,
            timesteps_per_actorbatch=4096,
            clip_param=0.2, entcoeff=0.0,
            optim_epochs=10, 
            optim_stepsize=3e-4, 
            optim_batchsize=64,
            gamma=0.995, lam=0.95, schedule='linear',
            model_directory=model_directory,
            save_model_interval=save_model_interval,
            save_model1_with_prefix=save_model1_with_prefix,
            save_model2_with_prefix=save_model2_with_prefix,
            restore_model1_from_file=restore_model1_from_file,
            restore_model2_from_file=restore_model2_from_file,
            play = play)

#    print("train_env is: ", train_env)
#    print("play_env is: ", play_env)
#   while(True):
#        print("looping...")
#        time.sleep(300)

    train_env.close()
    play_env.close()

import numpy as np
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type = str, default = "QuadPlane")
    parser.add_argument('--play', action='store_true')
    parser.add_argument('--controller1', type = str, default = "")
    parser.add_argument('--controller2', type = str, default = "")
    parser.add_argument('--visualize', action='store_true')

    args = parser.parse_args()

    model = args.model
    play = args.play
    controller1 = args.controller1
    controller2 = args.controller2
    visualize = args.visualize

    if controller1 == "":
        controller1 = None
    else:
        controller1 = controller1 + ".ckpt"

    if controller2 == "":
        controller2 = None
    else:
        controller2 = controller2 + ".ckpt"

    logger.configure()

    train(num_timesteps=30000000,
        copter_model = model,
        model_directory="./results/",
        save_model_interval = 5,
        save_model_with_prefix="model",
        restore_model1_from_file=controller1,
        restore_model2_from_file=controller2,
        play = play,
        visualize = visualize)

if __name__=='__main__':
    main()
