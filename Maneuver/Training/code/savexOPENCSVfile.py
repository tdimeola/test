#!/usr/bin/env python3

import sys
#(TD) sys.path.insert(0, '../../../../')

import tensorflow as tf
from mpi4py import MPI
from baselines.common import set_global_seeds
from baselines.common import tf_util as U
from baselines import logger
from baselines.common.cmd_util import make_robotics_env, robotics_arg_parser
import hybrid3D_envx
from Utils import firmware_generator
import random
import time
import os
import os.path
from os import path
import numpy as np
import math
import argparse
import csv
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def save(model_directory, model_filename):
    # check directory exist, if not create one
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)

#################################################################
#reset trajectory file or make it new
    if path.exists("trajectory_4.csv"):
        os.remove("trajectory_4.csv")

    this_list = (0,0,0)
    df = pd.DataFrame(this_list)
    df.to_csv('trajectory_4.csv')

    from baselines.ppo1 import mlp_policy, pposgd_simple
    U.make_session(num_cpu=1).__enter__()
    
    def policy_fn(name, ob_space, ac_space):
        return mlp_policy.MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
            hid_size=64, num_hid_layers=2)
    
    env = hybrid3D_envx.Hybrid3DEnv(data_folder = "../data/", config_file = "QuadPlanex.xml", play = True)

    pi = pposgd_simple.build_graph_only(env, policy_fn,
                max_timesteps=1000000,
                timesteps_per_actorbatch=4096,
                clip_param=0.2, entcoeff=0.0,
                optim_epochs=10, 
                optim_stepsize=3e-4, 
                optim_batchsize=64,
                gamma=0.995, lam=0.95, schedule='linear')

    saver  = tf.train.Saver()
    print('model_filename is:', model_filename)
    saver.restore(tf.get_default_session(), model_directory + model_filename)
    print("loaded")

    W0, b0, W_hidden, b_hidden, W1, b1, ob_mean, ob_std = pposgd_simple.get_policy_parameters(pi)

    fg = firmware_generator.FirmwareGenerator(W0 = W0, b0 = b0, W_hidden = W_hidden, b_hidden = b_hidden, \
                W1 = W1, b1 = b1, num_hidden_layer = len(W_hidden) + 1, ob_space_size = W0.shape[0], \
                hidden_layer_size = W0.shape[1], ac_space_size = W1.shape[1], state_size = 9, \
                final_bias = 3.5, ob_mean = ob_mean, ob_std = ob_std)
    fg.generate()

    # Test weights
    _iter = 0
    ob = env.reset()
    while True:
        print(ob)
        ob = (ob - ob_mean) / ob_std
        ob = np.clip(ob, -5.0, 5.0)
#        print("final ob = ", ob)
#        time.sleep(30)
        last_out = np.tanh(ob.dot(W0) + b0)
        for j in range(len(W_hidden)):
            last_out = np.tanh(last_out.dot(W_hidden[j]) + b_hidden[j])
        ac = last_out.dot(W1) + b1  
#        print('ac is:', ac)
#        time.sleep(3)
        ob, _, _, _ = env.step(ac)
        _iter += 1

    env.close()
    #show trajectory
    figure = plt.figure()
    axis = figure.add_subplot(111, projection = '3d')
	

    colnames = ['x', 'y', 'z']
    data = pd.read_csv('trajectory_4.csv', names=colnames)

    x = data.x.tolist()
    y = data.y.tolist()
    z = data.z.tolist()
    z = np.array([z,z])

    axis.plot_wireframe(x, y, z)

    axis.set_xlabel('x-axis')
    axis.set_ylabel('y-axis')
    axis.set_zlabel('z-axis')

    plt.show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller', type = str, default = "")

    args = parser.parse_args()
    controller = args.controller

    logger.configure()
    save(model_directory="./model/",
        model_filename=controller + ".ckpt")
    
if __name__=='__main__':
    main()
