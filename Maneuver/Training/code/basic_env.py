import rotor
from rotor import Rotor
import wing
from wing import Wing
from Utils import rigid_body
from Utils import euler_angle
import time
import xml.etree.ElementTree as ET
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from pyquaternion import Quaternion
import matplotlib.pyplot as plt
import seaborn as sns
import csv	
from csv import writer


class Hybrid3DEnv(gym.Env):
    def __init__(self, data_folder, config_file, play):
        self.render_filename = None
        self.mass = None
        self.inertia_tensor = None
        self.rotors = []
        self.wing = None

        # initialize constant value
        self.gravity = np.array([0, 0, 9.8])
        self.dt_mean = 0.01
        self.dt_std = 0.005
        self.total_iter = 2000

        # parse xml config file
        self.parse_config_file(data_folder + config_file)
        
        # construct rendering environment
        from mujoco_rendering_env import mujoco_env
        self.render_env = mujoco_env.MujocoEnv(model_path = data_folder + self.render_filename)
        self.render_intervel = int(1.0 / 50.0 / self.dt_mean)

        # self.render_env._get_viewer()
        
        # integral term
        self.I_dt = self.dt_mean
        self.I_error = None

        # mass property
        self.real_mass = self.mass
        self.real_inertia_tensor = self.inertia_tensor.copy()

        # train or play
        self.play = play

        # construct action space

        # construct observation space
        ob = self.get_observation_vector()
        ob_low = np.ones(len(ob)) * (-np.finfo(np.float32).max)
        ob_high = np.ones(len(ob)) * (np.finfo(np.float32).max)
        self.observation_space = spaces.Box(ob_low, ob_high, dtype=np.float32)
    
    #########################################################################
    # parse config file
    def parse_config_file(self, config_file):
        xml_tree = ET.parse(config_file)
        root = xml_tree.getroot()
        self.parse_xml_tree(root)


    #########################################################################

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def wrap2PI(self, x):
        while x > math.pi:
            x -= math.pi * 2.0
        while x < -math.pi:
            x += math.pi * 2.0
        return x


    def get_observation_vector(self):
        if self.state_his is None:
            # For observation space construction
            state = np.zeros(12)
        else:
            if self.simulate_delay:
                if self.timesofar > self.delay:
                    for i in reversed(range(len(self.state_his))):
                        if self.time_his[i] <= self.timesofar - self.delay:
                            state = self.state_his[i]
                            break
                else:
                    state = self.state_his[0]
            else:
                state = self.state_his[len(self.state_his) - 1]

        rpy = state[3:6]
        vel = state[6:9]
        omega = state[9:12]

        now = np.hstack((vel, rpy[2]))
        error = self.target - now

        coeff = 0.999
        self.I_error = coeff * self.I_error + self.I_dt * error
        self.I_error = np.clip(self.I_error, -10.0, 10.0)

        ob = np.hstack((rpy, omega, error[0:3], self.I_error))

        if (ob[0] < -math.pi or ob[1] < -math.pi or ob[2] < -math.pi or ob[0] > math.pi or ob[1] > math.pi or ob[2] > math.pi):
            print("ob wrong: ", ob)
            input()

        return ob

    def update_state(self):
        # get local vel
        vel_local = self.calc_local_velocity(self.rigid_body.rpy[2], self.rigid_body.velocity)
        
        now_state = np.concatenate([self.rigid_body.position, self.rigid_body.rpy, vel_local, self.rigid_body.omega_body])
 #       print('self.rigid_body.position= ', self.rigid_body.position)

        def append_list_as_row(file_name, list_of_elem):
            # Open file in append mode
            with open(file_name, 'a+', newline='') as write_obj:
                # Create a writer object from csv module
                csv_writer = writer(write_obj)
                # Add contents of list as last row in the csv file
                csv_writer.writerow(list_of_elem)
#                print('list of elements: ', list_of_elem)

        append_list_as_row('trajectory_3.csv', self.rigid_body.position)

        # add noise
        if self.noisy_sensor:
            noise = np.random.normal(self.state_noise_mean, self.state_noise_std)
            now_state = now_state + noise

        now_state[3] = self.wrap2PI(now_state[3])
        now_state[4] = self.wrap2PI(now_state[4])
        now_state[5] = self.wrap2PI(now_state[5])
        
        self.state_his.append(now_state)
        self.time_his.append(self.timesofar)
    
    def reset_noise(self):
        if self.noisy_body:
            self.mass = np.random.uniform(low = 0.95, high = 1.05) * self.real_mass
            for i in range(3):
                for j in range(3):
                    if i <= j:
                        self.inertia_tensor[i][j] = self.inertia_tensor[j][i] = self.real_inertia_tensor[i][j] * np.random.uniform(low = 0.6, high = 1.4)

        state_noise_mean_range = np.array([0.0, 0.0, 0.0, 0.02, 0.02, 0.1, 0.2, 0.2, 0.2, 0.00, 0.00, 0.00])
        self.state_noise_mean = np.random.uniform(low = -1.0 * state_noise_mean_range, high = state_noise_mean_range, size = 12)

    def reset(self):
 
        # set initial state
        
        mode = self.np_random.uniform(low = 0.0, high = 2.0)
        if mode < 1.0:    
            self.target_vx = self.np_random.uniform(low = 3.0, high = 6.0)
            self.target_vy = 0.0
            self.target_vz = self.np_random.uniform(low = -1.0, high = 1.0)
        else:
            self.target_vx = self.np_random.uniform(low = -1.0, high = 1.0)
            self.target_vy = self.np_random.uniform(low = -1.0, high = 1.0)
            self.target_vz = self.np_random.uniform(low = -1.0, high = 1.0)
        self.target = np.array([self.target_vx, self.target_vy, self.target_vz, 0])

        # training variables

        
        # for visualization
        self.rpy_his = [[], [], []]
        self.vel_local_his = [[], [], []]
        self.target_vel_his = [[], [], []]
        self.action_his = [[], [], [], []]
        self.omega_his = [[], [], []]
        self.state_his = []
        self.time_his = []
        self.error_his = [[], [], [], []]
        self.I_error_his = [[], [], [], []]
        self.aoa_his = []

        self.update_state()
        observation_vector = self.get_observation_vector()

        return observation_vector


    def step(self, action):

        # add noise to dt

        # apply gravity


        # update state and get observation vector
        self.update_state()
        ob = self.get_observation_vector()

        # render in render_env
        self.render_env.render()

        return ob, done, {}

    def close(self):
        if self.render_env:
            time.sleep(30)
            self.render_env.close()
            self.render_env = None

        
    
