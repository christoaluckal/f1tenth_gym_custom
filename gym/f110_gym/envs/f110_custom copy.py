# MIT License

# Copyright (c) 2020 Joseph Auckley, Matthew O'Kelly, Aman Sinha, Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

'''
Author: Hongrui Zheng
'''

# gym imports
import gym
from gym import error, spaces, utils
from gym.utils import seeding

# base classes
from f110_gym.envs.base_classes import Simulator, Integrator

# others
import numpy as np
import os
import time

# gl
import pyglet
pyglet.options['debug_gl'] = False
from pyglet import gl

from f110_gym.envs import path_gen as pg

from scipy.optimize import minimize

from numba import njit

# constants

# rendering
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 1000
WINDOW_H = 800

class F110_Cust_Env(gym.Env):
    """
    OpenAI gym environment for F1TENTH
    
    Env should be initialized by calling gym.make('f110_gym:f110-v0', **kwargs)

    Args:
        kwargs:
            seed (int, default=12345): seed for random state and reproducibility
            
            map (str, default='vegas'): name of the map used for the environment. Currently, available environments include: 'berlin', 'vegas', 'skirk'. You could use a string of the absolute path to the yaml file of your custom map.
        
            map_ext (str, default='png'): image extension of the map image file. For example 'png', 'pgm'
        
            params (dict, default={'mu': 1.0489, 'C_Sf':, 'C_Sr':, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch':7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}): dictionary of vehicle parameters.
            mu: surface friction coefficient
            C_Sf: Cornering stiffness coefficient, front
            C_Sr: Cornering stiffness coefficient, rear
            lf: Distance from center of gravity to front axle
            lr: Distance from center of gravity to rear axle
            h: Height of center of gravity
            m: Total mass of the vehicle
            I: Moment of inertial of the entire vehicle about the z axis
            s_min: Minimum steering angle constraint
            s_max: Maximum steering angle constraint
            sv_min: Minimum steering velocity constraint
            sv_max: Maximum steering velocity constraint
            v_switch: Switching velocity (velocity at which the acceleration is no longer able to create wheel spin)
            a_max: Maximum longitudinal acceleration
            v_min: Minimum longitudinal velocity
            v_max: Maximum longitudinal velocity
            width: width of the vehicle in meters
            length: length of the vehicle in meters

            num_agents (int, default=2): number of agents in the environment

            timestep (float, default=0.01): physics timestep

            ego_idx (int, default=0): ego's index in list of agents
    """
    metadata = {'render.modes': ['human', 'human_fast']}

    # rendering
    renderer = None
    current_obs = None
    render_callbacks = []

    def __init__(self, **kwargs):        
        # kwargs extraction
        try:
            self.seed = kwargs['seed']
        except:
            self.seed = 12345
        # try:
        #     self.map_name = kwargs['map']
        #     print(self.map_name)
        #     # different default maps
        #     if self.map_name == 'berlin':
        #         self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/berlin.yaml'
        #     elif self.map_name == 'skirk':
        #         self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/skirk.yaml'
        #     elif self.map_name == 'levine':
        #         self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/levine.yaml'
        #     else:
        #         self.map_path = self.map_name + '.yaml'
        # except:
        #     self.map_path = os.path.dirname(os.path.abspath(__file__)) + '/maps/vegas.yaml'
            
        map_configs = kwargs['config']

        self.map_path = map_configs['map_yaml'] + '.yaml'


        try:
            self.map_ext = kwargs['map_ext']
        except:
            self.map_ext = '.png'

        try:
            self.params = kwargs['params']
        except:
            self.params = {'mu': 1.0489, 'C_Sf': 4.718, 'C_Sr': 5.4562, 'lf': 0.15875, 'lr': 0.17145, 'h': 0.074, 'm': 3.74, 'I': 0.04712, 's_min': -0.4189, 's_max': 0.4189, 'sv_min': -3.2, 'sv_max': 3.2, 'v_switch': 7.319, 'a_max': 9.51, 'v_min':-5.0, 'v_max': 20.0, 'width': 0.31, 'length': 0.58}

        # simulation parameters
        try:
            self.num_agents = kwargs['num_agents']
        except:
            self.num_agents = 2

        try:
            self.timestep = kwargs['timestep']
        except:
            self.timestep = 0.01

        # default ego index
        try:
            self.ego_idx = kwargs['ego_idx']
        except:
            self.ego_idx = 0

        # default integrator
        try:
            self.integrator = kwargs['integrator']
        except:
            self.integrator = Integrator.RK4

        try:
            self.csv = kwargs['wpt_path']
        except:
            self.csv = '/home/caluckal/Developer/spring2024/thesis/f1tenth_gym/gym/f110_gym/unittest/centerline/map_44_1000_.csv'

        print(self.csv)

        self.waypoints, self.track_length, self.x_spline, self.y_spline = pg.get_scaled_spline_path(self.csv, x_idx=0, y_idx=1, scale=1.0)

        t_vec = np.linspace(0,self.track_length,1000)
        import matplotlib.pyplot as plt
        x_points = self.x_spline(t_vec)
        y_points = self.y_spline(t_vec)
        plt.plot(x_points,y_points)
        plt.show()

        # radius to consider done
        self.start_thresh = 0.5  # 10cm

        # env states
        self.poses_x = []
        self.poses_y = []
        self.poses_theta = []
        self.collisions = np.zeros((self.num_agents, ))
        # TODO: collision_idx not used yet
        # self.collision_idx = -1 * np.ones((self.num_agents, ))

        # loop completion
        self.near_start = True
        self.num_toggles = 0

        # race info
        self.lap_times = np.zeros((self.num_agents, ))
        self.lap_counts = np.zeros((self.num_agents, ))
        self.current_time = 0.0

        # finish line info
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True]*self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))
        self.start_xs = np.zeros((self.num_agents, ))
        self.start_ys = np.zeros((self.num_agents, ))
        self.start_thetas = np.zeros((self.num_agents, ))
        self.start_rot = np.eye(2)

        # initiate stuff
        self.sim = Simulator(self.params, self.num_agents, self.seed, time_step=self.timestep, integrator=self.integrator)
        self.sim.set_map(self.map_path, self.map_ext)

        # stateful observations for rendering
        self.render_obs = None

    def distance_to_spline(self, t, x, y):
        spline_x = self.x_spline(t)
        spline_y = self.y_spline(t)
        dx = (spline_x - x)**2
        dy = (spline_y - y)**2
        return np.sqrt(dx + dy)
    
    def euclidean_dist(self,p1,p2):
        return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
    

    def closest_spline_param(self,current_x,current_y,x_spline,y_spline,best_t=0):
        # res = minimize(fn,x0=best_t, args=(current_x, current_y,x_spline,y_spline))
        # return res.x
        skips = 5
        deltas = self.track_length/skips
        ts = np.arange(0,self.track_length,deltas)


        min_dist = 1e6
        min_t = 0

        for t in ts:
            res = minimize(self.distance_to_spline,x0=t, args=(current_x, current_y),bounds=((0,self.track_length),))
            p = [x_spline(res.x[0]),y_spline(res.x[0])]
            dist = self.euclidean_dist([current_x,current_y],p)
            if dist<min_dist:
                min_dist = dist
                min_t = res.x


        return min_t[0]

    def __del__(self):
        """
        Finalizer, does cleanup
        """
        pass

    def _check_done(self):
        """
        Check if the current rollout is done
        
        Args:
            None

        Returns:
            done (bool): whether the rollout is done
            toggle_list (list[int]): each agent's toggle list for crossing the finish zone
        """

        # this is assuming 2 agents
        # TODO: switch to maybe s-based
        left_t = 2
        right_t = 2
        
        poses_x = np.array(self.poses_x)-self.start_xs
        poses_y = np.array(self.poses_y)-self.start_ys
        delta_pt = np.dot(self.start_rot, np.stack((poses_x, poses_y), axis=0))
        temp_y = delta_pt[1,:]
        idx1 = temp_y > left_t
        idx2 = temp_y < -right_t
        temp_y[idx1] -= left_t
        temp_y[idx2] = -right_t - temp_y[idx2]
        temp_y[np.invert(np.logical_or(idx1, idx2))] = 0

        dist2 = delta_pt[0, :]**2 + temp_y**2
        closes = dist2 <= 0.1
        for i in range(self.num_agents):
            if closes[i] and not self.near_starts[i]:
                self.near_starts[i] = True
                self.toggle_list[i] += 1
            elif not closes[i] and self.near_starts[i]:
                self.near_starts[i] = False
                self.toggle_list[i] += 1
            self.lap_counts[i] = self.toggle_list[i] // 2
            if self.toggle_list[i] < 4:
                self.lap_times[i] = self.current_time
        
        done = (self.collisions[self.ego_idx]) or np.all(self.toggle_list >= 4)
        
        return bool(done), self.toggle_list >= 4

    def _update_state(self, obs_dict):
        """
        Update the env's states according to observations
        
        Args:
            obs_dict (dict): dictionary of observation

        Returns:
            None
        """
        self.poses_x = obs_dict['poses_x']
        self.poses_y = obs_dict['poses_y']
        self.poses_theta = obs_dict['poses_theta']
        self.collisions = obs_dict['collisions']

    def step(self, action):
        """
        Step function for the gym env

        Args:
            action (np.ndarray(num_agents, 2))

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        try:
            prev_x = self.poses_x[self.ego_idx]
            prev_y = self.poses_y[self.ego_idx]
        except:
            print("Ego idx not found")
            prev_x = 0
            prev_y = 0

        previous_closest_t = self.closest_spline_param(prev_x,prev_y,self.x_spline,self.y_spline)

        # call simulation step
        obs = self.sim.step(action)
        obs['lap_times'] = self.lap_times
        obs['lap_counts'] = self.lap_counts

        F110_Cust_Env.current_obs = obs

        self.render_obs = {
            'ego_idx': obs['ego_idx'],
            'poses_x': obs['poses_x'],
            'poses_y': obs['poses_y'],
            'poses_theta': obs['poses_theta'],
            'lap_times': obs['lap_times'],
            'lap_counts': obs['lap_counts']
            }
        
        curr_x = obs['poses_x'][self.ego_idx]
        curr_y = obs['poses_y'][self.ego_idx]

        current_closest_t = self.closest_spline_param(curr_x,curr_y,self.x_spline,self.y_spline,previous_closest_t)

        print(f"Prev_X: {np.round(prev_x,3)} Prev_Y: {np.round(prev_y,3)} Curr_X: {np.round(curr_x,3)} Curr_Y: {np.round(curr_y,3)} Prev_T: {np.round(previous_closest_t,3)} Curr_T: {np.round(current_closest_t,3)}")
        # times
        reward = self.timestep
        self.current_time = self.current_time + self.timestep
        
        # update data member
        self._update_state(obs)

        # check done
        done, toggle_list = self._check_done()
        info = {'checkpoint_done': toggle_list}

        return obs, reward, done, info

    def reset(self, poses):
        """
        Reset the gym environment by given poses

        Args:
            poses (np.ndarray (num_agents, 3)): poses to reset agents to

        Returns:
            obs (dict): observation of the current step
            reward (float, default=self.timestep): step reward, currently is physics timestep
            done (bool): if the simulation is done
            info (dict): auxillary information dictionary
        """
        # reset counters and data members
        self.current_time = 0.0
        self.collisions = np.zeros((self.num_agents, ))
        self.num_toggles = 0
        self.near_start = True
        self.near_starts = np.array([True]*self.num_agents)
        self.toggle_list = np.zeros((self.num_agents,))

        # states after reset
        self.start_xs = poses[:, 0]
        self.start_ys = poses[:, 1]
        self.start_thetas = poses[:, 2]
        self.start_rot = np.array([[np.cos(-self.start_thetas[self.ego_idx]), -np.sin(-self.start_thetas[self.ego_idx])], [np.sin(-self.start_thetas[self.ego_idx]), np.cos(-self.start_thetas[self.ego_idx])]])

        # call reset to simulator
        self.sim.reset(poses)

        # get no input observations
        action = np.zeros((self.num_agents, 2))
        obs, reward, done, info = self.step(action)

        self.render_obs = {
            'ego_idx': obs['ego_idx'],
            'poses_x': obs['poses_x'],
            'poses_y': obs['poses_y'],
            'poses_theta': obs['poses_theta'],
            'lap_times': obs['lap_times'],
            'lap_counts': obs['lap_counts']
            }
        
        return obs, reward, done, info

    # def reset(self):
    #     """
    #     Reset the gym environment by given poses

    #     Args:
    #         poses (np.ndarray (num_agents, 3)): poses to reset agents to

    #     Returns:
    #         obs (dict): observation of the current step
    #         reward (float, default=self.timestep): step reward, currently is physics timestep
    #         done (bool): if the simulation is done
    #         info (dict): auxillary information dictionary
    #     """
    #     # reset counters and data members
    #     self.current_time = 0.0
    #     self.collisions = np.zeros((self.num_agents, ))
    #     self.num_toggles = 0
    #     self.near_start = True
    #     self.near_starts = np.array([True]*self.num_agents)
    #     self.toggle_list = np.zeros((self.num_agents,))

    #     # states after reset
    #     self.start_xs = np.zeros((self.num_agents, ))
    #     self.start_ys = np.zeros((self.num_agents, ))
    #     self.start_thetas = np.zeros((self.num_agents, ))
    #     self.start_rot = np.array([[np.cos(-self.start_thetas[self.ego_idx]), -np.sin(-self.start_thetas[self.ego_idx])], [np.sin(-self.start_thetas[self.ego_idx]), np.cos(-self.start_thetas[self.ego_idx])]])

    #     poses = np.zeros((self.num_agents, 3))

    #     # call reset to simulator
    #     self.sim.reset(poses)

    #     # get no input observations
    #     action = np.zeros((self.num_agents, 2))
    #     obs, reward, done, info = self.step(action)

    #     self.render_obs = {
    #         'ego_idx': obs['ego_idx'],
    #         'poses_x': obs['poses_x'],
    #         'poses_y': obs['poses_y'],
    #         'poses_theta': obs['poses_theta'],
    #         'lap_times': obs['lap_times'],
    #         'lap_counts': obs['lap_counts']
    #         }
        
    #     return obs, reward, done, info

    def update_map(self, map_path, map_ext):
        """
        Updates the map used by simulation

        Args:
            map_path (str): absolute path to the map yaml file
            map_ext (str): extension of the map image file

        Returns:
            None
        """
        self.sim.set_map(map_path, map_ext)

    def update_params(self, params, index=-1):
        """
        Updates the parameters used by simulation for vehicles
        
        Args:
            params (dict): dictionary of parameters
            index (int, default=-1): if >= 0 then only update a specific agent's params

        Returns:
            None
        """
        self.sim.update_params(params, agent_idx=index)

    def add_render_callback(self, callback_func):
        """
        Add extra drawing function to call during rendering.

        Args:
            callback_func (function (EnvRenderer) -> None): custom function to called during render()
        """

        F110_Cust_Env.render_callbacks.append(callback_func)

    def render(self, mode='human'):
        """
        Renders the environment with pyglet. Use mouse scroll in the window to zoom in/out, use mouse click drag to pan. Shows the agents, the map, current fps (bottom left corner), and the race information near as text.

        Args:
            mode (str, default='human'): rendering mode, currently supports:
                'human': slowed down rendering such that the env is rendered in a way that sim time elapsed is close to real time elapsed
                'human_fast': render as fast as possible

        Returns:
            None
        """
        assert mode in ['human', 'human_fast']
        
        if F110_Cust_Env.renderer is None:
            # first call, initialize everything
            from f110_gym.envs.rendering import EnvRenderer
            F110_Cust_Env.renderer = EnvRenderer(WINDOW_W, WINDOW_H)
            F110_Cust_Env.renderer.update_map(self.map_name, self.map_ext)
            
        F110_Cust_Env.renderer.update_obs(self.render_obs)

        for render_callback in F110_Cust_Env.render_callbacks:
            render_callback(F110_Cust_Env.renderer)
        
        F110_Cust_Env.renderer.dispatch_events()
        F110_Cust_Env.renderer.on_draw()
        F110_Cust_Env.renderer.flip()
        if mode == 'human':
            time.sleep(0.005)
        elif mode == 'human_fast':
            pass
