import time
from f110_gym.envs.base_classes import Integrator
import yaml
import gym
import numpy as np
from argparse import Namespace
from gym.envs.registration import register
import argparse

from numba import njit

from pyglet.gl import GL_POINTS

from stable_baselines3.common.callbacks import BaseCallback,CheckpointCallback,CallbackList
import torch


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose: int = 0, 
                 config_type=1, 
                 modify_epoch=1, 
                 easy_policy_loc="",
                 base_policy_loc="", 
                 save_freq=1000, 
                 retain_ratio=0.8,
                 is_baseline=False
                 ):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.config_type = config_type
        self.modify_epoch = modify_epoch
        self.easy_policy_loc = easy_policy_loc
        self.base_policy_loc = base_policy_loc
        self.save_freq = save_freq
        self.retain_ratio = retain_ratio
        self.is_baseline = is_baseline
        self.modified = False
        self.has_saved = False

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """

        if self.config_type == 1:
            if self.n_calls % self.save_freq == 0:
                params = self.model.policy.state_dict()
                torch.save(params, self.easy_policy_loc)

        else:
            # if not self.is_baseline:
            #     try:
            #         if self.n_calls % self.modify_epoch == 0:
            #             curr_policy = self.model.policy.state_dict()
            #             tgt_policy = torch.load(self.policy_loc)

            #             print("Old Policy:", curr_policy['critic_target.qf0.4.bias'])

            #             for k in curr_policy.keys():
            #                 curr_policy[k] = curr_policy[k]*self.retain_ratio + tgt_policy[k]*(1-self.retain_ratio)
                        
            #             self.model.policy.load_state_dict(curr_policy)

            #             print("New Policy:", curr_policy['critic_target.qf0.4.bias'])
            #     except:
            #         pass
            if not self.is_baseline:
                try:
                    if self.n_calls % self.modify_epoch == 0:
                        if not self.modified:
                            self.model.policy.load_state_dict(torch.load(self.base_policy_loc))
                            self.modified = True

                        curr_policy = self.model.policy.state_dict()
                        tgt_policy = torch.load(self.easy_policy_loc)

                        for k in curr_policy.keys():
                            curr_policy[k] = curr_policy[k]*self.retain_ratio + tgt_policy[k]*(1-self.retain_ratio)
                        
                        self.model.policy.load_state_dict(curr_policy)
                except:
                    pass

            else:
                if not self.has_saved:
                    self.model.policy.save(self.base_policy_loc)
                    self.has_saved = True

            

        
        
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

"""
Planner Helpers
"""
@njit(fastmath=False, cache=True)
def nearest_point_on_trajectory(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.

    Same as nearest_point_on_line_segment, but vectorized. This method is quite fast, time constraints should
    not be an issue so long as trajectories are not insanely long.

        Order of magnitude: trajectory length: 1000 --> 0.0002 second computation (5000fps)

    point: size 2 numpy array
    trajectory: Nx2 matrix of (x,y) trajectory waypoints
        - these must be unique. If they are not unique, a divide by 0 error will destroy the world
    """
    diffs = trajectory[1:,:] - trajectory[:-1,:]
    l2s   = diffs[:,0]**2 + diffs[:,1]**2
    # this is equivalent to the elementwise dot product
    # dots = np.sum((point - trajectory[:-1,:]) * diffs[:,:], axis=1)
    dots = np.empty((trajectory.shape[0]-1, ))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0
    # t = np.clip(dots / l2s, 0.0, 1.0)
    projections = trajectory[:-1,:] + (t*diffs.T).T
    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))
    min_dist_segment = np.argmin(dists)
    return projections[min_dist_segment], dists[min_dist_segment], t[min_dist_segment], min_dist_segment

@njit(fastmath=False, cache=True)
def first_point_on_trajectory_intersecting_circle(point, radius, trajectory, t=0.0, wrap=False):
    """
    starts at beginning of trajectory, and find the first point one radius away from the given point along the trajectory.

    Assumes that the first segment passes within a single radius of the point

    http://codereview.stackexchange.com/questions/86421/line-segment-to-circle-collision-algorithm
    """
    start_i = int(t)
    start_t = t % 1.0
    first_t = None
    first_i = None
    first_p = None
    trajectory = np.ascontiguousarray(trajectory)
    for i in range(start_i, trajectory.shape[0]-1):
        start = trajectory[i,:]
        end = trajectory[i+1,:]+1e-6
        V = np.ascontiguousarray(end - start)

        a = np.dot(V,V)
        b = 2.0*np.dot(V, start - point)
        c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
        discriminant = b*b-4*a*c

        if discriminant < 0:
            continue
        #   print "NO INTERSECTION"
        # else:
        # if discriminant >= 0.0:
        discriminant = np.sqrt(discriminant)
        t1 = (-b - discriminant) / (2.0*a)
        t2 = (-b + discriminant) / (2.0*a)
        if i == start_i:
            if t1 >= 0.0 and t1 <= 1.0 and t1 >= start_t:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            if t2 >= 0.0 and t2 <= 1.0 and t2 >= start_t:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break
        elif t1 >= 0.0 and t1 <= 1.0:
            first_t = t1
            first_i = i
            first_p = start + t1 * V
            break
        elif t2 >= 0.0 and t2 <= 1.0:
            first_t = t2
            first_i = i
            first_p = start + t2 * V
            break
    # wrap around to the beginning of the trajectory if no intersection is found1
    if wrap and first_p is None:
        for i in range(-1, start_i):
            start = trajectory[i % trajectory.shape[0],:]
            end = trajectory[(i+1) % trajectory.shape[0],:]+1e-6
            V = end - start

            a = np.dot(V,V)
            b = 2.0*np.dot(V, start - point)
            c = np.dot(start, start) + np.dot(point,point) - 2.0*np.dot(start, point) - radius*radius
            discriminant = b*b-4*a*c

            if discriminant < 0:
                continue
            discriminant = np.sqrt(discriminant)
            t1 = (-b - discriminant) / (2.0*a)
            t2 = (-b + discriminant) / (2.0*a)
            if t1 >= 0.0 and t1 <= 1.0:
                first_t = t1
                first_i = i
                first_p = start + t1 * V
                break
            elif t2 >= 0.0 and t2 <= 1.0:
                first_t = t2
                first_i = i
                first_p = start + t2 * V
                break

    return first_p, first_i, first_t

@njit(fastmath=False, cache=True)
def get_actuation(pose_theta, lookahead_point, position, lookahead_distance, wheelbase):
    """
    Returns actuation
    """
    waypoint_y = np.dot(np.array([np.sin(-pose_theta), np.cos(-pose_theta)]), lookahead_point[0:2]-position)
    speed = lookahead_point[2]
    if np.abs(waypoint_y) < 1e-6:
        return speed, 0.
    radius = 1/(2.0*waypoint_y/lookahead_distance**2)
    steering_angle = np.arctan(wheelbase/radius)
    return speed, steering_angle

class PurePursuitPlanner:
    """
    Example Planner
    """
    def __init__(self, conf, wb, speed=1.0):
        self.wheelbase = wb
        self.conf = conf
        self.load_waypoints(conf)
        self.max_reacquire = 20.

        self.drawn_waypoints = []

        self.speed = speed

    def load_waypoints(self, conf):
        """
        loads waypoints
        """
        self.waypoints = np.loadtxt(conf.wpt_path, delimiter=conf.wpt_delim, skiprows=conf.wpt_rowskip)

    def render_waypoints(self, e):
        """
        update waypoints being drawn by EnvRenderer
        """

        #points = self.waypoints

        points = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        
        scaled_points = 50.*points

        for i in range(points.shape[0]):
            if len(self.drawn_waypoints) < points.shape[0]:
                b = e.batch.add(1, GL_POINTS, None, ('v3f/stream', [scaled_points[i, 0], scaled_points[i, 1], 0.]),
                                ('c3B/stream', [183, 193, 222]))
                self.drawn_waypoints.append(b)
            else:
                self.drawn_waypoints[i].vertices = [scaled_points[i, 0], scaled_points[i, 1], 0.]
        
    def _get_current_waypoint(self, waypoints, lookahead_distance, position, theta):
        """
        gets the current waypoint to follow
        """
        wpts = np.vstack((self.waypoints[:, self.conf.wpt_xind], self.waypoints[:, self.conf.wpt_yind])).T
        nearest_point, nearest_dist, t, i = nearest_point_on_trajectory(position, wpts)
        # print(f"position: {position}, nearest_point: {nearest_point}, nearest_dist: {nearest_dist}, t: {t}, i: {i}")
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            # current_waypoint[2] = waypoints[i, self.conf.wpt_vind]
            current_waypoint[2] = self.speed
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], self.speed)
        else:
            return None

    def plan(self, pose_x, pose_y, pose_theta, lookahead_distance, vgain):
        """
        gives actuation given observation
        """
        position = np.array([pose_x, pose_y])
        lookahead_point = self._get_current_waypoint(self.waypoints, lookahead_distance, position, pose_theta)

        if lookahead_point is None:
            return 4.0, 0.0

        speed, steering_angle = get_actuation(pose_theta, lookahead_point, position, lookahead_distance, self.wheelbase)
        speed = vgain * speed

        return speed, steering_angle


class FlippyPlanner:
    """
    Planner designed to exploit integration methods and dynamics.
    For testing only. To observe this error, use single track dynamics for all velocities >0.1
    """
    def __init__(self, speed=1, flip_every=1, steer=2):
        self.speed = speed
        self.flip_every = flip_every
        self.counter = 0
        self.steer = steer
    
    def render_waypoints(self, *args, **kwargs):
        pass

    def plan(self, *args, **kwargs):
        if self.counter%self.flip_every == 0:
            self.counter = 0
            self.steer *= -1
        return self.speed, self.steer


def main():
    """
    main entry point
    """

    work = {'mass': 3.463388126201571, 'lf': 0.15597534362552312, 'tlad': 0.82461887897713965, 'vgain': 1.375}#0.90338203837889}
    
    with open('config_example_map.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)
    conf = Namespace(**conf_dict)

    planner = PurePursuitPlanner(conf, (0.17145+0.15875),speed=3) #FlippyPlanner(speed=0.2, flip_every=1, steer=10)

    def render_callback(env_renderer):
        # custom extra drawing function

        e = env_renderer

        # update camera to follow car
        x = e.cars[0].vertices[::2]
        y = e.cars[0].vertices[1::2]
        top, bottom, left, right = max(y), min(y), min(x), max(x)
        e.score_label.x = left
        e.score_label.y = top - 700
        e.left = left - 800
        e.right = right + 800
        e.top = top + 800
        e.bottom = bottom - 800

        planner.render_waypoints(env_renderer)

    # env = gym.make('f110_gym:f110-v0', map=conf.map_path, map_ext=conf.map_ext, num_agents=1, timestep=0.01, integrator=Integrator.RK4)
    # env.add_render_callback(render_callback)
    map_config_1 = {
        'map_ext': '.png',
        'map': '/home/caluckal/Developer/spring2024/thesis/f1tenth_gym_custom/gym/f110_gym/unittest/maps/map_44_1000_',
        'waypoints': '/home/caluckal/Developer/spring2024/thesis/f1tenth_gym_custom/gym/f110_gym/unittest/centerline/map_44_1000_.csv',
        'reset_pose': [0.0,0.0,np.pi/5]
    }

    map_config_2 = {
        'map_ext': '.png',
        'map': '/home/caluckal/Developer/spring2024/thesis/f1tenth_gym_custom/gym/f110_gym/unittest/maps/map_7_100_',
        'waypoints': '/home/caluckal/Developer/spring2024/thesis/f1tenth_gym_custom/gym/f110_gym/unittest/centerline/map_7_100_.csv',
        'reset_pose': [0.0,0.0,0.0]
    }

    map_config_3 = {
        'map_ext': '.png',
        'map': '/home/caluckal/Developer/spring2024/thesis/f1tenth_gym_custom/gym/f110_gym/unittest/maps/map_15_100_',
        'waypoints': '/home/caluckal/Developer/spring2024/thesis/f1tenth_gym_custom/gym/f110_gym/unittest/centerline/map_15_100_.csv',
        'reset_pose': [0.0,0.0,0.0]
    }

    register('f110_gym:f110-cust-v0', entry_point='f110_gym.envs:F110_Cust_Env', max_episode_steps=10000)

    if args.config == 1:
        env = gym.make('f110_gym:f110-cust-v0',config=map_config_1, num_agents=1, timestep=0.01, integrator=Integrator.RK4, classic=False)
    elif args.config == 2:
        env = gym.make('f110_gym:f110-cust-v0',config=map_config_2, num_agents=1, timestep=0.01, integrator=Integrator.RK4, classic=False)
    elif args.config == 3:
        env = gym.make('f110_gym:f110-cust-v0',config=map_config_3, num_agents=1, timestep=0.01, integrator=Integrator.RK4, classic=False)

    from stable_baselines3 import PPO,SAC

    retain_string = int(args.retain*100)

    
    experiment_name = f"{args.config}_{args.car_idx}_{retain_string}_{args.exp}"


    model = SAC('MlpPolicy', env, verbose=1, tensorboard_log=f"logs/sac_{experiment_name}")

    custom_cb = CustomCallback(
        config_type=args.config, 
        easy_policy_loc=f"logs/policy_{args.exp}.pth",
        base_policy_loc=f"logs/base_{args.exp}.pth", 
        save_freq=args.save_freq,
        modify_epoch=args.modify_epoch,
        retain_ratio=args.retain,
        is_baseline=True if args.base == 1 else False
        )
    model.learn(total_timesteps=args.total_timesteps, callback=custom_cb)

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the f1tenth gym environment')
    parser.add_argument('--config', type=int, default=1, help='Config to run')
    parser.add_argument('--car_idx', type=int, default=1, help='Car index to use')
    parser.add_argument('--base',type=int,default=1,help='Base or custom callback')
    parser.add_argument('--retain',type=float,default=1,help='Retain ratio for custom callback')
    parser.add_argument('--exp',type=int,default=1,help='Experiment number')
    parser.add_argument('--save_freq',type=int,default=1000,help='Save frequency for custom callback')
    parser.add_argument('--modify_epoch',type=int,default=1,help='Modify epoch for custom callback')
    parser.add_argument('--total_timesteps',type=int,default=1e6,help='Total timesteps for training')

    args = parser.parse_args()

    print(args)

    main()
