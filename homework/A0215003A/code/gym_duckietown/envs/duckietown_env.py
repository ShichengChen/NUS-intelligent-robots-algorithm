# coding=utf-8
import numpy as np
from gym import spaces

from ..simulator import Simulator
from .. import logger


class DuckietownEnv(Simulator):
    """
    Wrapper to control the simulator using velocity and steering angle
    instead of differential drive motor velocities
    """

    def __init__(
        self,
        gain = 1.0,
        trim = 0.0,
        radius = 0.0318,
        k = 27.0,
        limit = 1.0,
        **kwargs
    ):
        Simulator.__init__(self, **kwargs)
        #logger.info('using DuckietownEnv')

        self.action_space = spaces.Box(
            low=np.array([-1,-1]),
            high=np.array([1,1]),
            dtype=np.float32
        )

        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit


    def trans(self,action):
        vel, angle = action
        baseline = self.unwrapped.wheel_dist
        k_r = self.k
        k_l = self.k
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l
        omega_r = (vel + 0.5 * angle * baseline) / self.radius
        omega_l = (vel - 0.5 * angle * baseline) / self.radius
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)
        vels = np.array([u_l_limited, u_r_limited])
        return vels


    def step(self, action):
        # import Const
        # if ((action != np.array([0.0, 0.0])).all()):
        #     mv,midx={},{}
        #     for k, v in Const.action.items():
        #         mv[k]=np.array([100,100])
        #         midx[k]=np.array([100,100])
        #     for i in range(0,400):
        #         for j in range(-400,401):
        #             action[0],action[1]=i/100,j/100
        #             trv=self.trans(action)
        #             for k, v in Const.action.items():
        #                 if np.sum(np.abs(trv-v))<np.sum(np.abs(mv[k]-v)):
        #                     mv[k]=trv.copy()
        #                     midx[k]=action.copy()
        #
        #     for k, v in Const.action.items():
        #         print('k',v,mv[k],midx[k])
        vel, angle = action

        # Distance between the wheels
        baseline = self.unwrapped.wheel_dist
        # assuming same motor constants k for both motors
        # gain = 1.0,
        # trim = 0.0,
        # radius = 0.0318,
        # k = 27.0,
        # limit = 1.0,

        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * baseline) / self.radius
        omega_l = (vel - 0.5 * angle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])

        # if((vels!=np.array([0.0,0.0])).all()):
        #     print('vels',vels)

        obs, reward, done, info = Simulator.step(self, vels)
        mine = {}
        mine['k'] = self.k
        mine['gain'] = self.gain
        mine['train'] = self.trim
        mine['radius'] = self.radius
        mine['omega_r'] = omega_r
        mine['omega_l'] = omega_l
        info['DuckietownEnv'] = mine
        return obs, reward, done, info


class DuckietownLF(DuckietownEnv):
    """
    Environment for the Duckietown lane following task with
    and without obstacles (LF and LFV tasks)
    """

    def __init__(self, **kwargs):
        DuckietownEnv.__init__(self, **kwargs)

    def step(self, action):
        obs, reward, done, info = DuckietownEnv.step(self, action)
        return obs, reward, done, info


class DuckietownNav(DuckietownEnv):
    """
    Environment for the Duckietown navigation task (NAV)
    """

    def __init__(self, **kwargs):
        self.goal_tile = None
        DuckietownEnv.__init__(self, **kwargs)

    def reset(self):
        DuckietownNav.reset(self)

        # Find the tile the agent starts on
        start_tile_pos = self.get_grid_coords(self.cur_pos)
        start_tile = self._get_tile(*start_tile_pos)

        # Select a random goal tile to navigate to
        assert len(self.drivable_tiles) > 1
        while True:
            tile_idx = self.np_random.randint(0, len(self.drivable_tiles))
            self.goal_tile = self.drivable_tiles[tile_idx]
            if self.goal_tile is not start_tile:
                break

    def step(self, action):
        obs, reward, done, info = DuckietownNav.step(self, action)

        info['goal_tile'] = self.goal_tile

        # TODO: add term to reward based on distance to goal?

        cur_tile_coords = self.get_grid_coords(self.cur_pos)
        cur_tile = self._get_tile(self.cur_tile_coords)

        if cur_tile is self.goal_tile:
            done = True
            reward = 1000

        return obs, reward, done, info
