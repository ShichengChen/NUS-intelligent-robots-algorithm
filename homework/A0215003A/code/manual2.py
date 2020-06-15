#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
from gym_duckietown.simulator import Simulator
import cv2
from gym_duckietown.envs import DuckietownEnv
# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckietown-udem1-v0')
#parser.add_argument('--map-name', default='udem1')
#parser.add_argument('--map-name', default='straight_road')
parser.add_argument('--map-name', default='zigzag_dists')
#parser.add_argument('--map-name', default='small_loop_cw')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        map_name='map5',
        domain_rand=False,
        draw_bbox=False,
        max_steps=2000,
        seed=2
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()

@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.unwrapped.cam_angle[0] = 0
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

accrew=0
def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    global a,b
    action = np.array([0.0, 0.0])
    global accrew
    if key_handler[key.UP]:
        action = np.array([4, 0.0])
        #action = np.array([2.0, 2.0])
    if key_handler[key.DOWN]:
        action = np.array([-1, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.98, 9.7])
    if key_handler[key.RIGHT]:
        action = np.array([0.98, -9.7])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])
    if key_handler[key.A]:
        a+=0.05
        print('a,b',a,b)
        #action=np.array([-0.1,0.1])
    if key_handler[key.D]:
        a -= 0.05
        print('a,b', a, b)
        #action=np.array([0.1,-0.1])
    if key_handler[key.Q]:
        b += 0.05
        print('a,b', a, b)
    if key_handler[key.E]:
        b-=0.05
        print('a,b', a, b)
    if key_handler[key.R]:
        accrew=0

    #print(action)
    # Speed boost
    if key_handler[key.LSHIFT]:
        action /= 2

    if ((action == np.array([0.0, 0.0])).all()):return
    obs, reward, done, info = env.step(action)
    if ((action == np.array([0.0, 0.0])).all()):
        pass
    else:
        accrew += reward

    print('reward',reward,accrew)
    mask1 = ((obs[:, :, 0]>150) &
             ((obs[:, :, 1]<50) & (obs[:, :, 1]>20)) &
             ((obs[:, :, 2]<50) & (obs[:, :, 2]>20)))
    # mask2 = ((obs[:, :, 0] > obs[:, :, 2]*1.25) & (obs[:, :, 1] > obs[:, :, 2]*1.25)
    #          & (obs[:, :, 2] < 160) & (obs[:, :, 2] >30) & (obs[:, :, 0]>150) & (obs[:, :, 1]>150) &
    #          (obs[:, :, 0] > obs[:, :, 1]))
    yw = (np.clip(mask1 * 1, 0, 1) * 255).astype(np.uint8)
    # yw = (np.clip(mask2 * 1.0, 0, 1) * 255).astype(np.uint8)
    yw = cv2.morphologyEx(yw, cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8))
    yw = cv2.morphologyEx(yw, cv2.MORPH_CLOSE, kernel=np.ones((30, 30), np.uint8))
    #
    # H,W=yw.shape[:2]
    # ywfull = yw.copy()
    # yw[:H//5*2,:]=0
    #
    # coor=np.argwhere(yw==255)
    # if(coor.size!=0):
    #     x1=coor[np.argmax(coor[:,0])]
    #     x0=coor[np.argmin(coor[:,0])]
    #     v01=x0-x1
    #     #print(x0, x1,v01)
    #     # for i in range(1,5):
    #     #     cv2.line(yw,tuple(x1[::-1]),tuple((x1+v01*i)[::-1]),(100),3)
    #     # cv2.line(yw,(W//2,H),(W//2,0),(100),3)
    # cannyed_image = cv2.Canny(yw, 100, 200)
    # lines = cv2.HoughLinesP(
    #     yw, rho=6,
    #     theta=np.pi / 60, threshold=200, lines=np.array([]), minLineLength=50, maxLineGap=40
    # )
    # onlyline=np.zeros_like(ywfull)
    # if(lines is not None):
    #     for line in lines:
    #         for x1, y1, x2, y2 in line:
    #             cv2.line(onlyline, (x1, y1), (x2, y2), (200), 3)
    #

    # cv2.imshow('obs', obs)
    # obs[mask2,0:2]=255
    # #mask=(obs[:,:,0]>150)&(obs[:,:,1]>150)&((obs[:,:,2]<100))
    #
    # gray_image = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
    #
    #cv2.imshow('img',cv2.cvtColor(obs, cv2.COLOR_BGR2RGB))
    # cv2.imshow('onlyline',onlyline)
    # #cv2.imshow('cannyed_image',cannyed_image)
    #cv2.imshow('mask1',yw)
    # cv2.imshow('cannyed_image',cannyed_image)
    #cv2.waitKey(1)


    #print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    if key_handler[key.RETURN]:
        from PIL import Image
        im = Image.fromarray(obs)

        im.save('screen.png')

    if done:
        accrew=0
        print('done!')
        env.reset()
        env.render()

    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
