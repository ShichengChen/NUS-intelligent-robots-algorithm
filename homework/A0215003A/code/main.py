from gym_duckietown.simulator import Simulator
import cv2
import numpy as np
import time
env = Simulator(
seed=123, # random seed
#map_name="loop_empty",
map_name="straight_road",
max_steps=500001, # we don't want the gym to reset itself
domain_rand=0,
camera_width=640,
camera_height=480,
accept_start_angle_deg=2, # start close to straight
full_transparency=False,
distortion=False,
user_tile_start=(20,0),
)
print('env.action_space',env.action_space)
print('env.action_space',env.action_space.high)
print('env.action_space',env.action_space.low)
preobs = env.reset()
num=0
state=[]
for _ in range(1000):
  env.render()
  #action = env.action_space.sample() # your agent here (this takes random actions)
  #print(action)
  action=np.array([1,1])
  #action=np.array([0.64,1])
  obs, reward, done, info = env.step(action)
  #print('reward',reward)
  #print(observation.shape)
  cv2.imshow("env", obs)
  mask=((obs[:,:,2]<obs[:,:,0])&(obs[:,:,2]<obs[:,:,1])).astype(np.uint8)
  close = cv2.morphologyEx(mask*255, cv2.MORPH_CLOSE, kernel = np.ones((20,20),np.uint8))
  #mask=cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
  #mask1=((obs[:,:,2]<50) & (obs[:,:,0]>50) & (obs[:,:,1]>50))
  cv2.imshow('mask',mask.astype(np.float32))
  cv2.imshow('close',close)
  state.append(cv2.resize(mask, (64,64)).astype(np.float32))
  preobs=obs.copy()
  cv2.imshow("env",obs)

  cv2.waitKey(1)
  if(_-num==200):done=True
  if done:
    print('number of step',_-num)
    print('done')
    print('reward',reward)
    observation = env.reset()
    num=_
env.close()
print('number of step',_-num)
imgs=np.array(state)
imgs=(imgs/imgs.max())
print(imgs.mean(axis=(0,1,2)))
print(imgs.std(axis=(0,1,2)))


'''
[0.24360505 0.23642332 0.22959285]
[0.41027179 0.40670541 0.40358895]

[0.24721774 0.23728222 0.23352849]
[0.41143565 0.40625676 0.40689975]
'''