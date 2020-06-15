import torch
from gym_duckietown.simulator import Simulator
import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from Env import cEnv
import Env
from tqdm import tqdm
import argparse
from Const import actionspace
import Const
from torch.distributions import Categorical

parser = argparse.ArgumentParser()
parser.add_argument('--map-name', default='map4')
parser.add_argument('--seed', type=int, default=1, help='random seed')
args = parser.parse_args()

#for testseed in range(2,11):
#for testseed in [1,4,5,6,7,8,9,10]:
#for testseed in [1, 2, 3, 4, 5, 6, 7, 8]: #2 5 6
#for testseed in [1, 2, 3, 4, 7, 8, 9, 10]: #3 9
ma={
"map1": [2, 3, 5, 6, 9, 10],
"map2": [1, 2, 3, 4, 5, 6, 7, 8],
#"map2": [2],
"map3": [1, 2, 3, 4, 7, 8, 9, 10],
"map4": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
"map5": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
}
best={'map5':{1:452,2:357,3:498,4:245,5:250,6:270,7:180,8:310,9:211,10:295},
      'map4':{1:555,2:770,3:722,4:717,5:684,6:704,7:656,8:645,9:748,10:627},
      'map3':{1:632,2:594,3:328,4:579,7:459,8:591,9:426,10:610},
      'map2':{1:1097,2:1078,3:1099,4:1094,5:994,6:960,7:1051,8:986},
      'map1':{2:1741,3:1695,5:1736,6:1720,9:1659,10:1772}}
Const.action[Const.stringToActionIdx['g']][0]=0.55
ALL=False
Env.DEBUG=ALL
Env.SHOW=ALL
Env.WAIT=ALL
for mapidx in range(2,3):
  args.map_name = 'map'+str(mapidx)
  for testseed in ma[args.map_name]:
    args.seed=testseed
    env=cEnv(args)
    obs = env.start()
    accrew=0
    degree=0
    for epoch in range(2000):
      #env.render()
      aidx=env.handFeature(obs)
      env.actions.append(aidx)
      env.updateStraight()
      action = Const.action[aidx]
      if(epoch>20):
        degree+=action[1]
        env.accdegree+=action[1]
      obs, reward, done, info = env.step(action)
      accrew+=reward
      if(Env.DEBUG):print('reward',reward,degree)
      #if(Env.DEBUG):print('accrew',accrew)
      #print('accrew', accrew)
      if(done):
        cv2.waitKey(1)
        break

    print('last accrew',accrew)
    env.close()
    if(accrew>best[args.map_name][testseed]):
      best[args.map_name][testseed]=accrew

      print('better than best, and save',testseed)
      ans=[]
      for i in env.actions:
        ans.append(Const.action[i])
      np.savetxt('./'+str(args.map_name)+'_seed'+str(args.seed)+'.txt', np.array(ans), delimiter=',')