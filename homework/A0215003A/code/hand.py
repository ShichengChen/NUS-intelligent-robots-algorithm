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
parser.add_argument('--map-name', default='map5')
parser.add_argument('--seed', type=int, default=1, help='random seed')
args = parser.parse_args()

#for testseed in range(2,11):
#for testseed in [1,4,5,6,7,8,9,10]:
#for testseed in [1, 2, 3, 4, 5, 6, 7, 8]: #2 5 6
#for testseed in [1, 2, 3, 4, 7, 8, 9, 10]: #3 9
Env.DEBUG=False
Env.SHOW=False
Env.WAIT=False
best=[0,48,11,22,4,128,6.9,46,24,100,264]
for g in range(50,69):
  Const.action[Const.stringToActionIdx['g']][0]=g*0.01
  for srg in range(5,15):
    Const.action[Const.stringToActionIdx['lggg']] = np.array([0.17,1.69])*(srg/10)
    Const.action[Const.stringToActionIdx['rggg']] = np.array([0.17,-1.69])*(srg/10)
    print('Const.action', g, srg / 10)
    for testseed in [7]:
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
      print('best', best[testseed])
      if(accrew>best[testseed]):
        best[testseed]=accrew

        print('better than best, and save',testseed)
        ans=[]
        for i in env.actions:
          ans.append(Const.action[i])
        np.savetxt('./'+str(args.map_name)+'_seed'+str(args.seed)+'.txt', np.array(ans), delimiter=',')