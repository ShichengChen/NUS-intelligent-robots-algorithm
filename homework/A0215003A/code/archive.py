from gym_duckietown.simulator import Simulator
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from Const import actionspace
import Const
import cv2
from torch.distributions import Categorical
from gym_duckietown.envs import DuckietownEnv
DEBUG=False

class cEnv(object):
    def __init__(self, arg,**kwargs):
        super(cEnv, self).__init__(**kwargs)
        self.env = DuckietownEnv(
            map_name=arg.map_name,
            domain_rand=False,
            draw_bbox=False,
            max_steps=2000,
            seed=arg.seed
        )
        print('arg',arg)
        self.arg=arg
        self.grayobs=None
        self.rgbobs=None
        self.actions=[]
        self.red=np.zeros([2100])
        self.turn=np.zeros([2100])
        self.straight=np.zeros([2100])
        self.lastloop=0
        self.accdegree=0
        self.turntimes=0
        self.beginlen=20
    def render(self):
        self.env.render()

    def getwhileLine(self,obs):
        h, w = obs.shape[:2]
        maskwhite = ((np.abs(obs[:, :, 0] - obs[:, :, 2]) < 10) &
                     (np.abs(obs[:, :, 0] - obs[:, :, 1]) < 10) &
                     (np.abs(obs[:, :, 1] - obs[:, :, 2]) < 10) &
                     (obs[:, :, 0] > 100))
        maskwhite[:h // 5 * 2, :] = False

        coorline = np.argwhere(maskwhite)
        if (coorline.size == 0):
            self.whitea =1/2
        else:
            a, b = np.sum(coorline[:, 1] <= w / 2), np.sum(coorline[:, 1] > w / 2)
            self.whitea =a / (a + b)



    def processForHandFeature(self,obs):
        cv2.imshow('obso', cv2.cvtColor(obs,cv2.COLOR_BGR2RGB))
        H, W = obs.shape[:2]

        if(self.arg.map_name[-1]=='3'):
            redmask = ((obs[:, :, 0] > 150) &
                     ((obs[:, :, 1] < 50) & (obs[:, :, 1] > 20)) &
                     ((obs[:, :, 2] < 50) & (obs[:, :, 2] > 20)))
            redy = (redmask * 255).astype(np.uint8)
            redy = cv2.morphologyEx(redy, cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8))
            redy = cv2.morphologyEx(redy, cv2.MORPH_CLOSE, kernel=np.ones((30, 30), np.uint8))
            redy[:H // 5 * 3, :] = 0
            if(np.sum(redy)>self.beginlen):
                if(DEBUG):print('red!!!!!!!!!!!!')
                self.red[len(self.actions)]=1

        mask2 = ((obs[:, :, 0] > obs[:, :, 2] * 1.25) & (obs[:, :, 1] > obs[:, :, 2] * 1.25)
                 & (obs[:, :, 2] < 160) & (obs[:, :, 2] > 30) &
                 (obs[:, :, 0] > 150) & (obs[:, :, 1] > 150) &
                 (obs[:, :, 0] > obs[:, :, 1]))
        #if(len(self.actions)<20):mask2=(mask2&(obs[:, :, 0] < 190))

        y0 = (np.clip(mask2 * 1.0, 0, 1) * Const.YellowLineColor).astype(np.uint8)
        y0 = cv2.morphologyEx(y0, cv2.MORPH_OPEN, kernel=np.ones((3, 3), np.uint8))
        y0 = cv2.morphologyEx(y0, cv2.MORPH_CLOSE, kernel=np.ones((30, 30 ), np.uint8))
        y0 = cv2.erode(y0, np.ones([3, 3]))
        ywfull0 = y0.copy()
        #y0[:H // 5 * 2, :] = 0
        y0[:H // 2, :] = 0

        yw = np.zeros_like(y0)
        ywfull = np.zeros_like(ywfull0)
        # if (np.sum(y0[:, :-Const.shift]) == 0):
        #     self.turn[len(self.actions)] = 1
        if (np.sum(self.turn[len(self.actions) - 1:len(self.actions) + 1]) > 0):
            if(DEBUG):print('no shift')
            yw[:, :] = y0[:, :]
            ywfull[:, :] = ywfull0[:, :]
        else:
            yw[:, Const.shift:] = y0[:, :-Const.shift]
            ywfull[:, Const.shift:] = ywfull0[:, :-Const.shift]


        lines = cv2.HoughLinesP(yw, rho=6,
             theta=np.pi / 60, threshold=160, lines=np.array([]), minLineLength=50, maxLineGap=40)
        if (lines is not None):
            for line in lines:
                for x1, y1, x2, y2 in line:
                    cv2.line(ywfull, (x1, y1), (x2, y2), (Const.HoughLineColor), 3)

        cv2.imshow('ywfull', cv2.resize(ywfull, (640, 480)))
        cv2.imshow('yw', cv2.resize(yw, (640, 480)))
        cv2.waitKey(1)
        yw = cv2.resize(yw, Const.envshape)
        ywfull = cv2.resize(ywfull, Const.envshape)
        if (np.sum(yw) == 0): self.getwhileLine(obs)
        self.yw = yw.copy()
        self.ywfull=ywfull.copy()
        return yw.copy()
    def step(self,action):
        obs, reward, done, info=self.env.step(action)
        obs=self.processForHandFeature(obs)
        cv2.waitKey(1)
        return obs, reward, done, info
    def reset(self):
        self.actions=[]
        return self.env.reset()
    def close(self):
        self.env.close()
    def start(self):
        obs = self.processForHandFeature(self.env.reset().copy())
        return obs

    def checkOnlyRotation(self):
        succ = True
        if (len(self.actions) > self.beginlen):
            for i in range(len(self.actions) - 8, len(self.actions)):
                if (self.actions[i] != Const.stringToActionIdx['ol'] and self.actions[i] != Const.stringToActionIdx[
                    'or']):
                    succ = False
        else:
            succ=False
        return succ

    def setDegree(self,cdegree):
        self.accdegree+=cdegree

    def checkstraight(self):
        if(len(self.actions)<20):return False
        pro=np.sum(self.straight[len(self.actions)-20:len(self.actions)])/20
        #print('pro',pro)
        if(pro>0.70):return True

    def updateStraight(self):
        if ((self.actions[-1]) == (Const.stringToActionIdx['ggg'])) or \
        ((self.actions[-1]) == (Const.stringToActionIdx['gg'])) or \
        ((self.actions[-1]) == (Const.stringToActionIdx['g'])):
            self.straight[len(self.actions)]=1


    def handFeature(self,img):
        if(self.accdegree>90 and (self.arg.map_name[-1] != '1') and self.checkstraight()):
            self.turntimes=17
            self.accdegree=-90
            self.red=np.zeros([2100])
        if (self.arg.map_name[-1] == '1' and 1050 == len(self.actions)):
            self.turntimes=16
        if self.turntimes>0:
            self.turntimes-=1
            #return Const.stringToActionIdx['llll']
            return Const.stringToActionIdx['rrrr']
        if(np.sum(self.yw)==0 and len(self.actions)<self.beginlen):
            if(DEBUG):print('at the beginning, only or')
            #return Const.stringToActionIdx['llll']
            return Const.stringToActionIdx['rrrr']
        if self.checkOnlyRotation():
            return Const.stringToActionIdx['g']
        h, w = img.shape[:2]
        coor = np.argwhere(img==Const.YellowLineColor)
        w = Const.envshape[0]
        if np.sum(self.yw)<2:
            if (np.sum(self.red[len(self.actions) - 10:len(self.actions) + 1]) > 0):
                if(DEBUG):print('red ggg')
                return Const.stringToActionIdx['ggg']
        if (coor.size == 0):
            if (self.whitea>1/2):
                return Const.stringToActionIdx['srg']
            else:
                return Const.stringToActionIdx['slg']
            # if (self.whitea>3/5):
            #     return Const.stringToActionIdx['srg']
            # elif(self.whitea<2/5):
            #     return Const.stringToActionIdx['slg']
            # else:
            #     return Const.stringToActionIdx['g']
        xmax = np.max(coor[:, 1])
        xmin = np.min(coor[:, 1])
        #print('xmax',xmax,xmin)
        b, a = max(xmax - w / 2, 0), max(w / 2 - xmin, 0)
        x0, x1 = coor[np.argmax([coor[:, 0]])], coor[np.argmin([coor[:, 0]])]
        midx = (x0 + x1) / 2
        midx = midx.astype(int)
        k = abs(x0[0] - x1[0]) / (abs(x0[1] - x1[1]) + 1e-6)
        # cv2.circle(self.ywfull,(midx[1],midx[0]),20,255)
        # cv2.imshow('ywfull',self.ywfull)
        ywfullcopy=self.ywfull.copy()
        ywfullcopy[max(midx[0] - 3, 0):min(midx[0] + 3, h - 1),
        max(midx[1] - 3, 0):min(midx[1] + 3, w - 1)]=255
        cv2.line(ywfullcopy,(0,int(h/5*4)),(64,int(h/5*4)),(255),1)
        cv2.imshow('rec',ywfullcopy)
        cv2.waitKey(1)
        straightline = np.array([(self.ywfull[max(midx[0] - 3, 0):min(midx[0] + 3, h - 1),
            max(midx[1] - 3, 0):min(midx[1] + 3, w - 1)] == Const.HoughLineColor).any(),
                                 (k > 1),np.min(coor[:, 0])<int(h/5*4)])
        if(DEBUG):print(a, b, straightline)
        c = a / (a + b+1e-5)

        ans = Const.stringToActionIdx['g']
        # if (straightline.all() and w*4/9<x0[0]<w*5/9 and c<1/2):
        #     ans=Const.stringToActionIdx['or']
        # elif (straightline.all() and w*4/9<x0[0]<w*5/9 and c>=1/2):
        #     ans = Const.stringToActionIdx['ol']

        if (w * 2 / 7 < xmin and xmax < w * 5 / 7 and straightline.all()):
            ans = Const.stringToActionIdx['ggg']
        elif (w * 1 / 3 < xmin and xmax < w * 2 / 3 and straightline.all()):
            ans = Const.stringToActionIdx['ggg']
        elif (coor.shape[0]<15 and np.abs(xmax - xmin) < 5 and w * 2 / 5 < (xmax + xmin) / 2 < w * 3 / 5):
            ans = Const.stringToActionIdx['ggg']
        elif (coor.shape[0]<15 and np.abs(xmax - xmin) < 5 and (xmax + xmin) / 2 < w * 2 / 5):
            if(DEBUG):print((xmax + xmin) / 2)
            if(DEBUG):print('follow point')
            ans = Const.stringToActionIdx['llg']
        elif (coor.shape[0]<15 and np.abs(xmax - xmin) < 5 and w * 3 / 5 < (xmax + xmin) / 2):
            if(DEBUG):print((xmax + xmin) / 2)
            if(DEBUG):print('follow point')
            ans = Const.stringToActionIdx['rrg']
        # elif(straightline.all() and c <= 1 / 2):
        #     print('rgg')
        #     ans=Const.stringToActionIdx['rgg']
        # elif(straightline.all() and c > 1 / 2):
        #     print('lgg')
        #     ans=Const.stringToActionIdx['lgg']
        elif (c <= 1 / 3):
            #ans = Const.stringToActionIdx['or']
            ans = Const.stringToActionIdx['srg']
        elif (1 / 3 < c <= 2 / 3):
            ans = Const.stringToActionIdx['g']
        elif (2 / 3 < c):
            #ans = Const.stringToActionIdx['ol']
            ans = Const.stringToActionIdx['slg']
        if(DEBUG):print('action', Const.action[ans])
        return ans