#!/usr/bin/env python
import rospy
import numpy as np

from geometry_msgs.msg import *
from nav_msgs.msg import *
from sensor_msgs.msg import *
from const import *
from math import *
import copy
import argparse
import time
import json

ROBOT_SIZE = 0.2552  # width and height of robot in terms of stage unit


def dump_action_table(action_table, filename):
    """dump the MDP policy into a json file

    Arguments:
        action_table {dict} -- your mdp action table. It should be of form {'1,2,0': (1, 0), ...}
        filename {str} -- output filename
    """
    tab = dict()
    for k, v in action_table.items():
        key = [str(i) for i in k]
        key = ','.join(key)
        tab[key] = v

    with open(filename, 'w') as fout:
        json.dump(tab, fout)


class Planner:
    def __init__(self, world_width, world_height, world_resolution, inflation_ratio=3):
        """init function of the base planner. You should develop your own planner
        using this class as a base.

        For standard mazes, width = 200, height = 200, resolution = 0.05. 
        For COM1 map, width = 2500, height = 983, resolution = 0.02

        Arguments:
            world_width {int} -- width of map in terms of pixels
            world_height {int} -- height of map in terms of pixels
            world_resolution {float} -- resolution of map

        Keyword Arguments:
            inflation_ratio {int} -- [description] (default: {3})
        """
        rospy.init_node('planner')
        self.map = None
        self.pose = None
        self.goal = None
        self.path = None
        self.action_seq = None  # output
        self.aug_map = None  # occupancy grid with inflation
        self.action_table = {}

        self.world_width = world_width
        self.world_height = world_height
        self.resolution = world_resolution
        self.inflation_ratio = inflation_ratio
        self.map_callback()
        self.sb_obs = rospy.Subscriber('/scan', LaserScan, self._obs_callback)
        self.sb_pose = rospy.Subscriber(
            '/base_pose_ground_truth', Odometry, self._pose_callback)
        self.sb_goal = rospy.Subscriber(
            '/move_base_simple/goal', PoseStamped, self._goal_callback)
        self.controller = rospy.Publisher(
            '/mobile_base/commands/velocity', Twist, queue_size=10)
        rospy.sleep(1)

        #print('self.map',rospy.wait_for_message('/map', OccupancyGrid).data)

    def map_callback(self):
        """Get the occupancy grid and inflate the obstacle by some pixels. You should implement the obstacle inflation yourself to handle uncertainty.
        """
        self.map = rospy.wait_for_message('/map', OccupancyGrid).data
        self.map = np.array(self.map).reshape(self.world_height,self.world_width)

        # TODO: FILL ME! implement obstacle inflation function and define self.aug_map = new_mask

        # you should inflate the map to get self.aug_map
        #self.aug_map = copy.deepcopy(self.map)
        self.aug_map=self.map.copy()
        b=self.inflation_ratio
        for y in range(self.map.shape[0]):
            for x in range(self.map.shape[1]):
                if(self.map[y,x]==100):
                    self.aug_map[y-b:y+b,x-b:x+b]=100

    def _pose_callback(self, msg):
        """get the raw pose of the robot from ROS

        Arguments:
            msg {Odometry} -- pose of the robot from ROS
        """
        self.pose = msg

    def _goal_callback(self, msg):
        self.goal = msg
        self.generate_plan2()

    def _get_goal_position(self):
        goal_position = self.goal.pose.position
        return (goal_position.x, goal_position.y)

    def set_goal(self, x, y, theta=0):
        """set the goal of the planner

        Arguments:
            x {int} -- x of the goal
            y {int} -- y of the goal

        Keyword Arguments:
            theta {int} -- orientation of the goal; we don't consider it in our planner (default: {0})
        """
        a = PoseStamped()
        a.pose.position.x = x
        a.pose.position.y = y
        a.pose.orientation.z = theta
        self.goal = a

    def _obs_callback(self, msg):
        """get the observation from ROS; currently not used in our planner; researve for the next assignment

        Arguments:
            msg {LaserScan} -- LaserScan ROS msg for observations
        """
        self.last_obs = msg

    def _d_from_goal(self, pose):
        """compute the distance from current pose to the goal; only for goal checking

        Arguments:
            pose {list} -- robot pose

        Returns:
            float -- distance to the goal
        """
        goal = self._get_goal_position()
        return sqrt((pose[0] - goal[0])**2 + (pose[1] - goal[1])**2)

    def _check_goal(self, pose):
        """Simple goal checking criteria, which only requires the current position is less than 0.25 from the goal position. The orientation is ignored

        Arguments:
            pose {list} -- robot post

        Returns:
            bool -- goal or not
        """
        if self._d_from_goal(pose) < 0.25:
            return True
        else:
            return False

    def create_control_msg(self, x, y, z, ax, ay, az):
        """a wrapper to generate control message for the robot.

        Arguments:
            x {float} -- vx
            y {float} -- vy
            z {float} -- vz
            ax {float} -- angular vx
            ay {float} -- angular vy
            az {float} -- angular vz

        Returns:
            Twist -- control message
        """
        message = Twist()
        message.linear.x = x
        message.linear.y = y
        message.linear.z = z
        message.angular.x = ax
        message.angular.y = ay
        message.angular.z = az
        return message

    def check(self, x, y, nx, ny, map, re,verb=False):
        nx, ny = int(nx / re), int(ny / re)
        x, y = int(x / re), int(y / re)
        #if (verb):
        #    print(nx,ny,x,y)
        #    print(self.world_height,self.world_width)
        if not (0 <= ny < self.world_height and 0 <= nx < self.world_width): return False
        if not (0 <= y < self.world_height and 0 <= x < self.world_width): return False
        if(x==0 or y==0 or nx==0 or ny==0):return False
        if(map[y,x]==100 or map[ny,nx]==100):return False

        if (y == ny):
            for cx in range(min(x, nx), max(x, nx) + 1):
                if (map[y, cx] == 100):
                    return False
        if (x == nx):
            for cy in range(min(y, ny), max(y, ny) + 1):
                if (map[cy, x] == 100):
                    return False
        if(x!=nx and y != ny):
            for cx in range(min(x, nx), max(x, nx) + 1):
                if (map[y, cx] == 100 or map[ny, cx] == 100):
                    return False
            for cy in range(min(y, ny), max(y, ny) + 1):
                if (map[cy, x] == 100 or map[cy, nx] == 100):
                    return False
        if re==1:return map[ny,nx]!=100 and map[y,nx]!=100 and map[ny,x]!=100
        return True


    def generate_plan1(self):
        # a.pose.position.x = x
        # a.pose.position.y = y
        # a.pose.orientation.z
        res = self.resolution
        d = -np.ones((np.ceil(self.world_height * res), np.ceil(self.world_width * res)),
                     dtype=np.int64)
        que = []
        que.append((self.goal.pose.position.x, self.goal.pose.position.y, 0))
        print('goal', self.goal.pose.position.y, self.goal.pose.position.x)
        d[self.goal.pose.position.y, self.goal.pose.position.x] = 0
        dir = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while (que != []):
            cur = que.pop(0)
            cx, cy, cw = cur

            #print(cur)
            for dx,dy in dir:
                nx, ny = cx + dx, cy + dy
                if (cx == 21 and cy == 14):
                    print(nx,ny)
                    print(self.check(cx, cy, nx, ny, self.aug_map,res))
                if (self.check(cx, cy, nx, ny, self.aug_map,res) and d[ny, nx] == -1):
                    d[ny, nx] = cw + 1
                    que.append((nx, ny, cw + 1))
        print(d)
        print(self.aug_map[::10, ::10])
        self.action_seq = []
        self.d = d
        start = self.get_current_discrete_state()
        cx, cy, cd = start
        turn = {"l": (0, 1), "r": (0, -1)}
        print(cx, cy, d[cy, cx])
        ma = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
        print('start', cx, cy, d[cy, cx])
        # 0 east 1 north 2 west 3 south
        while d[cy, cx] != 0:
            for idx,(dx,dy) in enumerate(dir):
                nx, ny = cx + dx, cy + dy
                if (self.check(cx, cy, nx, ny, self.aug_map,res) and d[ny, nx] == d[cy, cx] - 1):
                    print(nx, ny, d[ny, nx])
                    mx, my = nx - cx, ny - cy
                    ad = ma[(mx, my)]
                    cntl = ((ad - cd) + 4) % 4
                    cntr = 4 - cntl
                    cd = ad
                    if (cntl <= cntr):
                        print('cntl', cntl)
                        for i in range(cntl): self.action_seq.append(turn['l'])
                    else:
                        print('cntr', cntr)
                        for i in range(cntr): self.action_seq.append(turn['r'])
                    self.action_seq.append((1, 0))
                    cx, cy = nx, ny
                    break
                assert (idx != 3)
        print(self.action_seq)
    def generate_plan2(self):
        """TODO: this is 2222222222222222222222222222222222
        """

        res = self.resolution
        multi=0.1
        multi=1
        #to much states, move so slow in simulator, could tune this parameters
        #if follow the report the variable multi should be 1
        cuben = int(1 / res)
        print('res', res, 'ncude', cuben)

        d = ((1e9))*np.ones((np.ceil(self.world_height*multi), np.ceil(self.world_width*multi)),dtype=np.float64)
        pre = -np.ones((np.ceil(self.world_height*multi), np.ceil(self.world_width*multi),2),dtype=np.int64)


        print('d.shape', d.shape)

        import heapq
        goalx, goaly = int(self.goal.pose.position.x / res*multi), int(self.goal.pose.position.y / res*multi)
        que = []
        heapq.heappush(que, (0, goalx, goaly))
        d[goaly, goalx] = 0
        print('goal', goaly, goalx)
        pre[goaly, goalx]=(-1,-1)
        dir = [(1, 0), (-1, 0), (0, 1), (0, -1),(1,1),(1,-1),(-1,-1),(-1,1),
        (2, 1), (1, 2), (2, -1), (1, -2), (-2, -1), (-1, -2), (-1, 2), (-2, 1)]
        #it is very easy to add more direction here to make the path shorter
        while (que != []):
            cw, cx, cy  = heapq.heappop(que)
            #print(cx, cy, cw)
            if(d[cy,cx] < cw):continue
            for dx,dy in dir:
                nx, ny = cx + dx, cy + dy
                #print('nx,ny',nx,ny)
                w = np.sqrt(dx**2+dy**2)
                if (self.check(cx, cy, nx, ny, self.aug_map,multi) and d[ny, nx] > cw+w):
                    d[ny, nx] = cw + w
                    heapq.heappush(que, (d[ny, nx], nx, ny))
                    pre[ny,nx]=(cy,cx)

        self.action_seq = []
        self.d = d

        start = self.get_current_discrete_state()
        cx, cy, cd = int(start[0] / self.resolution*multi), int(start[1] / self.resolution*multi), int(start[2])*2
        cdx,cdy=1,0
        cd = np.arctan2(cdy, cdx) % (2 * np.pi)
        print('start',cx, cy, d[cy, cx])
        # 0 east 2 north 4 west 6 south
        forward=(1 / (1 / res / 2) / multi)
        while not (pre[cy, cx] == np.array([-1, -1])).all():
            prey,prex=pre[cy,cx]
            print('pre[cy,cx]',pre[cy,cx][0]/multi*res,pre[cy,cx][1]/multi*res)
            ax, ay = prex - cx, prey - cy
            ad=np.arctan2(ay,ax)
            dif=(ad-cd+2*np.pi)%(2*np.pi)
            cd=ad
            if(abs(dif)>1e-3):
                for i in range(4):self.action_seq.append((0,abs(dif)/4*2))
            for i in range(3):
                print(ax,ay)
                self.action_seq.append((np.sqrt(ax**2+ay**2)*forward/3,0))
            cy,cx=prey,prex
            assert self.check(cx, cy, prex, prey, self.aug_map,multi)

        import cv2
        img = self.aug_map.copy()
        cx, cy = int(start[0] / self.resolution * multi), int(start[1] / self.resolution * multi)
        while not (pre[cy, cx] == np.array([-1, -1])).all():
            prey, prex = pre[cy, cx]
            img[(int(cy / multi), int(cx / multi))] = 255
            cy, cx = prey, prex
        cv2.imwrite('result.jpg', img)

    def generate_plan3(self):
        """TODO: this is 333333333333333333333333333333333333333333

        """
        res = self.resolution
        dh,dw=int(np.ceil(self.world_height*res)), int(np.ceil(self.world_width*res))
        dhsafe,dwsafe=int(np.ceil(self.world_height*res))+3, int(np.ceil(self.world_width*res))+3
        pre = -np.ones((np.ceil(self.world_height), np.ceil(self.world_width), 2), dtype=np.int64)
        wd = np.zeros((dhsafe, dwsafe), dtype=np.float64)
        print('dh,dw',dh,dw)
        print('res',res)
        for y in range(dh):
            for x in range(dw):
                if self.check(x, y, x, y, self.aug_map, res) == False:
                    wd[y, x] = 1
                    continue
                hw=0
                c=1.1
                for dx, dy in [(c, 0), (-c, 0), (0, c), (0, -c),(c,c),(-c,-c),(c,-c),(-c,c)]:
                    if self.check(x,y,x+dx,y+dy,self.aug_map,res)==False:
                        #print(x,y,x+dx,y+dy)
                        hw=1
                wd[y,x]=hw


        dir = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        goalx, goaly = int(self.goal.pose.position.x), int(self.goal.pose.position.y)
        d = np.ones((dhsafe, dwsafe), dtype=np.float64) * (1e8)
        d[goaly, goalx] = 0
        print('goal', goaly, goalx)
        for iter in range(dhsafe*dwsafe):
            change=0
            for y in range(dh):
                for x in range(dw):
                    if(self.check(x,y,x,y,self.aug_map,res)==False or (y==goaly and x==goalx)):continue
                    for dx,dy in dir:
                        if (self.check(x, y, x + dx, y + dy, self.aug_map, res) == False): continue
                        ny=[y + dy,y + dy,y + dy]
                        nx=[x + dx,x + dx,x + dx]
                        if(dx==0):
                            ny[1]+=1
                            ny[2]-=1
                        if(dy==0):
                            nx[1]+=1
                            nx[2]-=1
                        #if (self.check(x, y, x + dx, y + dy, self.aug_map, res) == False):
                        reward = [1e4 if wd[ny[i],nx[i]]==1 else 1 for i in range(3)]
                        d0=((d[ny[0],nx[0]])*0.99+reward[0])
                        d1=np.sum((d[ny[1:],nx[1:]])*0.01)
                        #not the exact probability, need to tune hyperparamters here to get better results.

                        #d1=np.sum([1e4 if d[ny[i],nx[i]]==1e8 else d[ny[i],nx[i]] for i in range(1,3)])*0.1
                        #eliminate the influence of inf!

                        # d0=((d[ny[0],nx[0]]+reward[0])*0.9)
                        # d1=np.sum((d[ny[1:],nx[1:]]+reward[1:])*0.1)
                        if(d[y,x]>d0+d1):#get min positive reward is same with get max negtive reward
                            change=1
                            d[y, x] = d0 + d1
                            pre[y,x]=(ny[0],nx[0])
            if(change==0):
                print('iter num',iter)
                break

        #print('d[7,31]',d[7,31])
        start = self.get_current_discrete_state()
        cx, cy, cd = start
        import cv2
        img=self.aug_map.copy()
        while not (pre[cy, cx] == np.array([-1, -1])).all():
            prey,prex=pre[cy,cx]
            img[(int(cy/res),int(cx/res))]=255
            cy,cx=prey,prex
        cv2.imwrite('result.jpg',img)
        #np.set_printoptions(threshold=np.inf)
        #print(d)
        self.d = d
        print('d[10:15,4:7]',d[10:15,4:7])
        self.action_table={}
        ma = {(1, 0): 0, (0, 1): 1, (-1, 0): 2, (0, -1): 3}
        turn = {'l':(0, 1),'r':(0, -1)}
        forward=(1,0)
        start = self.get_current_discrete_state()
        cx, cy, cd = start
        print('start',cx, cy, d[cy, cx])
        # 0 east 1 north 2 west 3 south
        for y in range(dh):
            for x in range(dw):
                if (self.check(x, y, x, y, self.aug_map, res)==False or (pre[y,x]==np.array([-1, -1])).all()):continue
                prey, prex = pre[y, x]
                #print('pre[y, x]', pre[y, x])
                #print('y,x', y,x)
                mx, my = prex - x, prey - y
                assert abs(mx)!=abs(my)
                ad = ma[(mx, my)]
                for cd in range(4):
                    if (ad == cd):
                        self.action_table[(x,y,cd)]=forward
                    else:
                        cntl = ((ad - cd) + 4) % 4
                        cntr = 4 - cntl
                        if(cntl<=cntr):self.action_table[(x,y,cd)]=turn['l']
                        else:self.action_table[(x,y,cd)]=turn['r']


    def get_current_state(self):
        return self.get_current_discrete_state()
    def get_current_continuous_state(self):
        """Our state is defined to be the tuple (x,y,theta). 
        x and y are directly extracted from the pose information. 
        Theta is the rotation of the robot on the x-y plane, extracted from the pose quaternion. For our continuous problem, we consider angles in radians

        Returns:
            tuple -- x, y, \theta of the robot
        """
        x = self.pose.pose.pose.position.x
        y = self.pose.pose.pose.position.y
        orientation = self.pose.pose.pose.orientation
        ori = [orientation.x, orientation.y, orientation.z,
               orientation.w]

        phi = np.arctan2(2 * (ori[0] * ori[1] + ori[2] * ori[3]), 1 - 2 *
                         (ori[1] ** 2 + ori[2] ** 2))
        return (x, y, phi)

    def get_current_discrete_state(self):
        """Our state is defined to be the tuple (x,y,theta). 
        x and y are directly extracted from the pose information. 
        Theta is the rotation of the robot on the x-y plane, extracted from the pose quaternion. For our continuous problem, we consider angles in radians

        Returns:
            tuple -- x, y, \theta of the robot in discrete space, e.g., (1, 1, 1) where the robot is facing north
        """
        x, y, phi = self.get_current_continuous_state()
        def rd(x): return int(round(x))
        return rd(x), rd(y), rd(phi / (np.pi / 2))
        #return rd(x), rd(y), phi

    def collision_checker(self, x, y):
        """TODO: FILL ME!
        You should implement the collision checker.
        Hint: you should consider the augmented map and the world size
        
        Arguments:
            x {float} -- current x of robot
            y {float} -- current y of robot
        
        Returns:
            bool -- True for collision, False for non-collision
        """
        if(self.map[y,x]==100):return True
        return False

    def motion_predict(self, x, y, theta, v, w, dt=0.5, frequency=10):
        """predict the next pose of the robot given controls. Returns None if the robot collide with the wall
        The robot dynamics are provided in the homework description

        Arguments:
            x {float} -- current x of robot
            y {float} -- current y of robot
            theta {float} -- current theta of robot
            v {float} -- linear velocity 
            w {float} -- angular velocity

        Keyword Arguments:
            dt {float} -- time interval. DO NOT CHANGE (default: {0.5})
            frequency {int} -- simulation frequency. DO NOT CHANGE (default: {10})

        Returns:
            tuple -- next x, y, theta; return None if has collision
        """
        num_steps = int(dt * frequency)
        dx = 0
        dy = 0
        for i in range(num_steps):
            if w != 0:
                dx = - v / w * np.sin(theta) + v / w * \
                    np.sin(theta + w / frequency)
                dy = v / w * np.cos(theta) - v / w * \
                    np.cos(theta + w / frequency)
            else:
                dx = v*np.cos(theta)/frequency
                dy = v*np.sin(theta)/frequency
            x += dx
            y += dy

            if self.collision_checker(x, y):
                return None
            theta += w / frequency
        return x, y, theta

    def discrete_motion_predict(self, x, y, theta, v, w, dt=0.5, frequency=10):
        """discrete version of the motion predict. Note that since the ROS simulation interval is set to be 0.5 sec
        and the robot has a limited angular speed, to achieve 90 degree turns, we have to execute two discrete actions
        consecutively. This function wraps the discrete motion predict.

        Please use it for your discrete planner.

        Arguments:
            x {int} -- current x of robot
            y {int} -- current y of robot
            theta {int} -- current theta of robot
            v {int} -- linear velocity
            w {int} -- angular velocity (0, 1, 2, 3)

        Keyword Arguments:
            dt {float} -- time interval. DO NOT CHANGE (default: {0.5})
            frequency {int} -- simulation frequency. DO NOT CHANGE (default: {10})

        Returns:
            tuple -- next x, y, theta; return None if has collision
        """
        w_radian = w * np.pi/2
        first_step = self.motion_predict(x, y, theta*np.pi/2, v, w_radian)
        if first_step:
            second_step = self.motion_predict(
                first_step[0], first_step[1], first_step[2], v, w_radian)
            if second_step:
                return (round(second_step[0]), round(second_step[1]), round(second_step[2] / (np.pi / 2)) % 4)
        return None

    def publish_control(self):
        """publish the continuous controls
        """
        for action in self.action_seq:
            msg = self.create_control_msg(action[0], 0, 0, 0, 0, action[1])
            print(action)
            print(np.rad2deg(action[1]))
            self.controller.publish(msg)
            rospy.sleep(0.6)
            print(self.get_current_continuous_state())

        result = np.array(planner.action_seq)
        np.savetxt('task2/2_maze_{}_{}.txt'.format(self.goal.pose.position.x,self.goal.pose.position.y),
                   result)

    def publish_discrete_control(self):
        """publish the discrete controls
        """
        print('start',self.get_current_continuous_state())
        # (0, 1) left
        # (0, -1) RIGHT

        for action in self.action_seq:
            current_state = self.get_current_state()
            print('d[]',self.d[current_state[1],current_state[0]])
            msg = self.create_control_msg(
                action[0], 0, 0, 0, 0, action[1]*np.pi/2)
            self.controller.publish(msg)
            rospy.sleep(0.6)
            #print(self.get_current_continuous_state())
            #print(self.get_current_discrete_state())
            self.controller.publish(msg)
            rospy.sleep(0.6)
            #print(self.get_current_continuous_state())
            print(self.get_current_continuous_state())

        result = np.array(planner.action_seq)
        np.savetxt('task1/1_maze_{}_{}.txt'.format(self.goal.pose.position.x, self.goal.pose.position.y),
                   result, fmt="%.2e")

    def publish_stochastic_control(self):
        """publish stochastic controls in MDP. 
        In MDP, we simulate the stochastic dynamics of the robot as described in the assignment description.
        Please use this function to publish your controls in task 3, MDP. DO NOT CHANGE THE PARAMETERS :)
        We will test your policy using the same function.
        """
        current_state = self.get_current_state()
        actions = []
        new_state = current_state
        while not self._check_goal(current_state):
            current_state = self.get_current_state()
            action = self.action_table[current_state[0],
                                       current_state[1], current_state[2] % 4]
            #print('d[]',self.d[current_state[1],current_state[0]])
            print('current_state',current_state[0],current_state[1])
            print('d',self.d[current_state[1],current_state[0]])
            # print('d',self.d[current_state[1]+1,current_state[0]])
            # print('d',self.d[current_state[1],current_state[0]]+1)
            # print('d',self.d[current_state[1],current_state[0]]-1)
            # print('d',self.d[current_state[1]-1,current_state[0]])
            if action == (1, 0):
                r = np.random.rand()
                if r < 1.90:
                    action = (1, 0)
                elif r < 0.95:
                    action = (np.pi/2, 1)
                else:
                    action = (np.pi/2, -1)
            print("Sending actions:", action[0], action[1]*np.pi/2)
            msg = self.create_control_msg(action[0], 0, 0, 0, 0, action[1]*np.pi/2)
            self.controller.publish(msg)
            rospy.sleep(0.6)
            self.controller.publish(msg)
            rospy.sleep(0.6)
            #time.sleep(1)
            current_state = self.get_current_state()

        action_table={}
        for key in self.action_table.keys():
            action_table[str(key)[1:-1]] = self.action_table[key]
        with open('task3/3_maze_{}_{}.json'.format(self.goal.pose.position.x, self.goal.pose.position.y), 'w') as fp:
            json.dump(action_table, fp)


if __name__ == "__main__":
    # TODO: You can run the code using the code below
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal', type=str, default='1,8',
                        help='goal position')
    parser.add_argument('--com', type=int, default=0,
                        help="if the map is com1 map")
    parser.add_argument('--plan', type=int, default=1,
                        help="nth plan")
    args = parser.parse_args()

    try:
        goal = [int(pose) for pose in args.goal.split(',')]
    except:
        raise ValueError("Please enter correct goal format")

    if args.com:
        width = 2500
        height = 983
        resolution = 0.02
    else:
        width = 200
        height = 200
        resolution = 0.05

    # TODO: You should change this value accordingly
    inflation_ratio = 10
    planner = Planner(width, height, resolution, inflation_ratio=inflation_ratio)
    planner.set_goal(goal[0], goal[1])
    if planner.goal is not None:
        if(args.plan==1):planner.generate_plan1()
        elif(args.plan==2):planner.generate_plan2()
        elif(args.plan==3):
            planner.generate_plan3()
            # with open('3_com1_43_10.json') as f:
            #     planner.action_table = json.load(f)
            # action_table={}
            # for key in planner.action_table.keys():
            #     action_table[(int(key.split(',')[0][1:]),int(key.split(',')[1]),int(key.split(',')[2][:-1]))] = \
            #         planner.action_table[key]
            # planner.action_table=action_table
            # print(planner.action_table)
        else:
            #planner.action_seq=[(1,0),(1,0),(-1,0),(-1,0)]
            planner.action_seq=[(0,np.pi),(0,np.pi),(1,0),(1,0),(1,0),(1,0),(0,-np.pi)]


    # You could replace this with other control publishers

    if (args.plan == 1):
        planner.publish_discrete_control()
    elif (args.plan == 2):
        planner.publish_control()
    elif(args.plan == 3):
        planner.publish_stochastic_control()
    else:
        planner.publish_control()

    # save your action sequence


    # for MDP, please dump your policy table into a json file
    # dump_action_table(planner.action_table, 'mdp_policy.json')
    print('finish')
    # spin the ros
    rospy.spin()
