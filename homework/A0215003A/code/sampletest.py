import argparse
import numpy as np
from gym_duckietown.envs import DuckietownEnv

# declare the arguments
parser = argparse.ArgumentParser()

# Do not change this
parser.add_argument('--max_steps', type=int, default=2000, help='max_steps')

# You should set them to different map name and seed accordingly
parser.add_argument('--map-name', default='map5')
parser.add_argument('--seed', type=int, default=8, help='random seed')
args = parser.parse_args()
#for mapidx in range(5):
ma={
"map1": [2, 3, 5, 6, 9, 10],
"map2": [1, 2, 3, 4, 5, 6, 7, 8],
"map3": [1, 2, 3, 4, 7, 8, 9, 10],
"map4": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#"map5": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
"map5": [11]
#"map5": [7,9]
}
#for mapidx in [4,5]:
for mapidx in range(5,6):
    args.map_name = 'map'+str(mapidx)
    # for testidx in [6,7,9,10]:
    for testidx in ma[args.map_name]:
        args.seed=testidx
        env = DuckietownEnv(
            map_name=args.map_name,
            domain_rand=False,
            draw_bbox=False,
            max_steps=args.max_steps,
            seed=args.seed
        )
        obs = env.reset()
        #env.render()
        print('args',args)
        total_reward = 0

        # please remove this line for your own policy
        actions = np.loadtxt('ans/'+args.map_name+'_seed'+str(args.seed)+'.txt', delimiter=',')
        #actions = np.loadtxt(args.map_name+'_seed'+str(args.seed)+'.txt', delimiter=',')

        for (speed, steering) in actions:
            obs, reward, done, info = env.step([speed, steering])
            total_reward += reward

            #print((speed, steering))
            #print('Steps = %s, Timestep Reward=%.3f, Total Reward=%.3f' % (env.step_count, reward, total_reward))

            env.render()

        print("Total Reward", total_reward)