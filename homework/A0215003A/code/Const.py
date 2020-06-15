import numpy as np
actionspace=5
envshape=(64,64)
#odd is left
shift=160
HoughLineColor=100
YellowLineColor=255
WhiteLineColor=50

action25={}
for i in range(actionspace*actionspace):
    a, b = i / actionspace, i % actionspace
    action25[i]=np.array([(a + 1) * 0.2, (b + 1) * 0.2])

# action = {0: np.array([0.5, 0.5]),
#                  1: np.array([0.8, 1.0]), 2: np.array([1.0, 0.8]),
#                  3: np.array([0.4, 0.5]), 4: np.array([0.5, 0.4]),
#                  5: np.array([-0.1, 0.1]), 6: np.array([0.1, -0.1]),
#                  7: np.array([0.8, 0.8]),8: np.array([1, 1]),
#                  9: np.array([0.1, 0.5]),10: np.array([0.5, 0.1]),
#                  11: np.array([0.1, 0.3]),12: np.array([0.3, 0.1]),}
action = {0: np.array([ 0.43,0]),
                 1: np.array([0.82,2.61]), 2: np.array([ 0.82,-2.61]),
                 3: np.array([0.39,0.84]), 4: np.array([ 0.39,-0.88]),
                 5: np.array([0. ,1.68]), 6: np.array([ 0. , -1.68]),
                 7: np.array([ 0.69,0]),8: np.array([ 0.86,0]),
                 9: np.array([0.26,3.38]),10: np.array([ 0.26,-3.38]),
                 11: np.array([0.17,1.69]),12: np.array([ 0.17,-1.69]),
                    13:np.array([0,4]),14:np.array([0,-4]),
                  15:np.array([0.98,9.74]),16:np.array([0.98,-9.74])}
stringToActionIdx={'g':0,'lgg':1,'rgg':2,'lg':3,'rg':4,'ol':5,'or':6,'gg':7,'ggg':8,
                   'llg':9,'rrg':10,'slg':11,'srg':12,'llll':13,'rrrr':14,
                   'lggg':15,'rggg':16}
# slg srg for map1, smooth turn right/left
#lggg,rggg for other maps, since faster turn right/left

finalreward=1
inputType=4
COLOROBS,GRAYOBS,LINEOBS,POINTOBS,HANDOBS=0,1,2,3,4
print('actionspace',actionspace)


'''
#parser.add_argument('--map-name', default='straight_road')
#parser.add_argument('--map-name', default='zigzag_dists')
parser.add_argument('--map-name', default='small_loop_cw')
'''


'''
k [0.5 0.5] [0.50140927 0.50022129] [ 0.43 -0.01]
k [0.8 1. ] [0.80001165 1.        ] [0.82 2.61]
k [1.  0.8] [1.         0.80001165] [ 0.82 -2.61]
k [0.4 0.5] [0.40433263 0.50412299] [0.39 0.84]
k [0.5 0.4] [0.50649895 0.40195667] [ 0.39 -0.88]
k [-0.1  0.1] [-0.09979036  0.09979036] [0.   1.68]
k [ 0.1 -0.1] [ 0.09979036 -0.09979036] [ 0.   -1.68]
k [0.8 0.8] [0.80660377 0.80066387] [ 0.69 -0.05]
k [1 1] [1. 1.] [ 0.86 -0.02]
k [0.1 0.5] [0.10204985 0.50358724] [0.26 3.38]
k [0.5 0.1] [0.50358724 0.10204985] [ 0.26 -3.38]
k [0.1 0.3] [0.09761239 0.29838109] [0.17 1.69]
k [0.3 0.1] [0.29838109 0.09761239] [ 0.17 -1.69]


'''