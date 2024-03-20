#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt


def bilinear(x,y,vecs):
    if x >= 0 and x <= 19 and y>= 0 and y<=19:
        x_1 = int(x)
        x_2 = x_1 + 1
        y_1 = int(y)
        y_2 = y_1 + 1
        value = []
        #First let us compute it for the x direction
        vec_1 = np.array([[x_2 - x, x - x_1]])
        vec_2 = np.array([[y_2 - y],
                        [y - y_1]])
        matrix = np.array([[vecs[x_1][y_1][0],vecs[x_1][y_2][0]],
                        [vecs[x_2][y_1][0],vecs[x_2][y_2][0]]])
        k = np.matmul(vec_1,matrix)
        final = np.matmul(k,vec_2)
        value.append(final[0][0])

        #For the y directtion
        matrix = np.array([[vecs[x_1][y_1][1],vecs[x_1][y_2][1]],
                        [vecs[x_2][y_1][1],vecs[x_2][y_2][1]]])
        k = np.matmul(vec_1,matrix)
        final = np.matmul(k,vec_2)
        value.append(final[0][0])
        return np.array(value)

    
    else:
        return np.array([])

    
def run_euler(t,x0,bound,vecs):
    lst = []
    lst.append(x0)
    prev_pos = x0
    prev_vel = bilinear(x0[0],x0[1],vecs)
    for i in range(bound):
        if len(prev_vel) > 0:
            next_pos = prev_pos + t*prev_vel
            if next_pos[0]<=20 and next_pos[0] >= 0 and next_pos[1] >=0 and next_pos[1]<= 20:
                lst.append(next_pos)
            prev_pos = next_pos
            prev_vel = bilinear(prev_pos[0],prev_pos[1],vecs)
        else:
            break
    
    return np.array(lst)

    
def run_rk4(t,x0,bound,vecs):
    lst = []
    lst.append(x0)
    prev_pos = x0
    prev_vel = bilinear(x0[0],x0[1],vecs)
    for i in range(bound):

        k_1 =np.array([])
        k_2 = np.array([])
        k_3 = np.array([])
        k_4 = np.array([])

        k_1 = prev_vel
        if len(k_1)>0:
            k_2 = bilinear(prev_pos[0]+((t/2)*k_1[0]),prev_pos[1]+((t/2)*k_1[1]),vecs)
        if len(k_2)>0:
            k_3 = bilinear(prev_pos[0]+((t/2)*k_2[0]),prev_pos[1]+((t/2)*k_2[1]),vecs)
        if len(k_3)>0:
            k_4 = bilinear(prev_pos[0]+(t*k_3[0]),prev_pos[1]+(t*k_3[1]),vecs)
        if len(k_1) > 0 and len(k_2) > 0 and len(k_3) > 0 and len(k_4) > 0:
            next_pos = prev_pos + ((t/6)*(k_1 + (2*k_2) + (2*k_3) + k_4))
            lst.append(next_pos)
            prev_pos = next_pos
            prev_vel = bilinear(prev_pos[0],prev_pos[1],vecs)
        else:
            break

    return np.array(lst)




np.random.seed(100)
ran =19*np.random.rand(15,2)
# Get data
vecs = np.reshape(np.fromfile("wind_vectors.raw"), (20,20,2))
vecs_flat = np.reshape(vecs, (400,2)) # useful for plotting
vecs = vecs.transpose(1,0,2) # needed otherwise vectors don't match with plot
# X and Y coordinates of points where each vector is in space
xx, yy = np.meshgrid(np.arange(0, 20),np.arange(0, 20))
# Plot vectors
plt.plot(ran[:,0],ran[:,1],marker='.',color='black',linestyle='none')# This is for the seed points
plt.plot(xx, yy, marker='.', color='b', linestyle='none')
plt.quiver(xx, yy, vecs_flat[:,0], vecs_flat[:,1], width=0.001)
for i in range(15):
    stream = run_rk4(0.0375,ran[i],64,vecs)
    plt.plot(stream[:,0],stream[:,1],color = 'red')
plt.show()