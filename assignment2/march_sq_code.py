#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

# Draws the starting plot. Don't change this code
def draw_initial_plot(data, x, y):

    # Draw grid and hide labels
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    ax.set_xlim(-.5, len(x)-.5)
    ax.set_ylim(-.5, len(y)-.5)
    ax.grid(True)
    plt.xticks(np.arange(-.5,data.shape[0], step=1))
    plt.yticks(np.arange(-.5,data.shape[1], step=1))
    ax.xaxis.set_ticks_position('none')
    ax.yaxis.set_ticks_position('none')
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])

    # Add text on cells
    for i in range(len(x)):
        for j in range(len(y)):
            ax.text(i,j+.1, str(int(data[i,j])), ha='center', va='bottom', size=18)

    return fig, ax


# Load data and make range arrays for looping
data = np.load("scalars_2D.npy") # access scalar values by data[i,j]
x = np.arange(0,data.shape[0])
y = np.arange(0,data.shape[1])

fig, ax = draw_initial_plot(data, x, y)


# Draws a dot at a given point
# point - type of tuple or array with length 2: (x,y) or [x,y]
# color - string of color to draw dot - Ex: ""red", "green", "blue"
def draw_dot(point, color):
    ax.scatter(point[0], point[1], color=color)

# Draws a line from point0 to point1
# point0 - type of tuple or array with length 2: (x,y) or [x,y]
# point1 - type of tuple or array with length 2: (x,y) or [x,y]
def draw_line(point0, point1):
    x = [point0[0], point1[0]]
    y = [point0[1], point1[1]]
    ax.plot(x, y, color="black")


#-----------------------
# ASSIGNMENT STARTS HERE
#-----------------------

isovalue = 50
linear_interpolation = True# else, midpoint method

# Add colored points to identify if cells are below or above the isovalue threshold
for i in x:
    for j in y:
        # TODO Part 1
        temp_point= [i,j]
        if data[i][j]<isovalue:
            draw_dot(temp_point,'black')
        else:
            draw_dot(temp_point,'red')



# Draw Lines in Marching Squares - Midpoint
def march_sq_midpoint(data, i, j, isovalue):
    # TODO Part 2
    temp_str = ''
    if data[i][j]<= isovalue:
        temp_str = temp_str + '0'
    else:
        temp_str = temp_str + '1'
    
    if data[i+1][j]<= isovalue:
        temp_str = temp_str + '0'
    else:
        temp_str = temp_str + '1'
    
    if data[i][j+1]<= isovalue:
        temp_str = temp_str + '0'
    else:
        temp_str = temp_str + '1'

    if data[i+1][j+1]<= isovalue:
        temp_str = temp_str + '0'
    else:
        temp_str = temp_str + '1'

    # Now to implement 14 cases:
    if temp_str == '0100':
        temp_point1 = [i+0.5,j]
        temp_point2 = [i+1,j+0.5]
        draw_line(temp_point1,temp_point2)
    
    if temp_str == '1000':
        temp_point1 = [i+0.5,j]
        temp_point2 = [i,j+0.5]
        draw_line(temp_point1,temp_point2)
    
    if temp_str == '0010':
        temp_point1 = [i,j+0.5]
        temp_point2 = [i+0.5,j+1]
        draw_line(temp_point1,temp_point2)

    if temp_str == '0001':
        temp_point1 = [i+1,j+0.5]
        temp_point2 = [i+0.5,j+1]
        draw_line(temp_point1,temp_point2)
    
    if temp_str == '1100':
        temp_point1 = [i,j+0.5]
        temp_point2 = [i+1,j+0.5]
        draw_line(temp_point1,temp_point2)

    if temp_str == '0110':
        temp_point1 = [i+0.5,j]
        temp_point2 = [i,j+0.5]
        draw_line(temp_point1,temp_point2)

        temp_point1 = [i+1,j+0.5]
        temp_point2 = [i+0.5,j+1]
        draw_line(temp_point1,temp_point2)

    if temp_str == '0011':
        temp_point1 = [i,j+0.5]
        temp_point2 = [i+1,j+0.5]
        draw_line(temp_point1,temp_point2)
    

    if temp_str == '0101':
        temp_point1 = [i+0.5,j]
        temp_point2 = [i+0.5,j+1]
        draw_line(temp_point1,temp_point2)

    if temp_str == '1010':
        temp_point1 = [i+0.5,j]
        temp_point2 = [i+0.5,j+1]
        draw_line(temp_point1,temp_point2)

    if temp_str == '1001':
        temp_point1 = [i+0.5,j]
        temp_point2 = [i+1,j+0.5]
        draw_line(temp_point1,temp_point2)

        temp_point1 = [i,j+0.5]
        temp_point2 = [i+0.5,j+1]
        draw_line(temp_point1,temp_point2)

    if temp_str == '0111':
        temp_point1 = [i+0.5,j]
        temp_point2 = [i,j+0.5]
        draw_line(temp_point1,temp_point2)
    
    if temp_str == '1011':
        temp_point1 = [i+0.5,j]
        temp_point2 = [i+1,j+0.5]
        draw_line(temp_point1,temp_point2)

    if temp_str == '1101':
        temp_point1 = [i,j+0.5]
        temp_point2 = [i+0.5,j+1]
        draw_line(temp_point1,temp_point2)

    if temp_str == '1110':
        temp_point1 = [i+1,j+0.5]
        temp_point2 = [i+0.5,j+1]
        draw_line(temp_point1,temp_point2)

    return

# Draw Lines in Marching Squares - Linear Interpolation
def march_sq_lin_interp(data, i, j, isovalue):
    # TODO Part 3
    temp_str = ''
    if data[i][j]<= isovalue:
        temp_str = temp_str + '0'
    else:
        temp_str = temp_str + '1'
    
    if data[i+1][j]<= isovalue:
        temp_str = temp_str + '0'
    else:
        temp_str = temp_str + '1'
    
    if data[i][j+1]<= isovalue:
        temp_str = temp_str + '0'
    else:
        temp_str = temp_str + '1'

    if data[i+1][j+1]<= isovalue:
        temp_str = temp_str + '0'
    else:
        temp_str = temp_str + '1'

    # Now to implement 14 cases:

    if temp_str == '0100':
        temp_point1 = [i+((isovalue-data[i][j])/(data[i+1][j]-data[i][j])),j]
        temp_point2 = [i+1,j+((data[i+1][j]-isovalue)/(data[i+1][j]-data[i+1][j+1]))]
        draw_line(temp_point1,temp_point2)
    
    if temp_str == '1000':
        temp_point1 = [i+((data[i][j]-isovalue)/(data[i][j]-data[i+1][j])),j]
        temp_point2 = [i,j+((data[i][j]-isovalue)/(data[i][j]-data[i][j+1]))]
        draw_line(temp_point1,temp_point2)
    
    if temp_str == '0010':
        temp_point1 = [i,j+((isovalue-data[i][j])/(data[i][j+1]-data[i][j]))]
        temp_point2 = [i+((data[i][j+1]-isovalue)/(data[i][j+1]-data[i+1][j+1])),j+1]
        draw_line(temp_point1,temp_point2)

    if temp_str == '0001':
        temp_point1 = [i+1,j+((isovalue-data[i+1][j])/(data[i+1][j+1]-data[i+1][j]))]
        temp_point2 = [i+((isovalue-data[i][j+1])/(data[i+1][j+1]-data[i][j+1])),j+1]
        draw_line(temp_point1,temp_point2)

    if temp_str == '1100':
        temp_point1 = [i,j+((data[i][j]-isovalue)/(data[i][j]-data[i][j+1]))]
        temp_point2 = [i+1,j+((data[i+1][j]-isovalue)/(data[i+1][j]-data[i+1][j+1]))]
        draw_line(temp_point1,temp_point2)

    if temp_str == '0110':
        temp_point1 = [i+ ((isovalue-data[i][j])/(data[i+1][j]-data[i][j])),j]
        temp_point2 = [i,j+((isovalue-data[i][j])/(data[i][j+1]-data[i][j]))]
        draw_line(temp_point1,temp_point2)

        temp_point1 = [i+1,j+((data[i+1][j]-isovalue)/(data[i+1][j]-data[i+1][j+1]))]
        temp_point2 = [i+((data[i][j+1]-isovalue)/(data[i][j+1]-data[i+1][j+1])),j+1]
        draw_line(temp_point1,temp_point2)


    if temp_str == '0011':
        temp_point1 = [i,j+((isovalue-data[i][j])/(data[i][j+1]-data[i][j]))]
        temp_point2 = [i+1,j+((isovalue-data[i+1][j])/(data[i+1][j+1]-data[i+1][j]))]
        draw_line(temp_point1,temp_point2)
    

    if temp_str == '0101':
        temp_point1 = [i+((isovalue-data[i][j])/(data[i+1][j]-data[i][j])),j]
        temp_point2 = [i+((isovalue-data[i][j+1])/(data[i+1][j+1]-data[i][j+1])),j+1]
        draw_line(temp_point1,temp_point2)

    if temp_str == '1010':
        temp_point1 = [i+((data[i][j]-isovalue)/(data[i][j]-data[i+1][j])),j]
        temp_point2 = [i+((data[i][j+1]-isovalue)/(data[i][j+1]-data[i+1][j+1])),j+1]
        draw_line(temp_point1,temp_point2)

    if temp_str == '1001':
        temp_point1 = [i+((data[i][j]-isovalue)/(data[i][j]-data[i+1][j])),j]
        temp_point2 = [i+1,j+((isovalue-data[i+1][j])/(data[i+1][j+1]-data[i+1][j]))]
        draw_line(temp_point1,temp_point2)

        temp_point1 = [i,j+((data[i][j]-isovalue)/(data[i][j]-data[i][j+1]))]
        temp_point2 = [i+((isovalue-data[i][j+1])/(data[i+1][j+1]-data[i][j+1])),j+1]
        draw_line(temp_point1,temp_point2)

    if temp_str == '0111':
        temp_point1 = [i+((isovalue-data[i][j])/(data[i+1][j]-data[i][j])),j]
        temp_point2 = [i,j+((isovalue-data[i][j])/(data[i][j+1]-data[i][j]))]
        draw_line(temp_point1,temp_point2)

    if temp_str == '1011':
        temp_point1 = [i+((data[i][j] - isovalue)/(data[i][j]-data[i+1][j])),j]
        temp_point2 = [i+1,j+((isovalue-data[i+1][j])/(data[i+1][j+1]-data[i+1][j]))]
        draw_line(temp_point1,temp_point2)
  
    if temp_str == '1101':
        temp_point1 = [i,j+((data[i][j]-isovalue)/(data[i][j]-data[i][j+1]))]
        temp_point2 = [i+((isovalue-data[i][j+1])/(data[i+1][j+1]-data[i][j+1])),j+1]
        draw_line(temp_point1,temp_point2)

    if temp_str == '1110':
        temp_point1 = [i+1,j+((data[i+1][j]-isovalue)/(data[i+1][j]-data[i+1][j+1]))]
        temp_point2 = [i+((data[i][j+1]-isovalue)/(data[i][j+1]-data[i+1][j+1])),j+1]
        draw_line(temp_point1,temp_point2)

    return 

# Implement simple marching squares with midpoint approach
for i in x[0:-1]:
    for j in y[0:-1]:
        if (linear_interpolation):
            march_sq_lin_interp(data, i, j, isovalue)
        else:
            march_sq_midpoint(data, i, j, isovalue)


plt.show()