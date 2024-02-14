import matplotlib.pyplot as plt
import numpy as np 
import matplotlib.tri as tri
from matplotlib import colormaps
import random
import math
import pandas as pd

uniform = np.random.uniform(0,1,100)
normal = np.random.normal(50,20,200)
print(uniform)

# This is for the boxplots
box_uniform = plt.boxplot(uniform)
plt.ylabel('Values')
plt.title("BoxPlot-Uniform")
plt.show()
box_normal = plt.boxplot(normal)
plt.ylabel('Values')
plt.title("BoxPlot-Normal")
plt.show()


#For plotting the Histogram

#a) Uniform
uniform_bins = np.zeros(20)
x_axis = []
for i in range(1,21):
    x_axis.append(str(i))
for i in uniform:
    uniform_bins[int(i/0.05)] = uniform_bins[int(i/0.05)] + 1

plt.bar(x_axis, uniform_bins, color ='maroon', 
        width = 0.4)
plt.xlabel('Bins')
plt.ylabel('# of elements')
plt.title("BarChart-Uniform")

plt.show()

#b) Normal

normal_bins = np.zeros(20)
x_axis = []
for i in range(1,21):
    x_axis.append(str(i))
for i in normal:
    if i>0 and i<=100:
        normal_bins[math.ceil(i/5)-1] = normal_bins[math.ceil(i/5)-1] + 1

plt.bar(x_axis, normal_bins, color ='maroon', 
        width = 0.4)

plt.xlabel('Bins')
plt.ylabel('# of elements')
plt.title("BarChart-Normal")

plt.show()

# For plotting the cdf
#a) Uniform
uniform.tofile('uniform')
uniform = np.fromfile('uniform')
print(uniform)
uniform_bins = np.zeros(20)
x_axis = []
for i in range(21):
    x_axis.append(0.05*i)
for i in uniform:
    uniform_bins[int(i/0.05)] = uniform_bins[int(i/0.05)] + 1
cdf = np.cumsum(uniform_bins)
cdf = np.insert(cdf,0,0)
print(uniform_bins)
print(cdf)
cdf = cdf/100
plt.plot(x_axis,cdf)
plt.ylabel('CDF')
plt.xlabel('x')
plt.title("CDF-Uniform")
plt.show()

#b) Normal

normal.tofile('normal')
normal = np.fromfile('normal')
print(normal)
normal_bins = np.zeros(20)
x_axis = []
for i in range(21):
    x_axis.append(5*i)
for i in normal:
   if i>0 and i<=100:
        normal_bins[math.ceil(i/5)-1] = normal_bins[math.ceil(i/5)-1] + 1
cdf = np.cumsum(normal_bins)
cdf = np.insert(cdf,0,0)
print(normal_bins)
print(cdf)
cdf = cdf/200
plt.plot(x_axis,cdf)
plt.ylabel('CDF')
plt.xlabel('x')
plt.title("CDF-Normal")
plt.show()
'''
'''
# Moving on to the 2D case
#a) Uniform
uniform = np.random.uniform(0,1,(5000,2))
x = uniform[:,0]
y = uniform[:,1]
print(x)
#print(uniform)
plt.scatter(x,y)
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.xlim([0,1])
plt.ylim([0,1])
plt.title("Uniform Random Sampling")
plt.show()

# b) normal

normal = np.random.normal(0.5,0.1,(5000,2))
x = normal[:,0]
y = normal[:,1]
#print(uniform)
plt.scatter(x,y)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('X-Axis')
plt.ylabel('Y-Axis')
plt.title("Normal Random Sampling")
plt.show()

# Binning in the 2D case
# a)
uniform_bins = np.zeros((100,100))
for i in uniform:
    uniform_bins[int(i[0]/0.01)][int(i[1]/0.01)] = uniform_bins[int(i[0]/0.01)][int(i[1]/0.01)] + 1
plt.imshow(uniform_bins,cmap = 'inferno')
plt.title("2D Binning - Uniform")
plt.legend()
plt.xticks([])
plt.yticks([])
plt.show()

# b)
normal_bins = np.zeros((100,100))
for i in normal:
    if i[0]>=0 and i[0]<1 and i[1]>=0 and i[1]<1:
        normal_bins[int(i[0]/0.01)][int(i[1]/0.01)] = normal_bins[int(i[0]/0.01)][int(i[1]/0.01)] + 1
plt.imshow(normal_bins)
plt.xticks([])
plt.yticks([])
plt.title("2D Binning - Normal")
plt.show()

#c) Countour Plot

# Uniform
x = uniform[:,0]
y = uniform[:,1]
z = (x**2 + y**2)**(0.5)
triangulation = tri.Triangulation(x,y)
plt.tricontourf(triangulation,z,[0,0.15,0.3,0.45,0.6,0.75,0.9,1.05,1.2,1.35,1.5])
plt.xlim([0,1])
plt.ylim([0,1])
plt.title("Contour Plot - Uniform")
plt.show()

#Normal
x = normal[:,0]
y = normal[:,1]
z = (x**2 + y**2)**(0.5)
triangulation = tri.Triangulation(x,y)
plt.tricontourf(triangulation,z,[0,0.15,0.3,0.45,0.6,0.75,0.9,1.05,1.2,1.35,1.5])
plt.xlim([0,1])
plt.ylim([0,1])
plt.title("Contour Plot - Normal")
plt.show()

# The temperature ber chart
df = pd.read_csv('NOAA-Temperatures.csv')
df = df.iloc[4:]
df = np.array(df)
x = df[:,0]
y = df[:,1]
for i in range(len(y)):
    y[i] = float(y[i])
print(y)


#plt.bar(x[mask2], y[mask2], color ='red', width = 0.4)
colors = []
for i in range(len(y)):
    if y[i]>=0:
        colors.append('red')
    else:
        colors.append('blue')
#plt.bar(x[mask1], y[mask1], color ='blue', width = 0.4)

plt.bar(x, y, color =colors, width = 0.4)
plt.xticks([])
plt.xlabel('Year (1880-2017)')
plt.ylabel('Degrees F +/- from Average')
plt.title('NOAA Land Ocean Temperature Anomalies')
plt.show()

# Cereal Radar Chart
df = pd.read_excel('Breakfast-Cereals.xls')
print(df.max()['Calories'])

import plotly.graph_objects as go

categories = ['Calories-'+str(df.max()['Calories']),'Protein-'+str(df.max()['Protein']),'Fat-'+str(df.max()['Fat']),'Fiber-'+str(df.max()['Fiber']),'Carbohydrates-'+str(df.max()['Carbohydrates']),'Sugars-'+str(df.max()['Sugars']),'Vitamins-'+str(df.max()['Vitamins']),'Weight-'+str(df.max()['Weight'])]

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
      r=[df.loc[0]['Calories']/df.max()['Calories'],df.loc[0]['Protein']/df.max()['Protein'],df.loc[0]['Fat']/df.max()['Fat'],df.loc[0]['Fiber']/df.max()['Fiber'],df.loc[0]['Carbohydrates']/df.max()['Carbohydrates'],df.loc[0]['Sugars']/df.max()['Sugars'],df.loc[0]['Vitamins']/df.max()['Vitamins'],df.loc[0]['Weight']/df.max()['Weight']],
      theta=categories,
      fill='toself',
      name=df.loc[0]['Cereal']
))
fig.add_trace(go.Scatterpolar(
      r=[df.loc[1]['Calories']/df.max()['Calories'],df.loc[1]['Protein']/df.max()['Protein'],df.loc[1]['Fat']//df.max()['Fat'],df.loc[1]['Fiber']/df.max()['Fiber'],df.loc[1]['Carbohydrates']/df.max()['Carbohydrates'],df.loc[1]['Sugars']/df.max()['Sugars'],df.loc[1]['Vitamins']/df.max()['Vitamins'],df.loc[1]['Weight']/df.max()['Weight']],
      theta=categories,
      fill='toself',
      name=df.loc[1]['Cereal']
))

fig.add_trace(go.Scatterpolar(
      r=[df.loc[2]['Calories']/df.max()['Calories'],df.loc[2]['Protein']/df.max()['Protein'],df.loc[2]['Fat']//df.max()['Fat'],df.loc[2]['Fiber']/df.max()['Fiber'],df.loc[2]['Carbohydrates']/df.max()['Carbohydrates'],df.loc[2]['Sugars']/df.max()['Sugars'],df.loc[2]['Vitamins']/df.max()['Vitamins'],df.loc[2]['Weight']/df.max()['Weight']],
      theta=categories,
      fill='toself',
      name=df.loc[2]['Cereal']
))

fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
      range=[0, 1]
    )),
  showlegend=True,
  title = 'Cereal Radar Chart'
)
fig.show()

#Parallel Coordinates on drink.csv

x = ['beer_servings','spirit_servings','wine_servings']
data = pd.read_csv('drinks.csv',usecols=['country','beer_servings','spirit_servings','wine_servings'])
#data.drop(columns=data.columns[0], axis=1, inplace=True)
#data.drop(rows=data.rows[0], axis=0, inplace=True)
from pandas.plotting import parallel_coordinates
parallel_coordinates(data,'country')
plt.gca().legend_.remove()
plt.title('Alcohol Consumption of Various Countries')
plt.ylabel('# servings per person per year')
plt.show()

# Scatter Plot on hate_crimes.csv
data = pd.read_csv('hate_crimes.csv')
x = data.loc[:,'median_household_income']
x = np.array(x)

y = data.loc[:,'avg_hatecrimes_per_100k_fbi']
y = np.array(y)
plt.scatter(x,y)
plt.xlabel("Median Income")
plt.ylabel("Average Hatecrimes per 100k")
plt.title("Hate Crimes")
plt.show()

# Neuroimage

import nibabel as nib
img = nib.load('T2.nii')
print(img.shape)
cropped_image_data = img.get_fdata()[150,:,:]
cropped_image_data = cropped_image_data.reshape((320,256))
print(cropped_image_data.shape)
plt.imshow(cropped_image_data,cmap = colormaps['inferno'])
plt.xticks([])
plt.yticks([])
plt.title("X-direction -150")
plt.show()
plt.imshow(cropped_image_data,cmap = colormaps['flag'])
plt.xticks([])
plt.yticks([])
plt.title("X-direction -150")
plt.show()
cropped_image_data = img.get_fdata()[:,150,:]
cropped_image_data = cropped_image_data.reshape((320,256))
print(cropped_image_data.shape)
plt.imshow(cropped_image_data,cmap = colormaps['inferno'])
plt.xticks([])
plt.yticks([])
plt.title("Y-direction -150")
plt.show()
plt.imshow(cropped_image_data,cmap = colormaps['flag'])
plt.xticks([])
plt.yticks([])
plt.title("Y-direction -150")
plt.show()
cropped_image_data = img.get_fdata()[:,:,150]
cropped_image_data = cropped_image_data.reshape((320,320))
print(cropped_image_data.shape)
plt.imshow(cropped_image_data,cmap = colormaps['inferno'])
plt.xticks([])
plt.yticks([])
plt.title("Z-direction -150")
plt.show()
plt.imshow(cropped_image_data,cmap = colormaps['flag'])
plt.xticks([])
plt.yticks([])
plt.title("Z-direction -150")
plt.show()
