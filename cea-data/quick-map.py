import numpy as np
import cartopy
import pandas as pd
import matplotlib.pyplot as plt
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature

df = pd.read_csv("parsed-cea-all.csv")
df = df.loc[df['type'].isin(['Solar',])]

#df = pd.read_csv("parsed_solar-OSM.csv")


lons = df.lon.values
lats = df.lat.values

number_of_locations = len(lons)
number_of_unique_locations = len(set(list(zip(lons,lats))))

print(number_of_locations, number_of_unique_locations)


fig = plt.figure(figsize=(8,8))

ax=plt.axes(projection=cartopy.crs.PlateCarree())

ax.plot(lons, lats, 'r.')

ax.coastlines()
sname = '/home/users/kieran/python/mapindia/states/INDIA.shp'
shape_feature = ShapelyFeature(Reader(sname).geometries(), 
                                cartopy.crs.PlateCarree(), edgecolor='black', facecolor='None', linewidth=0.5,)
ax.add_feature(shape_feature)

plt.show()
