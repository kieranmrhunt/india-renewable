import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import shapely
import numpy as np

x1, x2, y1, y2 = 65, 100, 5, 40
res = 0.1

lons = np.arange(x1,x2,res)
lats = np.arange(y2,y1,-res)


df =gpd.read_file("data/river-widths/asrivs.shp")
polygon = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)])

df_clip = gpd.clip(df, polygon)
df_clip['CS_AREA'] = df_clip['a_WIDTH']*df['a_DEPTH']*np.pi/4
df_clip['CS_SQAREA'] = df_clip['CS_AREA']**2


centroids = np.array([(ls.centroid.x, ls.centroid.y) for ls in df_clip['geometry'].values])
print(centroids.shape)
areas = df_clip['CS_AREA'].values


X, Y = np.meshgrid(lons, lats)
total = np.zeros_like(X)
count = np.zeros_like(X)

for i, x in enumerate(lons):
	for j, y in enumerate(lats):
		print(x, y)
		x1, x2 = x, x+res
		y1, y2 = y-res/2, y+res/2

		ix = (centroids[:,0]>=x1)&(centroids[:,0]<x2)&(centroids[:,1]>=y1)&(centroids[:,1]<y2)
		total[j,i] = np.mean(areas[ix])


np.save("data/river-widths/gridded-areas-edge", total)



areas = np.load("data/river-widths/gridded-areas-edge.npy")

plt.pcolormesh(lons, lats, areas)
plt.colorbar()

plt.show()
