import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
import cartopy
from cartopy.io import shapereader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import geopandas
import fiona

bxmin, bxmax = 65,100
bymin, bymax = 5,40

gdf = gp.read_file("global_pv_inventory/predicted_set.geojson")
gdf = gdf.cx[bxmin:bxmax, bymin:bymax]

print(gdf)
print(gdf.columns)


dx = 1
gridx = np.arange(60,100,dx)
gridy = np.arange(5,40,dx)
p = np.zeros((len(gridy),len(gridx)))


for j in range(len(gridy)):
	ymin = gridy[j]-dx/2.
	ymax = ymin+dx

	for i in range(len(gridx)):
		xmin = gridx[i]-dx/2.
		xmax = xmin+dx
		
		gdf_small = gdf.cx[xmin:xmax, ymin:ymax]
		cap = gdf_small['capacity_mw'].sum()/1000.
		print(cap)
		
		p[j,i]=cap
		
np.save("gridded-capactity", p)
print(p)

ax = plt.subplot(1,1,1, projection=cartopy.crs.PlateCarree())

cs =ax.pcolormesh(gridx,gridy,p, cmap = plt.cm.hot_r)


shpfilename = shapereader.natural_earth('50m', 'cultural', 'admin_1_states_provinces')
df = geopandas.read_file(shpfilename)
df = df[df['admin']=='India']
df.plot(ax=ax, facecolor='none', edgecolor='k', linewidth=0.5)

ax.set_xticks(np.arange(0,100,10))
ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
ax.set_yticks(np.arange(0,100,10))
ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)

ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)
ax.set_xlim([bxmin, bxmax])
ax.set_ylim([bymin, bymax])



plt.colorbar(cs)
plt.show()




