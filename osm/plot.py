import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import cartopy
from cartopy.io import shapereader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
from scipy.ndimage import gaussian_filter as gf
import matplotlib.colors as colors


xmin, xmax = 65, 100
ymin, ymax = 5, 40

ds = 0.25
xgrid = np.arange(xmin, xmax, ds)
ygrid = np.arange(ymin, ymax, ds)

def to_grid(xpts, ypts, xgrid=xgrid, ygrid=ygrid, ds=ds):
	gridded = np.empty((len(ygrid), len(xgrid)))
	X,Y = np.meshgrid(xgrid, ygrid)
	
	for i, x in enumerate(xgrid):
		for j, y in enumerate(ygrid):
			N = np.sum((xpts>=x-ds/2)&(xpts<x+ds/2)&(ypts>=y-ds/2)&(ypts<y+ds/2))
			gridded[j,i]=N
	
	gridded/=(ds**2) #convert to sqdeg
	gridded/=(1.11*np.cos(Y*np.pi/180.)) #convert to /(100km)^2
	
	return gridded
		
	
	


fig = plt.figure(figsize=(12,5))

ax1 = plt.subplot(1,3,1, projection=cartopy.crs.PlateCarree())
ax2 = plt.subplot(1,3,2, projection=cartopy.crs.PlateCarree())
ax3 = plt.subplot(1,3,3, projection=cartopy.crs.PlateCarree())

sigma = 1.5

df = gpd.read_file("wind-turbines.geojson")
df['x'] = df['geometry'].centroid.x
df['y'] = df['geometry'].centroid.y
print(df)
gridded = gf(to_grid(df.x, df.y), (sigma, sigma))
cs1 = ax1.contourf(xgrid, ygrid, gridded, cmap=plt.cm.RdPu, levels=[50,100,250,500,1000,2000,3000],extend='max', norm=colors.PowerNorm(gamma=0.5))
cb1 = plt.colorbar(cs1, ax=ax1, orientation='horizontal')

df = gpd.read_file("solar.geojson")
df['x'] = df['geometry'].centroid.x
df['y'] = df['geometry'].centroid.y
print(df)
gridded = gf(to_grid(df.x, df.y), (sigma, sigma))
cs2 = ax2.contourf(xgrid, ygrid, gridded, cmap=plt.cm.YlOrRd, levels=[10,20,50,100,250,500,1000], extend='max', norm=colors.PowerNorm(gamma=0.5))
cb2 = plt.colorbar(cs2, ax=ax2, orientation='horizontal')

df = gpd.read_file("hydro.geojson")
df['x'] = df['geometry'].centroid.x
df['y'] = df['geometry'].centroid.y
print(df)
gridded = gf(to_grid(df.x, df.y), (sigma, sigma))
cs3 = ax3.contourf(xgrid, ygrid, gridded, cmap=plt.cm.PuBuGn, levels=np.arange(1,15,1), extend='max', norm=colors.PowerNorm(gamma=0.5))
cb3 = plt.colorbar(cs3, ax=ax3, orientation='horizontal')

ax1.set_title("(a) Wind turbines")
ax2.set_title("(b) Solar panel installations")
ax3.set_title("(c) Hydropower plants")


df = gpd.read_file("/home/users/kieran/geodata/india_states_lores.zip")
shape_feature = ShapelyFeature(shapereader.Reader("/home/users/kieran/geodata/ne_10m_IND_lores/ne_10m_admin_0_countries_ind.shp").geometries(),
                                cartopy.crs.PlateCarree(), facecolor='none', edgecolor='k', lw=0.75)


for ax in ax1,ax2,ax3:	
	ax.add_feature(cartopy.feature.OCEAN, facecolor='w', linewidth=0, zorder=1)
	
	df.plot(ax=ax, facecolor='none', edgecolor='k', linewidth=0.25, alpha=0.5)
	ax.add_feature(shape_feature, zorder=2)

	ax.set_xticks(np.arange(0,100,10))
	ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
	
	if ax in [ax1,ax3]:
		ax.set_yticks(np.arange(0,100,10))
		ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
	
	ax.set_xlabel("")
	ax.set_ylabel("")
	

ax3.yaxis.tick_right()

for cb in cb1, cb2, cb3:
	cb.set_label("Installation density [units (100 km)$^{-2}$]")

for ax in [ax1,ax2,ax3]:
	ax.coastlines()
	#ax.add_feature(cartopy.feature.BORDERS)
	ax.set_xlim([xmin, xmax])
	ax.set_ylim([ymin, ymax])

plt.show()
