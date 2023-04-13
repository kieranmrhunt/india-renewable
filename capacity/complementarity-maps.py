import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import ShapelyFeature
import geopandas

def np_pearson_cor(x, y):
	xv = x - x.mean(axis=0)
	yv = y - y.mean(axis=0)
	xvss = (xv * xv).sum(axis=0)
	yvss = (yv * yv).sum(axis=0)
	result = np.sum(xv*yv, axis=0)/np.sqrt(xvss*yvss)
	return np.maximum(np.minimum(result, 1.0), -1.0)


xmin, xmax = 65, 100
ymin, ymax = 5, 40


cmap_name = 'BlueWhiteOrangeRed'
rgb = np.genfromtxt("/home/users/kieran/miniconda3/lib/python3.7/site-packages/cfplot/colourmaps/{}.rgb".format(cmap_name))/255.
cmap = LinearSegmentedColormap.from_list(cmap_name, rgb, N=20)


solar_in = np.load("data/full_arrays/solarcapacity.npy")
print(solar_in.shape)
wind_in = np.load("data/full_arrays/windcapacity.npy")
print(wind_in.shape)


lons = np.linspace(xmin,xmax,wind_in.shape[-1])
lats = np.linspace(ymax,ymin,wind_in.shape[-2])

solar = solar_in.reshape((5,365,24,len(lats),len(lons))).mean(axis=0)
wind = wind_in.reshape((5,365,24,len(lats),len(lons))).mean(axis=0)

diurnal_comp = np_pearson_cor(solar.mean(axis=0), wind.mean(axis=0))
seasonal_comp = np_pearson_cor(solar.mean(axis=1), wind.mean(axis=1))
full_comp = np_pearson_cor(solar.reshape((-1,len(lats),len(lons))), wind.reshape((-1,len(lats),len(lons))))

fig = plt.figure(figsize=(15,8))

ax1 = plt.subplot(1,3,1, projection=cartopy.crs.PlateCarree())
ax2 = plt.subplot(1,3,2, projection=cartopy.crs.PlateCarree())
ax3 = plt.subplot(1,3,3, projection=cartopy.crs.PlateCarree())

axes = [ax1,ax2,ax3]

cs1 = ax1.pcolormesh(lons, lats, diurnal_comp, cmap=cmap, vmin=-1, vmax=1, rasterized=True)
cs2 = ax2.pcolormesh(lons, lats, seasonal_comp, cmap=cmap, vmin=-1, vmax=1, rasterized=True)
cs3 = ax3.pcolormesh(lons, lats, full_comp, cmap=cmap, vmin=-1, vmax=1, rasterized=True)

shpfilename = shapereader.natural_earth('50m', 'cultural', 'admin_1_states_provinces')
df = geopandas.read_file(shpfilename)
df = df[df['admin']=='India']
df.plot(ax=ax1, facecolor='none', edgecolor='k', linewidth=0.5)
df.plot(ax=ax2, facecolor='none', edgecolor='k', linewidth=0.5)


'''
fig.subplots_adjust(bottom=0.185, wspace=.1, hspace=.125)
cax = plt.gcf().add_axes([0.175, 0.1, 0.25, 0.025])
cb = plt.colorbar(cs, cax=cax, orientation='horizontal', extend='max')
cb.set_label("Mean hourly capacity factor (solar)")
'''


for ax in axes:
	ax.set_xticks(np.arange(0,100,10))
	ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)

ax1.set_yticks(np.arange(0,100,10))
ax1.yaxis.set_major_formatter(LATITUDE_FORMATTER)

ax3.set_yticks(np.arange(0,100,10))
ax3.yaxis.set_major_formatter(LATITUDE_FORMATTER)
ax3.yaxis.tick_right()


for ax in axes:
	ax.coastlines()
	ax.add_feature(cartopy.feature.BORDERS)
	ax.set_xlim([xmin, xmax])
	ax.set_ylim([ymin, ymax])

ax1.set_title("(a) diurnal complementarity")
ax2.set_title("(b) annual complementarity")
ax3.set_title("(c) full complementarity")





fig.subplots_adjust(wspace=.1, hspace=.125, right = 0.8)

y1 = ax3.get_position().y0
y2 = ax3.get_position().y1

cax2 = plt.gcf().add_axes([0.85, y1, 0.0125, y2-y1])
cb2 = plt.colorbar(cs2, cax=cax2, orientation='vertical')
cb2.set_label("Correlation coefficient")


plt.show()
