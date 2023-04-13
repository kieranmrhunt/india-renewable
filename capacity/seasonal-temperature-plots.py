import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import ShapelyFeature
import geopandas


cmap_name = 'amwg_blueyellowred'
rgb = np.genfromtxt("/home/users/kieran/miniconda3/lib/python3.7/site-packages/cfplot/colourmaps/{}.rgb".format(cmap_name))/255.
cmap = LinearSegmentedColormap.from_list(cmap_name, rgb)


seasons = [1,2], [4,5], [7,8], [10,11]

months = []
sw = []
t2 = []

'''
for year in range(2000,2001):
	print(year)

	t2infile = Dataset("../era5/data/{}_2m_temperature.nc".format(year))

	t2_data = t2infile.variables['t2m'][:]-273.15
	t2.extend(t2_data)

	lons = t2infile.variables['longitude'][:] 
	lats = t2infile.variables['latitude'][:]

	time = t2infile.variables['time']
	times = num2date(time[:], time.units)
	months.extend(np.array([t.month for t in times]))
'''

t2infile = Dataset("/home/users/kieran/incompass/users/kieran/era5/single-level-monthly-means/2m_temperature.nc")
t2 = t2infile.variables['t2m'][:]-273.15
lons = t2infile.variables['longitude'][:] 
lats = t2infile.variables['latitude'][:]

time = t2infile.variables['time']
times = num2date(time[:], time.units)
months = np.array([t.month for t in times])





t2 = np.array(t2)
mean_t2 = np.mean(t2, axis=0)


fig = plt.figure(figsize=(15,8))

ax1 = plt.subplot(1,2,1, projection=cartopy.crs.PlateCarree())
ax2 = plt.subplot(2,4,3, projection=cartopy.crs.PlateCarree())
ax3 = plt.subplot(2,4,4, projection=cartopy.crs.PlateCarree())
ax4 = plt.subplot(2,4,7, projection=cartopy.crs.PlateCarree())
ax5 = plt.subplot(2,4,8, projection=cartopy.crs.PlateCarree())

axes = [ax1,ax2,ax3,ax4,ax5]
vmin=10
vmax=35
levels = np.arange(vmin, vmax, 1)

t = 2

cs = ax1.contourf(lons[::t], lats[::t], mean_t2[::t,::t], cmap=cmap, levels = levels, extend='both')


df = geopandas.read_file("/home/users/kieran/geodata/india_states_lores.zip")
df.plot(ax=ax1, facecolor='none', edgecolor='k', linewidth=0.25)


fig.subplots_adjust(bottom=0.185, wspace=.1, hspace=.125)
cax = plt.gcf().add_axes([0.175, 0.1, 0.25, 0.025])
cb = plt.colorbar(cs, cax=cax, orientation='horizontal')
cb.set_label(u"Mean 2-m temperature (\u00B0C)")


print(mean_t2.min(), mean_t2.max())


for season, ax in zip(seasons, [ax2,ax3,ax4,ax5]):
	it = np.in1d(months, season)
	mean_t2= t2[it].mean(axis=0)
	ax.contourf(lons[::t], lats[::t], mean_t2[::t,::t], cmap=cmap, levels = levels, extend='both')
	
	

ax1.set_title("Annual mean")	
ax2.set_title("Winter (Jan-Feb)")	
ax3.set_title("Pre-monsoon (Apr-May)")	
ax4.set_title("Monsoon (Jul-Aug)")	
ax5.set_title("Post-monsoon (Oct-Nov)")	


for ax in ax1, ax4, ax5:
	ax.set_xticks(np.arange(0,100,10))
	ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)

for ax in ax3, ax5:
	ax.set_yticks(np.arange(0,100,10))
	ax.yaxis.tick_right()
	ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)


ax1.set_yticks(np.arange(0,100,10))
ax1.yaxis.set_major_formatter(LATITUDE_FORMATTER)


shape_feature = ShapelyFeature(shapereader.Reader("/home/users/kieran/geodata/ne_10m_IND_lores/ne_10m_admin_0_countries_ind.shp").geometries(),
                                cartopy.crs.PlateCarree(), facecolor='none', edgecolor='k', lw=0.75)

for ax in axes:
	ax.coastlines()
	ax.add_feature(shape_feature)
	ax.set_xlim([65,100])
	ax.set_ylim([5,40])


plt.show()
