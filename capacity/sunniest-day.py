import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature



sw = []


for year in range(1980,2020):
	print(year)
	swinfile = Dataset("../era5/data/{}_mean_surface_downward_short_wave_radiation_flux.nc".format(year))
	

	sw_data = swinfile.variables['msdwswrf'][:]
	
	
	sw_raw =  swinfile.variables['msdwswrf'][:]
	sw_daily = np.mean(sw_raw.reshape((-1,24,141,141)),axis=1)
	

	lons = swinfile.variables['longitude'][:] 
	lats = swinfile.variables['latitude'][:]

	sw.append(sw_daily[:365])





sw = np.mean(sw, axis=0)
sw_max = np.argmax(sw, axis=0)
sw_min = np.argmin(sw, axis=0)


#cmap = LinearSegmentedColormap.from_list('cmap',['w','silver','dimgray','yellow','red'])
cmap = plt.cm.hsv_r


fig = plt.figure(figsize=(10,6))

ax1 = plt.subplot(1,2,1, projection=cartopy.crs.PlateCarree())
ax2 = plt.subplot(1,2,2, projection=cartopy.crs.PlateCarree())


cs1 = ax1.pcolormesh(lons, lats, sw_max, cmap=cmap, vmin=0, vmax=365)
cs2 = ax2.pcolormesh(lons, lats, sw_min, cmap=cmap, vmin=0, vmax=365)


sname = '/home/users/kieran/geodata/india-states/gadm36_IND_1.shp'
shape_feature = ShapelyFeature(Reader(sname).geometries(), 
                                cartopy.crs.PlateCarree(), edgecolor='black', facecolor='None', linewidth=0.5,)


#cax = plt.gcf().add_axes([0.175, 0.1, 0.25, 0.025])
#cb1 = plt.colorbar(cs1, ax=ax1, extend='both', orientation='horizontal')
#cb.set_label("Mean downward SW flux at surface (W m$^{-2}$)")
#cb2 = plt.colorbar(cs2, ax=ax2, extend='both', orientation='horizontal')

cbar_ax = fig.add_axes([0.15, 0.1, 0.7, 0.025])
cb = fig.colorbar(cs1, cax=cbar_ax, orientation='horizontal')

cb.set_ticks(np.cumsum([0,31,28,31,30,31,30,31,31,30,31,30,31]))
cb.set_ticklabels(['Jan 1', 'Feb 1', 'Mar 1', 'Apr 1', 'May 1', 'Jun 1', 'Jul 1', 'Aug 1', 'Sep 1', 'Oct 1', 'Nov 1', 'Dec 1'])

	
ax1.set_title("Sunniest day of the year")	
ax2.set_title("Least sunny day of the year")	


for ax in ax1, ax2:
	ax.coastlines()
	ax.add_feature(cartopy.feature.BORDERS)
	ax.set_xlim([lons.min(), lons.max()])
	ax.set_ylim([lats.min(), lats.max()])
	ax.add_feature(shape_feature)

plt.show()



