import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy
from cartopy.io import shapereader
from cartopy.feature import ShapelyFeature
import geopandas
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

seasons = [1,2], [4,5], [7,8], [10,11]

year = '2018'

months = []
wp = []
wh = []

for year in range(2000,2021):
	print(year)
	wpinfile = Dataset("../era5/data/{}_mean_wave_period.nc".format(year))
	whinfile = Dataset("../era5/data/{}_significant_height_of_combined_wind_waves_and_swell.nc".format(year))

	wp_data = wpinfile.variables['mwp'][:]
	wh_data = whinfile.variables['swh'][:]

	wp.extend(wp_data)
	wh.extend(wh_data)

	lons = wpinfile.variables['longitude'][:] 
	lats = wpinfile.variables['latitude'][:]

	time = wpinfile.variables['time']
	times = num2date(time[:], time.units)
	months.extend(np.array([t.month for t in times]))

wp = np.ma.array(wp)
wh = np.ma.array(wh)

months = np.array(months)

power = ((9.81**2)/(64*np.pi))*(wh**2)*wp

power_mean = power.mean(axis=0)
np.save("data/means/wavemean", power_mean.data)

cmap = LinearSegmentedColormap.from_list('cmap',['w','silver','dimgray','yellow','red'])


fig = plt.figure(figsize=(15,8))

ax1 = plt.subplot(1,2,1, projection=cartopy.crs.PlateCarree())
ax2 = plt.subplot(2,4,3, projection=cartopy.crs.PlateCarree())
ax3 = plt.subplot(2,4,4, projection=cartopy.crs.PlateCarree())
ax4 = plt.subplot(2,4,7, projection=cartopy.crs.PlateCarree())
ax5 = plt.subplot(2,4,8, projection=cartopy.crs.PlateCarree())

axes = [ax1,ax2,ax3,ax4,ax5]
vmin=0
vmax=30
levels = np.arange(0,30,1.5)

cs = ax1.contourf(lons, lats, power_mean, cmap=cmap, levels=levels, extend='max')

df = geopandas.read_file("/home/users/kieran/geodata/india_states_lores.zip")
df.plot(ax=ax1, facecolor='none', edgecolor='k', linewidth=0.25)

fig.subplots_adjust(bottom=0.185, wspace=.1, hspace=.125)
cax = plt.gcf().add_axes([0.175, 0.1, 0.25, 0.025])
cb = plt.colorbar(cs, cax=cax, orientation='horizontal')
cb.set_label("Mean hourly wave energy flux (kW m$^{-1}$)")


print(power_mean.min(), power_mean.max())

for season, ax in zip(seasons, [ax2,ax3,ax4,ax5]):
	it = np.in1d(months, season)
	power_mean = power[it].mean(axis=0)
	ax.contourf(lons, lats, power_mean, cmap=cmap, levels=levels, extend='max')
	
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
	ax.set_xlim([lons.min(), lons.max()])
	ax.set_ylim([lats.min(), lats.max()])


plt.show()



