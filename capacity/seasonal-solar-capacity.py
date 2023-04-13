import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import ShapelyFeature
import geopandas


cmap_name = 'StepSeq25'
rgb = np.genfromtxt("/home/users/kieran/miniconda3/lib/python3.7/site-packages/cfplot/colourmaps/{}.rgb".format(cmap_name))/255.
cmap = LinearSegmentedColormap.from_list(cmap_name, rgb, N=25)


seasons = [1,2], [4,5], [7,8], [10,11]

year = '2018'

months = []
sw = []
t2 = []

for year in range(1979,2023):
	print(year)
	swinfile = Dataset("../era5/data/{}_mean_surface_downward_short_wave_radiation_flux.nc".format(year))
	t2infile = Dataset("../era5/data/{}_2m_temperature.nc".format(year))

	sw_data = swinfile.variables['msdwswrf'][:]#[:365*24]
	if year == 2022:
		sw_data = sw_data[:,0]
	sw.extend(sw_data)
	
	t2_data = t2infile.variables['t2m'][:]-273.15
	if year == 2022:
		t2_data = t2_data[:,0]
	t2.extend(t2_data)

	lons = swinfile.variables['longitude'][:] 
	lats = swinfile.variables['latitude'][:]

	time = t2infile.variables['time']
	times = num2date(time[:], time.units)
	months.extend(np.array([t.month for t in times]))




sw = np.array(sw)
#sw = np.concatenate([np.zeros((7,len(lats),len(lons))),sw],axis=0)
t2 = np.array(t2)
months = np.array(months)


T_ref = 25. 
eff_ref = 0.9 
beta_ref = 0.0042
G_ref = 1000.
 
rel_efficiency_of_panel = eff_ref*(1 - beta_ref*(t2 - T_ref))
capacity_factor_of_panel = np.nan_to_num(rel_efficiency_of_panel*
                                        (sw/G_ref)) 

#np.save("data/full_arrays/solarcapacity-old", capacity_factor_of_panel.reshape(-1,24,141,141).mean(axis=1))
np.save("data/full_arrays/solarcapacity-since2020", capacity_factor_of_panel.reshape(-1,24,141,141).mean(axis=1))


mean_capacity = capacity_factor_of_panel.mean(axis=0)
np.save("data/means/solarcapacity", mean_capacity)


levels = np.arange(0,0.50,0.02)
print(len(levels))

#cmap = LinearSegmentedColormap.from_list('cmap',['w','grey','magenta', 'b'])


fig = plt.figure(figsize=(15,8))

ax1 = plt.subplot(1,2,1, projection=cartopy.crs.PlateCarree())
ax2 = plt.subplot(2,4,3, projection=cartopy.crs.PlateCarree())
ax3 = plt.subplot(2,4,4, projection=cartopy.crs.PlateCarree())
ax4 = plt.subplot(2,4,7, projection=cartopy.crs.PlateCarree())
ax5 = plt.subplot(2,4,8, projection=cartopy.crs.PlateCarree())

axes = [ax1,ax2,ax3,ax4,ax5]



cs = ax1.pcolormesh(lons, lats, mean_capacity, cmap=cmap, vmin=0, vmax=0.5, rasterized=True)

df = geopandas.read_file("/home/users/kieran/geodata/india_states_lores.zip")
df.plot(ax=ax1, facecolor='none', edgecolor='k', linewidth=0.25)


fig.subplots_adjust(bottom=0.185, wspace=.1, hspace=.125)
cax = plt.gcf().add_axes([0.175, 0.1, 0.25, 0.025])
cb = plt.colorbar(cs, cax=cax, orientation='horizontal', extend='max')
cb.set_label("Mean hourly capacity factor (solar)")


print(mean_capacity.min(), mean_capacity.max())


for season, ax in zip(seasons, [ax2,ax3,ax4,ax5]):
	it = np.in1d(months, season)
	mean_capacity = capacity_factor_of_panel[it].mean(axis=0)
	ax.pcolormesh(lons, lats, mean_capacity, cmap=cmap, vmin=0, vmax=0.5, rasterized=True)
	
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

