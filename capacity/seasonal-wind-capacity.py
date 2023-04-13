import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from scipy.interpolate import InterpolatedUnivariateSpline

seasons = [1,2], [4,5], [7,8], [10,11]



months = []
ws = []

for year in np.arange(2000,2005):
	print(year)
	uinfile = Dataset("../era5/data/{}_100m_u_component_of_wind.nc".format(year))
	vinfile = Dataset("../era5/data/{}_100m_v_component_of_wind.nc".format(year))

	u = uinfile.variables['u100'][:365*24]
	v = vinfile.variables['v100'][:365*24]
	ws.extend(np.hypot(u,v))
	print(np.shape(ws))

	lons = uinfile.variables['longitude'][:] 
	lats = uinfile.variables['latitude'][:]

	time = uinfile.variables['time']
	times = num2date(time[:365*24], time.units)
	months.extend(np.array([t.month for t in times]))




ws = np.array(ws)
months = np.array(months)

capacities = []

turbines = ["Gamesa_G87_2000MW","Vestas_v110_2000MW", "Enercon_E70_2300MW"]

for turbine in turbines:
	
	power_curve = np.genfromtxt('../energy-model-scripts/extra_files_to_run_scripts/{}_ECEM_turbine.csv'.format(turbine))
	pc_w = power_curve[:,0]
	pc_c = power_curve[:,2]

	#pc_interpolate = interp1d(pc_w,pc_c, kind='linear', fill_value='extrapolate')
	#capacity = pc_interpolate(ws)
	pc_interpolate = InterpolatedUnivariateSpline(pc_w, pc_c)
	capacity = pc_interpolate(ws)
	
	
	capacities.append(capacity)

capacities = np.array(capacities)
best_turbine = np.argmax(np.mean(capacities,axis=1),axis=0)

capacity = np.zeros(capacities.shape[1:])
for i in range(capacity.shape[-1]):
	for j in range(capacity.shape[-2]):
		capacity[:,j,i] = capacities[best_turbine[j,i],:,j,i]


np.save("data/full_arrays/windcapacity", capacity.reshape(-1,24,141,141).mean(axis=1))

capacity_mean = np.mean(capacity, axis=0)


cmap = LinearSegmentedColormap.from_list('cmap',['w','grey','dodgerblue','magenta', 'red', 'k'])


plt.figure(figsize=(15,8))

ax1 = plt.subplot(1,2,1, projection=cartopy.crs.PlateCarree())
ax2 = plt.subplot(2,4,3, projection=cartopy.crs.PlateCarree())
ax3 = plt.subplot(2,4,4, projection=cartopy.crs.PlateCarree())
ax4 = plt.subplot(2,4,7, projection=cartopy.crs.PlateCarree())
ax5 = plt.subplot(2,4,8, projection=cartopy.crs.PlateCarree())

axes = [ax1,ax2,ax3,ax4,ax5]
vm=1

cs = ax1.pcolormesh(lons, lats, capacity_mean, cmap=cmap, vmin=0, vmax=vm)
sname = '/home/users/kieran/geodata/india-states/gadm36_IND_1.shp'
shape_feature = ShapelyFeature(Reader(sname).geometries(), 
                                cartopy.crs.PlateCarree(), edgecolor='black', facecolor='None', linewidth=0.5,)
ax1.add_feature(shape_feature)

cax = plt.gcf().add_axes([0.175, 0.1, 0.25, 0.025])
cb = plt.colorbar(cs, cax=cax, orientation='horizontal', extend='max')
cb.set_label("Mean hourly capacity factor (optimal turbine)")


print(capacity_mean.min(), capacity_mean.max())


for season, ax in zip(seasons, [ax2,ax3,ax4,ax5]):
	it = np.in1d(months, season)
	capacity_mean = capacity[it].mean(axis=0)
	ax.pcolormesh(lons, lats, capacity_mean, cmap=cmap, vmin=0, vmax=vm)
	
ax1.set_title("Annual mean")	
ax2.set_title("Winter (Jan-Feb)")	
ax3.set_title("Pre-monsoon (Apr-May)")	
ax4.set_title("Monsoon (Jul-Aug)")	
ax5.set_title("Post-monsoon (Oct-Nov)")	

for ax in axes:
	ax.coastlines()
	ax.add_feature(cartopy.feature.BORDERS)
	ax.set_xlim([lons.min(), lons.max()])
	ax.set_ylim([lats.min(), lats.max()])


plt.show()



