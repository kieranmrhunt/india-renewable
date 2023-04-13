import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import ShapelyFeature
import geopandas


seasons = [1,2], [4,5], [7,8], [10,11]

year = '2018'

months = []
ws = []
us = []
vs = []

for year in range(1979,2021):
	print(year)
	uinfile = Dataset("../era5/data/{}_100m_u_component_of_wind.nc".format(year))
	vinfile = Dataset("../era5/data/{}_100m_v_component_of_wind.nc".format(year))

	u = uinfile.variables['u100'][:]
	v = vinfile.variables['v100'][:]
	us.extend(u)
	vs.extend(v)
	ws.extend(np.hypot(u,v))

	lons = uinfile.variables['longitude'][:] 
	lats = uinfile.variables['latitude'][:]

	time = uinfile.variables['time']
	times = num2date(time[:], time.units)
	months.extend(np.array([t.month for t in times]))




ws = np.array(ws)
us = np.array(us)
vs = np.array(vs)

months = np.array(months)


ws_mean = ws.mean(axis=0)
u_mean = us.mean(axis=0)
v_mean = vs.mean(axis=0)

cmap = LinearSegmentedColormap.from_list('cmap',['w','silver','dimgray','yellow','red'])


fig = plt.figure(figsize=(15,8))

ax1 = plt.subplot(1,2,1, projection=cartopy.crs.PlateCarree())
ax2 = plt.subplot(2,4,3, projection=cartopy.crs.PlateCarree())
ax3 = plt.subplot(2,4,4, projection=cartopy.crs.PlateCarree())
ax4 = plt.subplot(2,4,7, projection=cartopy.crs.PlateCarree())
ax5 = plt.subplot(2,4,8, projection=cartopy.crs.PlateCarree())

axes = [ax1,ax2,ax3,ax4,ax5]
vm=9.5
levels = np.arange(0,10,0.5)

s = 10
t = 2

#cs = ax1.pcolormesh(lons, lats, ws_mean, cmap=cmap, vmin=0, vmax=vm, rasterized=True)
cs = ax1.contourf(lons[::t], lats[::t], ws_mean[::t,::t], cmap=cmap, levels=levels, extend='max')
q = ax1.quiver(lons[::s], lats[::s], u_mean[::s,::s], v_mean[::s,::s], scale=100)


df = geopandas.read_file("/home/users/kieran/geodata/india_states_lores.zip")
df.plot(ax=ax1, facecolor='none', edgecolor='k', linewidth=0.25)




fig.subplots_adjust(bottom=0.185, wspace=.1, hspace=.125)
cax = plt.gcf().add_axes([0.175, 0.1, 0.25, 0.025])
cb = plt.colorbar(cs, cax=cax, orientation='horizontal', extend='max')
cb.set_label("Mean 100-m wind speed (m s$^{-1}$)")


print(ws_mean.min(), ws_mean.max())

s=10
t=2

for season, ax in zip(seasons, [ax2,ax3,ax4,ax5]):
	it = np.in1d(months, season)
	ws_mean = ws[it].mean(axis=0)
	us_mean = us[it].mean(axis=0)
	vs_mean = vs[it].mean(axis=0)
	
	#ax.pcolormesh(lons, lats, ws_mean, cmap=cmap, vmin=0, vmax=vm, rasterized=True)
	cs = ax.contourf(lons[::t], lats[::t], ws_mean[::t,::t], cmap=cmap, levels=levels, extend='max')
	
	ax.quiver(lons[::s], lats[::s], us_mean[::s,::s], vs_mean[::s,::s], scale=100)
	
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



