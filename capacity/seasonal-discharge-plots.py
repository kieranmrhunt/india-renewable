import numpy as np
from netCDF4 import Dataset, num2date
from datetime import datetime as dt, timedelta as td
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.colors as colors
import cartopy
from cartopy.io import shapereader
from cartopy.feature import ShapelyFeature
import geopandas
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER


find = lambda x,arr: np.argmin(np.abs(x-arr))



seasons = [1,2], [4,5], [7,8], [10,11]

x1, x2, y1, y2 = 65,100,5,40

months = []
flows = []

fdir = "/home/users/kieran/incompass/users/kieran/glofas/"

datelist = []
start = dt(2000,1,1)
#stop = dt(2001,1,1)
stop = dt(2019,1,1)
while start<stop:
	datelist.append(start)
	start+=td(days=1)


lons = np.arange(-180,180,0.1)
lats = np.arange(90,-60,-0.1)

ix1, ix2 = find(x1,lons), find(x2,lons)
iy1, iy2 = find(y2,lats), find(y1,lats)

for date in datelist:
	print(date)
	
	infile = Dataset(fdir+date.strftime("CEMS_ECMWF_dis24_%Y%m%d_glofas_v2.1.nc"))
	
	dis = infile.variables['dis24'][:].squeeze()[iy1:iy2,ix1:ix2]
	#print(dis, ix1,ix2,iy1,iy2)
	flows.append(dis)

	months.append(date.month)

lons = lons[ix1:ix2]
lats = lats[iy1:iy2]


flows = np.array(flows).squeeze()
months = np.array(months)

flows = np.ma.masked_where(flows>1e10, flows)

print(flows.shape)

flows_mean = flows.mean(axis=0)
np.save("data/means/discharge", flows_mean.data)

#print(flows_mean)

cmap = LinearSegmentedColormap.from_list('cmap',['w','silver','dimgray','yellow','red'])


fig = plt.figure(figsize=(15,8))

ax1 = plt.subplot(1,2,1, projection=cartopy.crs.PlateCarree())
ax2 = plt.subplot(2,4,3, projection=cartopy.crs.PlateCarree())
ax3 = plt.subplot(2,4,4, projection=cartopy.crs.PlateCarree())
ax4 = plt.subplot(2,4,7, projection=cartopy.crs.PlateCarree())
ax5 = plt.subplot(2,4,8, projection=cartopy.crs.PlateCarree())

axes = [ax1,ax2,ax3,ax4,ax5]
vmin=1e-3
vmax=1e4

cs = ax1.pcolormesh(lons, lats, flows_mean, cmap=cmap, norm=colors.LogNorm(vmin=vmin, vmax=vmax),rasterized=True)


df = geopandas.read_file("/home/users/kieran/geodata/india_states_lores.zip")
df.plot(ax=ax1, facecolor='none', edgecolor='k', linewidth=0.25)


fig.subplots_adjust(bottom=0.185, wspace=.1, hspace=.125)
cax = plt.gcf().add_axes([0.175, 0.1, 0.25, 0.025])
cb = plt.colorbar(cs, cax=cax, orientation='horizontal', extend='both')
cb.set_label("Mean streamflow (m$^{3}$ s$^{-1}$)")


print(flows_mean.min(), flows_mean.max())


for season, ax in zip(seasons, [ax2,ax3,ax4,ax5]):
	it = np.in1d(months, season)
	flows_mean = flows[it].mean(axis=0)
	ax.pcolormesh(lons, lats, flows_mean, cmap=cmap, norm=colors.LogNorm(vmin=vmin, vmax=vmax), rasterized=True)
	
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



