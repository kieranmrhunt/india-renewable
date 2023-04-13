import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import ShapelyFeature
import geopandas
import xarray as xr
from affine import Affine
from skimage.measure import block_reduce


cmap1 = LinearSegmentedColormap.from_list('cmap', ['w','palegreen','forestgreen','brown','k'],N=10)
cmapc = LinearSegmentedColormap.from_list('cmap', ['grey','palegreen','yellow','green','orange','cyan','tab:blue','purple','red', 'w'],N=10)

fig = plt.figure(figsize=(15,8))

ax1 = plt.subplot(1,2,1, projection=cartopy.crs.PlateCarree())
ax2 = plt.subplot(1,2,2, projection=cartopy.crs.PlateCarree())


axes = [ax1,ax2,]



df = geopandas.read_file("/home/users/kieran/geodata/india_states_lores.zip")



da = xr.open_rasterio("data/tree_cover/gm_ve_v2_1_5.tif")
mask_lon = (da.x >= 65) & (da.x <= 100)
mask_lat = (da.y >= 5) & (da.y <= 40)
da = da.where(mask_lon & mask_lat, drop=True)
x, y = da['x'], da['y']
tpc = da.variable.data[0]
tpc[tpc>100] = -1
s = 10
tpc = block_reduce(tpc, block_size=(s,s), func=np.mean)
np.save("data/means/treecoverage", tpc)

cs1 = ax1.pcolormesh(x[::s], y[::s], tpc, vmin=0, vmax=100, cmap=cmap1, rasterized=True)



fig.subplots_adjust(bottom=0.185, wspace=.1, hspace=.125, right = 0.8)

x0 = ax1.get_position().x0
x1 = ax1.get_position().x1
L = x1-x0

cax1 = plt.gcf().add_axes([x0+L/6., 0.1, 2*L/3., 0.025])
cb1 = plt.colorbar(cs1, cax=cax1, orientation='horizontal', extend='max')
cb1.set_label("Fraction of tree cover (%)")



da = xr.open_rasterio("data/lulc/glc_shv10_DOM.Tif")

mask_lon = (da.x >= 65) & (da.x <= 100)
mask_lat = (da.y >= 5) & (da.y <= 40)
da = da.where(mask_lon & mask_lat, drop=True)
x, y = da['x'], da['y']
lulc = da.variable.data[0]
np.save("data/means/lulc", lulc)

cs2 = ax2.pcolormesh(x,y,lulc, rasterized=True, vmin=0.5, vmax=10.5, cmap=cmapc)


y1 = ax2.get_position().y0
y2 = ax2.get_position().y1


cax2 = plt.gcf().add_axes([0.825, y1, 0.025, y2-y1])
cb2 = plt.colorbar(cs2, cax=cax2, orientation='vertical')
cb2.set_ticks(np.arange(1,11))
cb2.set_ticklabels(["Urban", "Crop","Grass","Tree","Shrub","Herbaceous\nor aquatic","Mangrove","Sparse\nvegetation","Bare soil", "Snow\nor glacier"])
cb2.ax.invert_yaxis()
cb2.set_label("Land use category")

	
ax1.set_title("(a) tree cover")	
ax2.set_title("(b) land use")	

ax1.set_yticks(np.arange(0,100,10))
ax1.yaxis.set_major_formatter(LATITUDE_FORMATTER)


shape_feature = ShapelyFeature(shapereader.Reader("/home/users/kieran/geodata/ne_10m_IND_lores/ne_10m_admin_0_countries_ind.shp").geometries(),
                                cartopy.crs.PlateCarree(), facecolor='none', edgecolor='k', lw=0.75)


df.plot(ax=ax1, facecolor='none', edgecolor='k', linewidth=0.25)
df.plot(ax=ax2, facecolor='none', edgecolor='k', linewidth=0.25)

for ax in ax1, ax2:
	ax.set_xticks(np.arange(0,100,10))
	ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)

	
	ax.coastlines(zorder=2)
	ax.add_feature(shape_feature, zorder=2)
	ax.add_feature(cartopy.feature.OCEAN, facecolor='lightgrey',zorder=1)
	ax.set_xlim([65,100])
	ax.set_ylim([5,40])



plt.show()
