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
import xarray as xr
from skimage.measure import block_reduce
from scipy.ndimage.filters import gaussian_filter as gf




xmin, xmax = 65, 100
ymin, ymax = 5, 40

fig = plt.figure(figsize=(10,8))

ax = plt.subplot(1,1,1, projection=cartopy.crs.PlateCarree())


dstopo = xr.open_dataset("~/geodata/ETOPO1_Ice_c_gmt4.nc")

mask_lon = (dstopo.x >= xmin) & (dstopo.x <= xmax)
mask_lat = (dstopo.y >= ymin) & (dstopo.y <= ymax)

dstopo = dstopo.where(mask_lon & mask_lat, drop=True)
oro = dstopo.z
lons = dstopo.x
lats = dstopo.y


df = geopandas.read_file("/home/users/kieran/geodata/india_states_lores.zip")
df.plot(ax=ax, facecolor='none', edgecolor='k', linewidth=0.25, zorder=5)

ax.set_xticks(np.arange(0,100,10))
ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
ax.set_yticks(np.arange(0,100,10))
ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)

shape_feature = ShapelyFeature(shapereader.Reader("/home/users/kieran/geodata/ne_10m_IND_lores/ne_10m_admin_0_countries_ind.shp").geometries(),
                                cartopy.crs.PlateCarree(), facecolor='none', edgecolor='k', lw=0.75)


x, y = np.gradient(gf(oro,(3,3)))
slope = np.pi/2. - np.arctan(np.sqrt(x*x + y*y))

aspect = np.arctan2(x, y)

altitude = np.pi/4.
azimuth = np.pi/2.

shaded = np.sin(altitude) * np.sin(slope) + np.cos(altitude) * np.cos(slope)\
    * np.cos((azimuth - np.pi/2.) - aspect)

#print(shaded.min(), shaded.max())


cmap1 =  LinearSegmentedColormap.from_list('my_colormap', [(0, 'w'),
                                                           (0.67, 'tab:blue'),
														   (1, 'darkblue')])


cmap2 =  LinearSegmentedColormap.from_list('my_colormap', [(0, 'forestgreen'),
                                                           (0.1, 'lightgreen'),
														   (0.25, '#fcf879'),
														   (0.67, '#bf9a6f'),
														   (1, 'w')])

colors1 = cmap1(np.linspace(0., 1, 64))
colors2 = cmap2(np.linspace(0, 1, 128))

# combine them and build a new colormap
colors = np.vstack((colors1, colors2))
mymap = LinearSegmentedColormap.from_list('mymap', colors)


cs = plt.pcolormesh(lons, lats, oro, cmap=mymap, vmin=-2500, vmax=5000, alpha=0.75, rasterized=True)
#plt.pcolormesh(lons, lats, shaded, cmap=plt.cm.Greys, alpha=0.25, vmin=-0.75,vmax=1.5)

cb=  plt.colorbar(cs, extend='both', orientation='vertical')
cb.set_label("Orography/bathymetry (m)")

ax.add_feature(shape_feature)
ax.add_feature(cartopy.feature.RIVERS, color='blue', linewidth=0.5)
ax.coastlines()
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

plt.savefig("/home/users/kieran/incompass/public/kieran/scratch/basemap.png", dpi=600)
#plt.show()










