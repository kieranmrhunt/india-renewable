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

cmap_name = 'scale27'
rgb = np.genfromtxt("/home/users/kieran/miniconda3/lib/python3.7/site-packages/cfplot/colourmaps/{}.rgb".format(cmap_name))/255.
rgb[0] = [0.99,0.99,0.99]
cmap = LinearSegmentedColormap.from_list(cmap_name, rgb, N=20)


x1, x2, y1, y2 = 65,100,5,40


ds = Dataset("data/gpw_v4_population_density_rev11_15_min.nc")
pd = ds.variables['Population Density, v4.11 (2000, 2005, 2010, 2015, 2020): 15 arc-minutes'][4]




lons = ds.variables['longitude']
lats = ds.variables['latitude']




fig = plt.figure(figsize=(5,4))
ax = plt.subplot(1,1,1, projection=cartopy.crs.PlateCarree())


cs = ax.pcolormesh(lons, lats, pd, cmap=cmap, vmax=3000)


#df = geopandas.read_file("/home/users/kieran/geodata/india_states_lores.zip")
#df.plot(ax=ax, facecolor='none', edgecolor='k', linewidth=0.25)

shape_feature = ShapelyFeature(shapereader.Reader("/home/users/kieran/geodata/ne_10m_IND_lores/ne_10m_admin_0_countries_ind.shp").geometries(),
                                cartopy.crs.PlateCarree(), facecolor='none', edgecolor='k', lw=0.75)

ax.coastlines()
ax.add_feature(shape_feature)



ax.set_xlim([x1, x2])
ax.set_ylim([y1, y2])

plt.colorbar(cs, label="Population Density (people km$^{-2}$)", extend='max')


plt.show()
