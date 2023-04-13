import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import ShapelyFeature
import geopandas

cmap = LinearSegmentedColormap.from_list('cmap',['tab:red','tab:blue','yellow'],N=3)


ttype = np.load("data/best_turbine.npy")
lons = np.linspace(65,100,ttype.shape[1])
lats = np.linspace(40,5,ttype.shape[0])


fig = plt.figure(figsize=(6,6))

ax = plt.subplot(1,1,1, projection=cartopy.crs.PlateCarree())

cs = ax.pcolormesh(lons, lats, ttype, cmap=cmap, vmin=-0.5, vmax=2.5)



shpfilename = shapereader.natural_earth('50m', 'cultural', 'admin_1_states_provinces')
df = geopandas.read_file(shpfilename)
df = df[df['admin']=='India']
df.plot(ax=ax, facecolor='none', edgecolor='k', linewidth=0.5)



fig.subplots_adjust(bottom=0.22, )

x0 = ax.get_position().x0
x1 = ax.get_position().x1
L = x1-x0

cax = plt.gcf().add_axes([x0+L/8, 0.125, 3*L/4., 0.025])
cb = plt.colorbar(cs, cax=cax, orientation='horizontal',)
cb.set_label("Optimal turbine type")

cb.set_ticks([0,1,2])
cb.set_ticklabels(["Gamesa G87\n2000 MW","Vestas v110\n2000 MW", "Enercon E70\n2300 MW"])



ax.set_xticks(np.arange(0,100,10))
ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
ax.set_yticks(np.arange(0,100,10))
ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)


ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)
ax.set_xlim([lons.min(), lons.max()])
ax.set_ylim([lats.min(), lats.max()])


plt.show()



