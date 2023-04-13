import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy

cmap_name = 'cmp_flux'
rgb = np.genfromtxt("/home/users/kieran/miniconda3/lib/python3.7/site-packages/cfplot/colourmaps/{}.rgb".format(cmap_name))/255.
cmap = LinearSegmentedColormap.from_list(cmap_name, rgb)


year = 2000

uinfile = Dataset("../era5/data/{}_100m_u_component_of_wind.nc".format(year))
vinfile = Dataset("../era5/data/{}_100m_v_component_of_wind.nc".format(year))

u = uinfile.variables['u100'][0]
v = vinfile.variables['v100'][0]

ws_mean = np.hypot(u,v)

lons = uinfile.variables['longitude'][:] 
lats = uinfile.variables['latitude'][:]

vm=9.5




fig = plt.figure(figsize=(8,8))

ax = plt.subplot(1,1,1, projection=cartopy.crs.PlateCarree())

cs = ax.pcolormesh(lons, lats, ws_mean, cmap=cmap, vmin=0, vmax=vm, rasterized=True)

fig.subplots_adjust(bottom=0.185, wspace=.1, hspace=.125)
cax = plt.gcf().add_axes([0.175, 0.1, 0.25, 0.025])
cb = plt.colorbar(cs, cax=cax, orientation='horizontal', extend='max')
cb.set_label("Mean 100-m wind speed (m s$^{-1}$)")



ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)
ax.set_xlim([lons.min(), lons.max()])
ax.set_ylim([lats.min(), lats.max()])


plt.show()


