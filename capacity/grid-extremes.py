import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import ShapelyFeature
import geopandas
from skimage.measure import block_reduce
from shapely.geometry import Point
from scipy.interpolate import interp2d
from scipy.stats import linregress

solar_in = np.concatenate([np.load("data/full_arrays/solarcapacity-old.npy"),
                           np.load("data/full_arrays/solarcapacity.npy")])

wind_in = np.concatenate([np.load("data/full_arrays/windcapacity-old.npy"),
                           np.load("data/full_arrays/windcapacity.npy")])



xmin, xmax = 65, 100
ymin, ymax = 5, 40
lons = np.linspace(xmin,xmax,wind_in.shape[-1])
lats = np.linspace(ymin,ymax,wind_in.shape[-2])

solar = solar_in.reshape((-1,365,len(lats),len(lons)))[:,:,::-1]
wind = wind_in.reshape((-1,365,len(lats),len(lons)))[:,:,::-1]

solar = block_reduce(solar, block_size = (1,1,4,4), func=np.mean)[:,:,:-1,:-1]
wind = block_reduce(wind, block_size = (1,1,4,4), func=np.mean)[:,:,:-1,:-1]

new_lons = block_reduce(lons, block_size=(4,), func=np.mean)[:-1]
new_lats = block_reduce(lats, block_size=(4,), func=np.mean)[:-1]


dx = 1
gridx = np.arange(60,100,dx)
gridy = np.arange(5,40,dx)



kw_arr = np.load("../kruitwagen/gridded-capactity.npy")
cea_solar_arr = np.load("../cea-data/gridded-solar-cap.npy")

cea_wind_arr = np.load("../cea-data/gridded-wind-cap.npy")
twp_arr = np.load("../cea-data/gridded-twp-cap.npy")

solar_installed = 0.5*(kw_arr+cea_solar_arr)
wind_installed = 0.5*(twp_arr+cea_wind_arr)


solar_installed = interp2d(gridx, gridy, solar_installed)(new_lons, new_lats)
wind_installed = interp2d(gridx, gridy, wind_installed)(new_lons, new_lats)


df = geopandas.read_file("/home/users/kieran/geodata/ne_10m_IND_lores/ne_10m_admin_0_countries_ind.shp")
poly = df.loc[df['ADMIN'] == 'India']['geometry'].values[0]

mask = np.zeros_like(solar_installed)

for i, x in enumerate(new_lons):
	for j, y in enumerate(new_lats):
		p = Point(x,y)		
		if poly.contains(p):
			mask[j,i]=1

solar_installed *=mask

print(np.sum(solar_installed))
print(np.sum(wind_installed))

solar_produced = solar*solar_installed[None, None, :, :] 
wind_produced = wind*wind_installed[None, None, :, :]

solar_total = np.sum(solar_produced, axis=(-1,-2))*2
wind_total = np.sum(wind_produced, axis=(-1,-2))

total_produced = wind_total + solar_total


cmap_name = 'cmp_b2r'
rgb = np.genfromtxt("/home/users/kieran/miniconda3/lib/python3.7/site-packages/cfplot/colourmaps/{}.rgb".format(cmap_name))/255.
cmap = LinearSegmentedColormap.from_list(cmap_name, rgb, N=20)

doys = np.arange(1,366)
years = np.arange(1979,2021)

plt.figure(figsize=(8,4))
plt.pcolormesh(doys,years,total_produced, cmap=cmap,)


deficient = total_produced<np.percentile(total_produced, 1)
dx, dy = np.where(deficient)
plt.plot(doys[dy], years[dx], marker = 'o', mfc = 'yellow', lw=0, mec='k', mew=0.5)
counts = np.array([np.sum(dx==i) for i in range(42)])
print(np.mean(counts))
print(linregress(range(42), counts))

surplus = total_produced>np.percentile(total_produced, 99)
dx, dy = np.where(surplus)
plt.plot(doys[dy], years[dx], marker = 'o', mfc = 'grey', lw=0, mec='k', mew=0.5)
counts = np.array([np.sum(dx==i) for i in range(42)])
print(np.mean(counts))
print(linregress(range(42), counts))


plt.xticks([1,32,60,91,121,152,182,213,244,274,305,335])
plt.gca().set_xticklabels(['Jan 1', 'Feb 1', 'Mar 1', 'Apr 1', 'May 1', 'Jun 1',
                 'Jul 1', 'Aug 1', 'Sep 1', 'Oct 1', 'Nov 1', 'Dec 1'], 
					 rotation=45)


cb = plt.colorbar(extend='both')
cb.set_label("Total realised wind production (GW)\n(daily average)")


plt.tight_layout()
plt.show()

