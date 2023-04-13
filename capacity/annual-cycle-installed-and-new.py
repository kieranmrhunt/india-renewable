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
from matplotlib.gridspec import GridSpec
import fiona
import shapely.geometry as sgeom
import shapely.vectorized as v
from shapely.prepared import prep
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter as gf
import xarray as xr

geoms = fiona.open(shapereader.natural_earth(resolution='50m', category='physical', name='land'))
land_geom = sgeom.MultiPolygon([sgeom.shape(geom['geometry']) for geom in geoms])
land = prep(land_geom)

df = geopandas.read_file("/home/users/kieran/geodata/ne_10m_IND_lores/ne_10m_admin_0_countries_ind.shp")
poly = df.loc[df['ADMIN'] == 'India']['geometry'].values[0]


def is_land(x, y):
    return land.contains(sgeom.Point(x, y))

def make_landmask(lons, lats):
	X, Y = np.meshgrid(lons, lats)
	land_mask = []
	for x, y in zip(X.ravel(), Y.ravel()):
		land_mask.append(is_land(x,y))
	land_mask = np.reshape(land_mask, (len(lats), len(lons)))
	return(land_mask)

def dist_from_coast(lons, lats):
	X, Y = np.meshgrid(lons, lats)
	coast_distance = np.empty(X.shape, dtype=np.float64)
	for i, x_pt in enumerate(lons):
		for j, y_pt in enumerate(lats):
			coast_distance[j, i] = poly.distance(sgeom.Point(x_pt, y_pt))
	coast_distance *= 111*np.cos(Y*np.pi/180.)
	return(coast_distance)

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

solar_installed = 0.5*(kw_arr+cea_solar_arr)*2
wind_installed = 0.5*(twp_arr+cea_wind_arr)


solar_installed = interp2d(gridx, gridy, solar_installed)(new_lons, new_lats)
wind_installed = interp2d(gridx, gridy, wind_installed)(new_lons, new_lats)



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

solar_total = np.sum(solar_produced, axis=(-1,-2))
wind_total = np.sum(wind_produced, axis=(-1,-2))

total_produced = solar_total + wind_total


winter_onshore_wind_mean = wind[:,-92:].mean(axis=(0,1))*mask
winter_wind_mask = winter_onshore_wind_mean>np.percentile(winter_onshore_wind_mean,99.2)
print(winter_wind_mask.sum())
onshore_winter_wind = np.sum(wind*winter_wind_mask[None, None, :, :] , axis=(-1,-2))*2


winter_onshore_solar_mean = solar[:,-92:].mean(axis=(0,1))*mask
winter_solar_mask = winter_onshore_solar_mean>np.percentile(winter_onshore_solar_mean,99.2)
print(winter_solar_mask.sum())
onshore_winter_solar = np.sum(solar*winter_solar_mask[None, None, :, :] , axis=(-1,-2))*2


land_mask = make_landmask(new_lons, new_lats)
coast_distance = dist_from_coast(new_lons, new_lats)
offshore_wind_mean = wind.mean(axis=(0,1))*(1-land_mask)*(coast_distance<250)
offshore_wind_mask = offshore_wind_mean>np.percentile(offshore_wind_mean,99.2)
print(offshore_wind_mask.sum())
offshore_wind = np.sum(wind*offshore_wind_mask[None, None, :, :] , axis=(-1,-2))*2


fig = plt.figure(figsize=(9,5))
gs = GridSpec(2, 3)
ax1 = fig.add_subplot(gs[:, :2])
ax2 = fig.add_subplot(gs[1, 2], projection=cartopy.crs.PlateCarree())



ax2.contourf(new_lons, new_lats, winter_solar_mask, levels=[0.5,1.5], colors=['r',])
ax2.contourf(new_lons, new_lats, winter_wind_mask, levels=[0.5,1.5], colors=['b',], alpha=0.67)
ax2.contourf(new_lons, new_lats, offshore_wind_mask , levels=[0.5,1.5], colors=['grey',])


ds = xr.Dataset(
    {
        'winter_solar_mask': (['lat', 'lon'], winter_solar_mask),
        'winter_wind_mask': (['lat', 'lon'], winter_wind_mask),
        'offshore_wind_mask': (['lat', 'lon'], offshore_wind_mask),
    },
    coords={
        'lat': (['lat'], new_lats),
        'lon': (['lon'], new_lons),
    },
)

ds.attrs['description'] = 'Example dataset with three masks'
ds['winter_solar_mask'].attrs['description'] = 'Winter-focused solar (20 GW)'
ds['winter_solar_mask'].attrs['units'] = 'binary'
ds['winter_wind_mask'].attrs['description'] = 'Winter-focused wind (20 GW)'
ds['winter_wind_mask'].attrs['units'] = 'binary'
ds['offshore_wind_mask'].attrs['description'] = 'Offshore wind (20 GW)'
ds['offshore_wind_mask'].attrs['units'] = 'binary'

ds.to_netcdf('data-for-export/areas-for-exploration.nc')



doys = np.arange(1,366)

ax1.plot(doys, np.mean(solar_total,axis=0), label="existing solar ($\sim$60 GW)", c='tab:orange', ls=':')
ax1.plot(doys, np.mean(wind_total,axis=0), label="existing wind ($\sim$40 GW)", c='tab:blue', ls=':')
ax1.plot(doys, np.mean(onshore_winter_wind,axis=0), label="proposed onshore wind\n(20 GW; winter focused)", c='b', alpha=0.67)
ax1.plot(doys, np.mean(offshore_wind,axis=0), label="proposed offshore wind\n(20 GW)", c='grey')
ax1.plot(doys, np.mean(onshore_winter_solar,axis=0), label="proposed solar\n(20 GW; winter focused)", c='r')

print(np.mean(wind_total))
print(np.mean(offshore_wind))

ax1.legend(loc='upper left', bbox_to_anchor=(1.05, 1.02))

ax1.set_xticks([1,32,60,91,121,152,182,213,244,274,305,335])
ax1.set_xticklabels(['Jan 1', 'Feb 1', 'Mar 1', 'Apr 1', 'May 1', 'Jun 1',
                 'Jul 1', 'Aug 1', 'Sep 1', 'Oct 1', 'Nov 1', 'Dec 1'], 
					 rotation=45)

ax1.set_xlim([1,365])
ax1.set_ylabel("Realised/potential production (GW)")


ax2.set_xticks(np.arange(0,100,10))
ax2.xaxis.set_major_formatter(LONGITUDE_FORMATTER)

ax2.set_yticks(np.arange(0,100,10))
ax2.yaxis.set_major_formatter(LATITUDE_FORMATTER)

shape_feature = ShapelyFeature(shapereader.Reader("/home/users/kieran/geodata/ne_10m_IND_lores/ne_10m_admin_0_countries_ind.shp").geometries(),
                                cartopy.crs.PlateCarree(), facecolor='none', edgecolor='k', lw=0.25)

ax2.coastlines(lw=0.25)
ax2.add_feature(shape_feature)
ax2.set_xlim([xmin, xmax])
ax2.set_ylim([ymin, ymax])
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position("right")

plt.show()
