import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors
import matplotlib.lines as mlines
import cartopy
from cartopy.io import shapereader
from cartopy.feature import ShapelyFeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import geopandas
import fiona
import shapely.geometry as sgeom
import shapely.vectorized as v
from shapely.prepared import prep
from scipy.interpolate import interp2d
from scipy.ndimage import gaussian_filter as gf

geoms = fiona.open(shapereader.natural_earth(resolution='50m', category='physical', name='land'))
land_geom = sgeom.MultiPolygon([sgeom.shape(geom['geometry']) for geom in geoms])
land = prep(land_geom)

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
			coast_distance[j, i] = land_geom.distance(sgeom.Point(x_pt, y_pt))
	coast_distance *= 111*np.cos(Y*np.pi/180.)
	return(coast_distance)

def haversine(lon1_, lat1_, lon2_, lat2_):
	lon1=lon1_*np.pi/180.
	lat1=lat1_*np.pi/180.
	lon2=lon2_*np.pi/180.
	lat2=lat2_*np.pi/180.
	dlon = lon2 - lon1 
	dlat = lat2 - lat1 
	a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
	c = 2 * np.arcsin(np.sqrt(a)) 
	r = 6371.
	return c * r 


xmin, xmax = 65, 100
ymin, ymax = 5, 40


fig = plt.figure(figsize=(8,8))

ax = plt.subplot(1,1,1, projection=cartopy.crs.PlateCarree())

oro = np.load("data/era5_orography.npy")

wind = np.load("data/means/windcapacity.npy")
solar = np.load("data/means/solarcapacity.npy")
waves = np.load("data/means/wavemean.npy")
hydro = np.load("data/means/hydropower.npy")

lulc = np.load("data/means/lulc.npy")
trees = np.load("data/means/treecoverage.npy")


###wind
lons = np.linspace(xmin,xmax,wind.shape[1])
lats = np.linspace(ymax,ymin,wind.shape[0])

land_mask = make_landmask(lons, lats)
coast_distance = dist_from_coast(lons, lats)

offwind = wind*(1-land_mask)*(coast_distance<250)
thresh = np.percentile(offwind[offwind>0], 80)
print(thresh)

cs = ax.contourf(lons,lats, offwind, levels = [thresh,1e10], hatches=['\\\\\\',], colors='none', zorder=2)
for i, collection in enumerate(cs.collections):
	collection.set_edgecolor('tab:blue')
	

onwind = wind*land_mask*(oro<2000)
thresh = np.percentile(onwind[onwind>0], 85)
print(thresh)

cs = ax.contourf(lons,lats, onwind, levels = [thresh,1e10], hatches=['\\\\\\',], colors='none',zorder=2)
for i, collection in enumerate(cs.collections):
	collection.set_edgecolor('tab:red')
	

###solar
isolar = solar*land_mask*(oro<3000)
thresh = np.percentile(isolar[isolar>0], 70)
print(thresh)
cs = ax.contourf(lons,lats, isolar, levels = [thresh,1e10], colors='yellow', zorder=1)

isolar = solar*land_mask*(oro>3000)
thresh = np.percentile(isolar[isolar>0], 90)
print(thresh)
cs = ax.contourf(lons,lats, isolar, levels = [thresh,1e10], colors='orange', zorder=1)


###waves
lons = np.linspace(xmin,xmax,waves.shape[1])
lats = np.linspace(ymax,ymin,waves.shape[0])

thresh = np.percentile(waves[waves>0], 75)
print(thresh)
cs = ax.contourf(lons,lats, waves, levels = [thresh,1e10], colors='gray', zorder=1)


###biomass
lons = np.linspace(xmin,xmax,lulc.shape[1])
lats = np.linspace(ymax,ymin,lulc.shape[0])

tlons = np.linspace(xmin,xmax,trees.shape[1])
tlats = np.linspace(ymax,ymin,trees.shape[0])

interp_trees = interp2d(tlons, tlats, trees)(lons, lats)
valid = gf(np.in1d(lulc, [2,3,5,6]).reshape((len(lats),len(lons))),(1,1))
good_land = interp_trees*valid

thresh = np.percentile(good_land[good_land>0], 90)
print(thresh)

cs = ax.contourf(lons,lats, good_land, levels = [thresh,1e10], colors='forestgreen', zorder=1)


###hydro
##large hydro
n_plants = 35
clearance = 150 #km
hcopy = hydro.copy()
lons = np.linspace(xmin,xmax,hydro.shape[1])
lats = np.linspace(ymin,ymax,hydro.shape[0])
X, Y = np.meshgrid(lons, lats)
hcopy[X>97]=0

locx, locy = [], []
for _ in range(n_plants):
	max_hydro = np.max(hcopy)
	y, x = np.where(hcopy==max_hydro)
	x, y = lons[x[0]], lats[y[0]]
	locx.append(x)
	locy.append(y)
	
	hcopy[haversine(x,y,X,Y)<clearance] = 0
	
	#print(x,y)

#ax.pcolormesh(X,Y,hydro, cmap=plt.cm.Reds, norm=colors.LogNorm(vmin=1e-1, vmax=1e5))
ax.plot(locx, locy, 'bo', label="Large hydro")


df = geopandas.read_file("/home/users/kieran/geodata/india_states_lores.zip")
df.plot(ax=ax, facecolor='none', edgecolor='k', linewidth=0.25)

ax.set_xticks(np.arange(0,100,10))
ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
ax.set_yticks(np.arange(0,100,10))
ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)

shape_feature = ShapelyFeature(shapereader.Reader("/home/users/kieran/geodata/ne_10m_IND_lores/ne_10m_admin_0_countries_ind.shp").geometries(),
                                cartopy.crs.PlateCarree(), facecolor='none', edgecolor='k', lw=0.75)


ax.add_feature(shape_feature)
ax.coastlines()
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])

handles = []
handles.append(plt.Rectangle((0,0),1,1, fc= 'none', ec = 'tab:blue', hatch='\\\\\\', label='Offshore wind'))
handles.append(plt.Rectangle((0,0),1,1, fc= 'none', ec = 'tab:red', hatch='\\\\\\', label='Onshore wind'))
handles.append(plt.Rectangle((0,0),1,1, fc= 'yellow', label='Solar'))
handles.append(plt.Rectangle((0,0),1,1, fc= 'orange', label='High-altitude solar'))
handles.append(plt.Rectangle((0,0),1,1, fc= 'grey', label='Wave'))
handles.append(plt.Rectangle((0,0),1,1, fc= 'forestgreen', label='Biomass'))
handles.append(mlines.Line2D([], [], color='blue', marker='o', lw=0, label='Large hydro'))

plt.legend(handles = handles, loc='upper right', framealpha=1)
plt.show()



