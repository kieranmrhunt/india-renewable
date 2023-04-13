import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import ShapelyFeature
import geopandas
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

find = lambda x,arr: np.argmin(np.abs(x-arr))

def np_pearson_cor(x, y):
	xv = x - x.mean(axis=0)
	yv = y - y.mean(axis=0)
	xvss = (xv * xv).sum(axis=0)
	yvss = (yv * yv).sum(axis=0)
	result = np.sum(xv*yv, axis=0)/np.sqrt(xvss*yvss)
	return np.maximum(np.minimum(result, 1.0), -1.0)


xmin, xmax = 65, 100
ymin, ymax = 5, 40


city_coords = {
		  'Jodhpur':[73.0243, 26.2389],
          'Delhi':[77.1025, 28.7041],
		  'Mumbai': [72.8777, 19.0760],
		  #'Bangalore':[77.5946, 12.9716],
		  'Hyderabad': [78.4867,17.3850],
		  'Trivandrum':[76.9366,8.5241],
		  'Chennai':[80.2707,13.0827],
		  'Kolkata':[88.3639,22.5726],
		  'Itanagar':[93.6053,27.0844],}

city_names = np.array([c for c in city_coords])
city_lons = np.array([city_coords[city][0] for city in city_names])
city_lats = np.array([city_coords[city][1] for city in city_names])

group1 = np.argsort(city_lons)[:4]
group2 = np.argsort(city_lons)[4:]

g1ord = np.argsort(city_lats[group1])[::-1]
g2ord = np.argsort(city_lats[group2])[::-1]

it = np.r_[group1[g1ord], group2[g2ord]]

city_names = city_names[it]
city_lons = city_lons[it]
city_lats = city_lats[it]

cmap_name = 'BlueWhiteOrangeRed'
rgb = np.genfromtxt("/home/users/kieran/miniconda3/lib/python3.7/site-packages/cfplot/colourmaps/{}.rgb".format(cmap_name))/255.
cmap = LinearSegmentedColormap.from_list(cmap_name, rgb, N=20)


solar_in = np.concatenate([np.load("data/full_arrays/solarcapacity-old.npy"),
                           np.load("data/full_arrays/solarcapacity.npy")])
print(solar_in.shape)
wind_in = np.concatenate([np.load("data/full_arrays/windcapacity-old.npy"),
                           np.load("data/full_arrays/windcapacity.npy")])


lons = np.linspace(xmin,xmax,wind_in.shape[-1])
lats = np.linspace(ymax,ymin,wind_in.shape[-2])

solar = solar_in.reshape((-1,365,len(lats),len(lons)))*2
wind = wind_in.reshape((-1,365,len(lats),len(lons)))

seasonal_comp = np_pearson_cor(solar.mean(axis=0), wind.mean(axis=0))


fig = plt.figure(figsize=(15,8))
gs = GridSpec(8, 3)
ax1 = fig.add_subplot(gs[0:2, 0])
ax2 = fig.add_subplot(gs[2:4, 0])
ax3 = fig.add_subplot(gs[4:6, 0])
ax4 = fig.add_subplot(gs[6:8, 0])
ax5 = fig.add_subplot(gs[0:2, 2])
ax6 = fig.add_subplot(gs[2:4, 2])
ax7 = fig.add_subplot(gs[4:6, 2])
ax8 = fig.add_subplot(gs[6:8, 2])

axc  = fig.add_subplot(gs[:, 1], projection=cartopy.crs.PlateCarree())

axes_left = [ax1, ax2, ax3, ax4]
axes_right = [ax5, ax6, ax7, ax8]
axes = axes_left + axes_right

d = range(365)
for ax, city, cx, cy in zip(axes, city_names, city_lons, city_lats):
	ix, iy  = find(cx, lons), find(cy, lats)
	s90, s10 = np.percentile(solar[...,iy,ix],90, axis=0), np.percentile(solar[...,iy,ix],10, axis=0)
	w90, w10 = np.percentile(wind[...,iy,ix],90, axis=0), np.percentile(wind[...,iy,ix],10, axis=0)
	
	ax.fill_between(d, s90, s10, color='tab:red', alpha=0.25)
	ax.fill_between(d, w90, w10, color='tab:blue', alpha=0.25)
	
	ax.plot(d, solar[...,iy,ix].mean(axis=0), c='tab:red', label='solar capacity factor')
	ax.plot(d, wind[...,iy,ix].mean(axis=0), c='tab:blue', label='wind capacity factor')
	
	city_comp_val = -seasonal_comp[iy,ix]
	ax.text(0.95,0.9, "{:1.2f}".format(city_comp_val), ha='right', va='top', transform=ax.transAxes)
	
	ax.set_xlim([0,364])
	ax.set_ylabel(city)
	
	ax.set_ylim([0,1])
	ax.set_yticks(np.arange(0,1,0.2))
	
	ax.set_xticks([0,59,120,181,243,304], )
	if ax in axes_right: 
		ax.yaxis.set_label_position("right")
		ax.yaxis.tick_right()

for ax in axes:
	ax.set_xticklabels([])

for ax in [ax4,ax8]:
	ax.set_xticklabels(['Jan 1', 'Mar 1', 'May 1', 'Jul 1', 'Sep 1', 'Nov 1'], rotation=45)


cs2 = axc.pcolormesh(lons, lats, -seasonal_comp, cmap=cmap, vmin=-1, vmax=1, rasterized=True)


xcb1 = axc.get_position().x0
xcb2 = axc.get_position().x1
cax2 = plt.gcf().add_axes([xcb1, 0.185, xcb2-xcb1, 0.025])
cb2 = plt.colorbar(cs2, cax=cax2, orientation='horizontal')
cb2.set_label("Complementarity coefficient")




for city in city_coords:
	x, y = city_coords[city]
	axc.plot(x, y, marker='o', mec='k', mfc='lightgrey', lw=0)


df = geopandas.read_file("/home/users/kieran/geodata/india_states_lores.zip")
df.plot(ax=axc, facecolor='none', edgecolor='k', linewidth=0.25)



axc.set_xticks(np.arange(0,100,10))
axc.xaxis.set_major_formatter(LONGITUDE_FORMATTER)

axc.set_yticks(np.arange(0,100,10))
axc.yaxis.set_major_formatter(LATITUDE_FORMATTER)

shape_feature = ShapelyFeature(shapereader.Reader("/home/users/kieran/geodata/ne_10m_IND_lores/ne_10m_admin_0_countries_ind.shp").geometries(),
                                cartopy.crs.PlateCarree(), facecolor='none', edgecolor='k', lw=0.75)

axc.coastlines()
axc.add_feature(shape_feature)
axc.set_xlim([xmin, xmax])
axc.set_ylim([ymin, ymax])

fig.subplots_adjust(wspace=.2, hspace=.125)
fig.canvas.draw()

axcxs = [ax.get_position().x1 for ax in axes_left] + [ax.get_position().x0 for ax in axes_right]
axcys = [(ax.get_position().y0+ax.get_position().y1)/2 for ax in axes]


axis_to_fig = axc.transData + fig.transFigure.inverted()
city_points = axis_to_fig.transform([(x, y) for x,y in zip(city_lons, city_lats)])


cities_x = city_points[:,0]
cities_y = city_points[:,1]

for x0, y0, x1, y1 in zip(axcxs, axcys, cities_x, cities_y):
	fig.add_artist(Line2D([x0,x1], [y0,y1], color='k', linewidth=0.75,linestyle=':'))
	#print([x0,y0], [x1,y1])


ax1.legend(bbox_to_anchor=(1.4, 0.75), loc='upper left')


'''
y1 = ax3.get_position().y0
y2 = ax3.get_position().y1

cax2 = plt.gcf().add_axes([0.85, y1, 0.0125, y2-y1])
cb2 = plt.colorbar(cs2, cax=cax2, orientation='vertical')
cb2.set_label("Correlation coefficient")
'''

plt.show()
