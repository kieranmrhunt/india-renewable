import pandas as pd
import numpy as np

import cartopy
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import geopandas
from cartopy.io import shapereader
from cartopy.feature import ShapelyFeature

cmap = LinearSegmentedColormap.from_list('cmap',['grey','gold','tab:red'], N=15)

all_df = pd.read_csv("parsed-cea-all.csv")
flex_df = pd.read_csv("parsed_solar-OSM.csv")


state_totals_df = pd.read_csv("state-totals.csv")

state_list = state_totals_df['state'].values

# compare and bias-correct wind data

cea_wind = all_df.loc[all_df['type'].isin(['Solar',])]
wind_total_by_state = state_totals_df[['state','solar']]

print(wind_total_by_state)

cea_grouped = cea_wind.groupby(['state']).sum().reset_index()
flex_grouped = flex_df.groupby(['state']).sum().reset_index()

#print(cea_grouped)
print(flex_grouped)

best_wind_bc = {'lon':[], 'lat':[], 'capacity':[]}
cea_wind_bc = {'lon':[], 'lat':[], 'capacity':[]}
flex_wind_bc = {'lon':[], 'lat':[], 'capacity':[]}


for state in state_list:
	actual = wind_total_by_state[wind_total_by_state.state==state]['solar'].values
	cea = cea_grouped[cea_grouped.state==state]['capacity_MW'].values
	flex = flex_grouped[flex_grouped.state==state]['capacity_MW'].values
	
	if actual==0: continue
	
	if len(cea)==0: 
		cea = np.array([0])
	if len(flex)==0:
		flex = np.array([0])
	
	cea_factor = actual/cea
	flex_factor = actual/flex
	
	print(state, cea_factor, flex_factor)
	
	cea_state = cea_wind[cea_wind.state==state]
	cea_wind_bc['lon'].extend(cea_state.lon.values)
	cea_wind_bc['lat'].extend(cea_state.lat.values)
	cea_wind_bc['capacity'].extend(cea_state.capacity_MW.values*cea_factor)
	
	flex_state = flex_df[flex_df.state==state]
	flex_wind_bc['lon'].extend(flex_state.lon.values)
	flex_wind_bc['lat'].extend(flex_state.lat.values)
	flex_wind_bc['capacity'].extend(flex_state.capacity_MW.values)
	
	
	#if np.abs(cea_factor-1)<=np.abs(flex_factor-1):
	#	best_wind_bc['lon'].extend(cea_state.lon.values)
	#	best_wind_bc['lat'].extend(cea_state.lat.values)
	#	best_wind_bc['capacity'].extend(cea_state.capacity_MW.values*cea_factor)
	#else:
	#	best_wind_bc['lon'].extend(flex_state.lon.values)
	#	best_wind_bc['lat'].extend(flex_state.lat.values)
	#	best_wind_bc['capacity'].extend(flex_state.capacity_MW.values*flex_factor)
	
	
	

best_wind_bc = pd.DataFrame.from_dict(best_wind_bc)
cea_wind_bc = pd.DataFrame.from_dict(cea_wind_bc)
flex_wind_bc = pd.DataFrame.from_dict(flex_wind_bc)
	
plt.figure(figsize = (12,5))
axes = [plt.subplot(1,3,i, projection=cartopy.crs.PlateCarree()) for i in [1,2,3]]

dx = 1
gridx = np.arange(60,100,dx)
gridy = np.arange(5,40,dx)
p = []

for ax, df in zip(axes, [cea_wind_bc,]):# flex_wind_bc, best_wind_bc]):
	total_power = np.zeros((len(gridy),len(gridx)))
	lons = df.lon.values
	lats = df.lat.values 
	caps = df.capacity.values
	
	for j in range(len(gridy)):
		ymin = gridy[j]-dx/2.
		ymax = ymin+dx
		
		for i in range(len(gridx)):
			xmin = gridx[i]-dx/2.
			xmax = xmin+dx
			
			ix = np.logical_and.reduce([lons<=xmax,lons>xmin,lats<=ymax,lats>ymin])
			#print(np.sum(ix), xmin, xmax, ymin, ymax)
			total_power[j,i] = np.nansum(caps[ix])
	
	print(total_power.min(), total_power.max())
	np.save("gridded-solar-cap.npy", total_power/1000)
	total_power[total_power<=0]=np.nan
	cs = ax.pcolormesh(gridx,gridy,total_power/1000.,vmin=0,vmax=3,cmap = cmap)
	

kw_arr = np.load("../kruitwagen/gridded-capactity.npy")
kw_arr[kw_arr<=0] = np.nan
cs = axes[1].pcolormesh(gridx,gridy,kw_arr,vmin=0,vmax=3,cmap = cmap)
	

#cs = axes[2].pcolormesh(gridx,gridy,np.mean(p,axis=0)/1000.,vmin=0,vmax=3,cmap = plt.cm.hot_r)


df = pd.read_csv("/home/users/kieran/ncas/india-renewable/capacity/data/installed_oct22.dat", sep='\t', encoding_errors='ignore')
df['solar_density'] = (111**2)/1000*df['Solar']/df['Area'] 
states_df = geopandas.read_file("/home/users/kieran/geodata/india_states_lores.zip").drop_duplicates()
merged = states_df.merge(df, how='left', left_on='ST_NM', right_on='State')
merged.plot(ax=axes[2], edgecolor='k', linewidth=0.25, column='solar_density', vmin=0, vmax=3, cmap=cmap)

print(merged)









df = geopandas.read_file("/home/users/kieran/geodata/india_states_lores.zip")
shape_feature = ShapelyFeature(shapereader.Reader("/home/users/kieran/geodata/ne_10m_IND_lores/ne_10m_admin_0_countries_ind.shp").geometries(),
                                cartopy.crs.PlateCarree(), facecolor='none', edgecolor='k', lw=0.75)


for ax in axes:
	ax.coastlines()
	df.plot(ax=ax, facecolor='none', edgecolor='k', linewidth=0.25)
	ax.add_feature(shape_feature)
	ax.set_xticks(np.arange(0,100,10))
	ax.xaxis.set_major_formatter(LONGITUDE_FORMATTER)
	ax.set_yticks(np.arange(0,100,10))
	ax.yaxis.set_major_formatter(LATITUDE_FORMATTER)
	
	ax.set_xlim([65,98])
	ax.set_ylim([5,38])

axes[0].set_title("(a) Central Electricity Authority")
axes[1].set_title("(b) satellite-derived")
axes[2].set_title("(c) MNRE state totals")


cax = plt.gcf().add_axes([0.925, 0.2, 0.0125, 0.6])
cb = plt.colorbar(cs, cax=cax, orientation='vertical', extend='max')
cb.set_label("Installed solar capacity [GW (100 km)$^{-2}$]")

plt.show()
	
	
	


