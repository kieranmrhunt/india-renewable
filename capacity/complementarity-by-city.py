import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

find = lambda x,arr: np.argmin(np.abs(x-arr))


city_coords = {
		  'Jodhpur':[73.0243, 26.2389],
          'Delhi':[77.1025, 28.7041],
		  'Mumbai': [72.8777, 19.0760],
		  'Bangalore':[77.5946, 12.9716],
		  'Trivandrum':[76.9366,8.5241],
		  'Chennai':[80.2707,13.0827],
		  'Kolkata':[88.3639,22.5726],
		  'Itanagar':[93.6053,27.0844],}

solars = {k:[] for k in city_coords}
winds = {k:[] for k in city_coords}

T_ref = 25. 
eff_ref = 0.9 
beta_ref = 0.0042
G_ref = 1000.
 

for year in range(1980,2021):
	print(year)
	swinfile = Dataset("../era5/data/{}_mean_surface_downward_short_wave_radiation_flux.nc".format(year))
	t2infile = Dataset("../era5/data/{}_2m_temperature.nc".format(year))
	wcfinfile = Dataset("data/{}_wind_capacity_factor_by_turbine.nc".format(year))
	lons = swinfile.variables['longitude'][:] 
	lats = swinfile.variables['latitude'][:]
	
	for city in city_coords:
		
		ix, iy  = find(city_coords[city][0], lons), find(city_coords[city][1], lats)	

		sw = swinfile.variables['msdwswrf'][:365*24,iy,ix]
		t2 = t2infile.variables['t2m'][:365*24,iy,ix]-273.15
		rel_efficiency_of_panel = eff_ref*(1-beta_ref*(t2-T_ref))
		capacity_factor = np.nan_to_num(rel_efficiency_of_panel*(sw/G_ref)) 
		capacity_factor = capacity_factor.reshape((-1,24))
		solars[city].append(capacity_factor)


		wcf = wcfinfile.variables["Gamesa_G87_2000MW"][:365*24,iy,ix]
		wcf = wcf.reshape((-1,24))
		winds[city].append(wcf)


fig, axes = plt.subplots(len(city_coords), 2, figsize=(8,10), sharex='col', sharey='col')

for n, city in enumerate(city_coords):
	diurnal_solar = np.roll(np.mean(solars[city], axis=(0,1)),6)
	annual_solar = np.mean(solars[city], axis=(0,2))
	
	diurnal_winds = np.roll(np.mean(winds[city], axis=(0,1)),6)
	annual_winds = np.mean(winds[city], axis=(0,2))
	
	
	axes[n,0].plot(annual_solar, color='tab:orange')
	axes[n,1].plot(np.r_[diurnal_solar,diurnal_solar[0]], color='tab:orange')
	
	axes[n,0].plot(annual_winds, color='tab:blue')
	axes[n,1].plot(np.r_[diurnal_winds,diurnal_winds[0]], color='tab:blue')	
	
	axes[n,0].set_xlim([0,364])
	axes[n,1].set_xlim([0,24])
	
	axes[n,1].set_ylim([0,0.7])
	axes[n,0].set_ylim([0,0.45])
	
	axes[n,0].set_ylabel("{}\n\ncap. fac.".format(city))

axes[-1,0].set_xticks([0,59,120,181,243,304], )
axes[-1,0].set_xticklabels(['Jan 1', 'Mar 1', 'May 1', 'Jul 1', 'Sep 1', 'Nov 1'], 
					 rotation=45)

L = [0,6,12,18,24]
axes[-1,1].set_xticks(L)
axes[-1,1].set_xticklabels(["{:d}h".format(h) for h in L])

axes[0,0].set_title("(a) annual")
axes[0,1].set_title("(b) diurnal")

axes[-1,0].set_xlabel("Date")
axes[-1,1].set_xlabel("Local Time")


import matplotlib.lines as mlines
blue_line = mlines.Line2D([], [], color='tab:blue', label='wind')
orange_line = mlines.Line2D([], [], color='tab:orange', label='solar')

plt.legend(handles=[blue_line, orange_line],loc='upper center', bbox_to_anchor=(0.5, -0.65),ncol=2)


plt.show()
