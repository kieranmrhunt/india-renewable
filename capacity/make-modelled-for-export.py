import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import geopandas
from skimage.measure import block_reduce
from shapely.geometry import Point
from scipy.interpolate import interp2d


solar = []
wind = []

years = np.arange(1979,2023)

for year in years:
	print(year, end="\r")
	sol_year = Dataset("data-for-export/solar_capacity_factor/solar_capacity_factor_{}.nc".format(year)).variables['capacity_factor_of_panel'][:]
	win_year = Dataset("data-for-export/wind_capacity_factor/wind_capacity_factor_{}.nc".format(year)).variables['Vestas_v110_2000MW'][:]
	
	solar.extend(sol_year)
	wind.extend(win_year)
	
solar = np.array(solar)
wind = np.array(wind)


xmin, xmax = 65, 100
ymin, ymax = 5, 40
lons = np.linspace(xmin,xmax,wind.shape[-1])
lats = np.linspace(ymin,ymax,wind.shape[-2])



solar = block_reduce(solar, block_size = (1,4,4), func=np.mean)[:,:-1,:-1]
wind = block_reduce(wind, block_size = (1,4,4), func=np.mean)[:,:-1,:-1]

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

solar_produced = solar*solar_installed[None, :, :] 
wind_produced = wind*wind_installed[None, :, :]

solar_total = np.sum(solar_produced, axis=(-1,-2))*2
wind_total = np.sum(wind_produced, axis=(-1,-2))

total_produced = wind_total + solar_total


import xarray as xr
import pandas as pd

'''
dates = pd.date_range(start='1979-01-01', end='2022-10-31 23:00', freq='H')
modelled_grid_output = xr.DataArray(total_produced, coords=[dates], dims=['time'])
ds = xr.Dataset({'modelled_grid_output': modelled_grid_output})
ds.attrs['description'] = 'Hourly power output from renewables in India, if the grid as of October 2022 were exposed to historical conditions'
ds.to_netcdf('data-for-export/modelled-historical-hourly-renewable_output.nc')


'''
dates = pd.date_range(start='1979-01-01', end='2022-10-31 23:00', freq='H')
hourly_data = xr.DataArray(total_produced, coords=[dates], dims=['time'])
daily_data = hourly_data.resample(time='1D').mean()
ds = xr.Dataset({'modelled_grid_output': daily_data})
ds.attrs['description'] = 'Daily average power output from renewables in India, if the grid as of October 2022 were exposed to historical conditions'
ds.to_netcdf('data-for-export/modelled-historical-daily-renewable_output.nc')




