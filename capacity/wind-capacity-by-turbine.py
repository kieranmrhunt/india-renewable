import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
from scipy.interpolate import InterpolatedUnivariateSpline




turbines = ["Gamesa_G87_2000MW","Vestas_v110_2000MW", "Enercon_E70_2300MW"]


for year in np.arange(2020,2023):
	print(year)
	uinfile = xr.open_dataset("../era5/data/{}_100m_u_component_of_wind.nc".format(year))
	vinfile = xr.open_dataset("../era5/data/{}_100m_v_component_of_wind.nc".format(year))

	u = uinfile['u100'].values
	v = vinfile['v100'].values
	ws = np.hypot(u,v)

	if year==2022:
		ws = ws[:,0]
	print(ws.shape)

	lons = uinfile['longitude'] 
	lats = uinfile['latitude'].values

	times = uinfile['time']
	
	
	ds = xr.Dataset(coords={'lon': lons, 'lat': lats, 'time': times})
	
	for turbine in turbines:
		print(turbine)
		power_curve = np.genfromtxt('../energy-model-scripts/extra_files_to_run_scripts/{}_ECEM_turbine.csv'.format(turbine))
		pc_w = power_curve[:,0]
		pc_c = power_curve[:,2]

		pc_interpolate = InterpolatedUnivariateSpline(pc_w, pc_c)
		capacity = pc_interpolate(ws)

	
		ds[turbine] = (('time', 'lat', 'lon'), capacity)


	ds.to_netcdf("data/{}_wind_capacity_factor_by_turbine.nc".format(year))
