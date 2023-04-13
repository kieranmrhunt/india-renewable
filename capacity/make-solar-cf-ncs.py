from netCDF4 import Dataset, num2date
import numpy as np

def write_netcdf(year, lats, lons, times, capacity_factor_of_panel):
    with Dataset(f"data-for-export/solar_capacity_factor_{year}.nc", "w") as ncfile:
        # Create dimensions
        ncfile.createDimension("time", size=None)
        ncfile.createDimension("lat", size=len(lats[:]))
        ncfile.createDimension("lon", size=len(lons[:]))

        # Create variables
        ncfile.createVariable("time", "f8", ("time",))
        ncfile.createVariable("lat", "f4", ("lat",))
        ncfile.createVariable("lon", "f4", ("lon",))
        ncfile.createVariable("capacity_factor_of_panel", "f4", ("time", "lat", "lon"))

        # Assign values to dimensions
        ncfile.variables["lat"][:] = lats
        ncfile.variables["lon"][:] = lons

        # Assign values to variables
        ncfile.variables["time"][:] = time[:]
        ncfile.variables["time"].setncattr("units", time.units)
		
        ncfile.variables["capacity_factor_of_panel"][:] = capacity_factor_of_panel



for year in range(1979,2023):
	
	print(year)#, end="\r")
	swinfile = Dataset("../era5/data/{}_mean_surface_downward_short_wave_radiation_flux.nc".format(year))
	t2infile = Dataset("../era5/data/{}_2m_temperature.nc".format(year))

	sw_data = swinfile.variables['msdwswrf'][:]#[:365*24]
	if year == 2022:
		sw_data = sw_data[:,0]
	
	
	t2_data = t2infile.variables['t2m'][:]-273.15
	if year == 2022:
		t2_data = t2_data[:,0]
	

	lons = swinfile.variables['longitude'][:]
	lats = swinfile.variables['latitude'][:]
	
	time = t2infile.variables['time']
	times = num2date(time[:], time.units)
	
	
	T_ref = 25. 
	eff_ref = 0.9 
	beta_ref = 0.0042
	G_ref = 1000.

	rel_efficiency_of_panel = eff_ref*(1 - beta_ref*(t2_data - T_ref))
	capacity_factor_of_panel = np.nan_to_num(rel_efficiency_of_panel*
                                        	(sw_data/G_ref)) 

	write_netcdf(year, lats, lons, time, capacity_factor_of_panel)
