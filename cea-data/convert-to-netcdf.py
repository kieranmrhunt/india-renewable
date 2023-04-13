import xarray as xr
import numpy as np

def process(infile, outfile):
	dx = 1
	gridx = np.arange(60,100,dx)
	gridy = np.arange(5,40,dx)

	data = np.load(infile)
	da=xr.DataArray(data, coords=[gridy,gridx], dims=['latitude','longitude'])
	da.to_netcdf(f"../capacity/data-for-export/{outfile}.nc")


process("gridded-twp-cap.npy","TWP_1x1_gridded_installed_wind_cap")
process("gridded-wind-cap.npy","CEA_1x1_gridded_installed_wind_cap")
process("gridded-solar-cap.npy","CEA_1x1_gridded_installed_solar_cap")
process("../kruitwagen/gridded-capactity.npy","K21_1x1_gridded_installed_solar_cap")
