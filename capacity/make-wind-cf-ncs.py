import xarray as xr

for year in range(1979,2023):
	
	print(year)
	
	ds = xr.open_dataset(f"data/{year:04d}_wind_capacity_factor_by_turbine.nc")
		
	var = ds['Vestas_v110_2000MW']
	var = var.astype('float32')
	
	var_with_coords = xr.DataArray(var, coords=[ds['time'],ds['lat'],ds['lon']], dims=['time', 'lat', 'lon'])
	var_with_coords = var_with_coords.rename({'lat':'latitude', 'lon':'longitude'})

	var_with_coords.to_netcdf(f"data-for-export/wind_capacity_factor/wind_capacity_factor_{year:04d}.nc")
	

