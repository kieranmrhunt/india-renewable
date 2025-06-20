import os
import cdsapi
from datetime import datetime as dt, timedelta as td

c = cdsapi.Client()

years = range(1979,2025)
var_list = ['100m_u_component_of_wind', '100m_v_component_of_wind', '10m_u_component_of_wind',
            '10m_v_component_of_wind', '2m_temperature', 'mean_surface_direct_short_wave_radiation_flux_clear_sky',
            'mean_surface_downward_short_wave_radiation_flux', 'mean_wave_period', 'significant_height_of_combined_wind_waves_and_swell',
            'significant_height_of_wind_waves', 'surface_solar_radiation_downward_clear_sky', 'surface_solar_radiation_downwards',
            'total_cloud_cover', '2m_dewpoint_temperature', '10m_wind_gust_since_previous_post_processing']


for y in years:
	for var in var_list[10::]:
	
		fname = 'data/{}_{}.nc'.format(y,var)
		if os.path.isfile(fname): continue
		print(y,var)
		c.retrieve(
    		'reanalysis-era5-single-levels',
    		{
        		'product_type': 'reanalysis',
        		'format': 'netcdf',
        		'variable': var,
        		'year': str(y),
        		'month': ["{:02d}".format(m) for m in range(1,13)],
        		'day': ["{:02d}".format(d) for d in range(1,32)],
        		'time': [
            		'00:00', '01:00', '02:00',
            		'03:00', '04:00', '05:00',
            		'06:00', '07:00', '08:00',
            		'09:00', '10:00', '11:00',
            		'12:00', '13:00', '14:00',
            		'15:00', '16:00', '17:00',
            		'18:00', '19:00', '20:00',
            		'21:00', '22:00', '23:00',
        		],
        		'area': [
            		40, 65, 5,
            		100,
        		],
    		},
    		fname)



