import pandas as pd
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim, Bing
import geopy
import geopandas
from shapely.geometry import Point, Polygon
from geopy.extra.rate_limiter import RateLimiter
import numpy as np


geopy.geocoders.options.default_timeout = None


bing_key = 'ApCNxgnrilmnmhwRpdkZ7kc-L6J1M_VkkrSU7aUrGXsXaTgI4K9wy48TVFmxxW6d'
geolocator = Bing(api_key=bing_key)
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=2)

state_shps = geopandas.read_file("/home/users/kieran/geodata/india-states/gadm36_IND_1.shp")

def find_state(lon, lat):
	p = Point(lon, lat)
	pnts = geopandas.GeoDataFrame(geometry=[p,],)
	pnts = pnts.assign(**{key: pnts.within(geom) for key, geom in zip(state_shps.NAME_1,state_shps.geometry)})
	
	in_state = pnts.loc[0][1:].values.astype(bool)
	state_names = pnts.columns[1:].values
	
	if np.sum(in_state):
		return(state_names[in_state][0])
	else:
		return("")




df = pd.read_csv("India_windfarms_production.csv")

output = {"lon":[],"lat":[],"capacity_MW":[], "state":[], "type":[]}

for lon, lat, power, name, city in df[['Longitude','Latitude','Total power','Name','City']].values:
	print(lon,power)
	if lon != '#ND':
		state = find_state(float(lon),float(lat))
		print("found: {}, {}, {}".format(lon, lat, state))
		
		output['lon'].append(lon)
		output['lat'].append(lat)
		output['state'].append(state)
		
	
	else:
		if city =='#ND':
			lstring = name+", India"
		else: 
			lstring = name+", "+city+", India"
	
	
		location = geocode(lstring)
	
		if location is not None:
			lon = location.longitude
			lat = location.latitude
			state = find_state(lon,lat)
			print(lon, lat, state)
			
			output['lon'].append(lon)
			output['lat'].append(lat)
			output['state'].append(state)
		else:
			output['lon'].append("")
			output['lat'].append("")
			output['state'].append("")
	
	if power!='#ND':
		output['capacity_MW'].append(float(power)/1000.)
	else:
		output['capacity_MW'].append("")
	output['type'].append('Wind')
	
df = pd.DataFrame.from_dict(output)

df.to_csv("twp-coords-added.csv")









