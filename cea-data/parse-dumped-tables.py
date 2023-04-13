import pandas as pd
import numpy as np
import dateparser
from glob import glob
import re
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim, Bing
import geopy
import geopandas
from shapely.geometry import Point, Polygon
from geopy.extra.rate_limiter import RateLimiter


#geolocator = Nominatim(user_agent="my_request")
#locator = Nominatim(user_agent="myGeocoder")

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



string_in_df = lambda s, df: np.sum([df[col].str.contains(s, flags = re.IGNORECASE).sum() for col in df])
find_col_with_string = lambda s, df: np.where([df[col].str.contains(s, flags = re.IGNORECASE).sum() for col in df])[0][0]


tlist = sorted(glob("cea-table-dump/*.csv"))

output = {"lon":[],"lat":[],"capacity_MW":[],"installed":[], "district":[], "state":[], "type":[]}
#output = {"capacity_MW":[],"installed":[],"district":[],"type":[]}


for nt,tname in enumerate(tlist):
	
	tcode = int(tname.split("/")[-1].split(".")[0])
	if tcode in range(1293,1301): continue #avoid the weird nested tables for part of Maharashtra
	

	#tname = tlist[87]
	print(nt, tname)

	#skip files that are summary headers
	textfile = open(tname, 'r')
	filetext = textfile.read()
	textfile.close()
	matches = re.findall("summary of RE", filetext, re.IGNORECASE)
	if matches: continue

	try:
		df = pd.read_csv(tname, lineterminator="\n").dropna(how='all').dropna(thresh=4)
	except pd.errors.EmptyDataError:
		continue #skip blank files
	
	
	df = df.replace({r'\s+$': '', r'^\s+': ''}, regex=True).replace(r'\r',  ' ', regex=True)
	
	
	print(df)	

	#check for and remove summary row, if present
	if df.iloc[-1].str.contains('[Tt]otal', flags = re.IGNORECASE, na=False).any():
		df = df[:-1]
	
	

	#first check the whole table for clues about the energy source

	try:
		if string_in_df('[Ww]ind', df):
			source = "Wind"
		elif string_in_df('[Ss]olar', df):
			source = "Solar"
		elif string_in_df('[Hh]ydro', df):
			source = "Hydro"
		elif string_in_df('[Bb]io[\s\S]?[Mm]ass', df):
			source = "Biomass"
		else:
			source = "other"

	except:continue #this only fails on the horribly messy nested Maharashtra tables, so skip those
	

	print(source)

	#use the fact that all tables contain some capacity info to figure out the structure of table
	iy = find_col_with_string('capacity', df)
	cap_col = df.iloc[:,iy]
	start_index = np.where(cap_col.str.contains('capacity', flags = re.IGNORECASE, na=False))[0][-1]
	#print(cap_col)
	#print(start_index)

	

	new_header = df.iloc[start_index] 
	
	
	df = df[start_index+1:] 
	df.columns = [str(h).lower() for h in new_header]
	
	#manual fix to sort botched tables in some Karnatka data
	if tname in ["cea-table-dump/{:04d}.csv".format(R) for R in range(98,191)]:	
		df.columns = np.roll(df.columns,-1)

	
	print(df)


	#extract and convert useful data

	cap_raw = np.ravel(df.filter(like='capacity').values)
	capacities = []
	for c in cap_raw:
		if type(c)==str:
			c = re.sub(r'[^0-9\.]', '', c)
			if c:
				if c[0] ==".":
					c = c[1:]
			else:
				capacities.append(np.nan)
				continue
		capacities.append(float(c))
			
	

	date_raw = np.ravel(df.filter(like='date').values)
	dates = [dateparser.parse(d) if type(d)==str else np.nan for d in date_raw]

	if "state" in df.columns:
		states = np.ravel(df.filter(like='state').values)
	else:
		states = np.zeros_like(dates)
		states[:]=""


	locations = np.ravel(df.filter(like='location').values)
	
	#filter out remaining nans
	locations = [L.split("-")[0] if type(L)==str else "" for L in locations]
	states = [S if type(S)==str else states[0] for S in states]

	lstrings = [a.replace("\"","").replace("dist","").replace("Dist","").strip()+", "+b+", India" for a,b in zip(locations,states)]
	print(lstrings)


	
	
	for lstring in lstrings:
		location = geocode(lstring)
		if location is not None:
			lon = location.longitude
			lat = location.latitude
			state = find_state(lon,lat)
			print(lon, lat, state)

		else:
			lon = np.nan
			lat = np.nan
			state = states[0]
		
		output['lon'].append(lon)
		output['lat'].append(lat)
		output['state'].append(state)
	

	types = [source,]*len(capacities)
	
	output['type'].extend(types)
	output['capacity_MW'].extend(capacities)
	output['installed'].extend(dates)
	output['district'].extend(locations)
	
	

df = pd.DataFrame.from_dict(output)

df.to_csv("parsed_with_type.csv")



















