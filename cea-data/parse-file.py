import pandas as pd
import datetime as dt
import numpy as np
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="my_request")

locator = Nominatim(user_agent="myGeocoder")

f = open("installed-capacity-raw.csv", "r", errors='ignore')
L = f.readlines()

output = {"serial":[],"lon":[],"lat":[],"capacity_MW":[],"installed":[], "state":[], "address":[]}

all_species = ['Solar', 'Small Hydro', 'Bio Mass', 'Bio-Mass', 'Biomass', 'Bio-mass', 'Wind']
species = ['Solar',]
state_list = [s.strip() for s in open("state-list.csv", "r").readlines()]
print(state_list)
NX=0

for line in L:
	line = line.strip()
	line_split = line.split(",")
	line_split_clean = list(filter(None, line_split))
	
	try:
		int(line_split_clean[0])
	except: continue
	
	contains_species = [(S in l) or (S.lower() in l) for S in species for l in line_split_clean]
	if not any(contains_species):
		continue
	index = np.where(contains_species)[0][0]
	line_split_clean.pop(index)
	
	
	if len(line_split_clean)<4: continue
	
	serial_number = int(line_split_clean.pop(0))
	
	state=""
	for S in state_list:
		if (S in line_split_clean) or (S.lower() in line_split_clean):
			state = S
			index = line_split_clean.index(S)
			line_split_clean.pop(index)
			print(state)
	
	capacity = np.nan
	for index,F in enumerate(line_split_clean):
		try:
			f = float(F)
			capacity = f
			line_split_clean.pop(index)
			print(capacity)
		
		except:
			pass
			
	
	date = np.nan
	for index,D in enumerate(line_split_clean):
		try:
			d = dt.datetime.strptime(D, '%d-%b-%Y')
			date = d
			line_split_clean.pop(index)
			print(date)
		except Exception as inst:
			pass
	
	loc_string = ""
	for W in line_split_clean:
		if W.capitalize() in all_species:
			continue
		W = W.replace("\"","").replace("dist","").replace("Dist","").strip()
		loc_string+=W.capitalize()
		loc_string+=", "
	if state: loc_string += state
	loc_string += ", India"
		
	print(loc_string)
	location = geolocator.geocode(loc_string)
	
	if location is None:
		loc_string_clip = loc_string[loc_string.find(", ")+2:]
		location = geolocator.geocode(loc_string_clip)
	
	if location is None:
		loc_string_clip2 = loc_string_clip[loc_string_clip.find(", ")+2:]
		if loc_string_clip2 != "India":
			location = geolocator.geocode(loc_string_clip2)
	
	if location:
		print((location.latitude, location.longitude))
		lon = location.longitude
		lat = location.latitude
	else:
		lon, lat = np.nan, np.nan
	
	output['serial'].append(serial_number)
	output['lon'].append(lon)
	output['lat'].append(lat)
	output['capacity_MW'].append(capacity)
	output['installed'].append(date)
	output['state'].append(state)
	output['address'].append(loc_string)
	
	

df = pd.DataFrame.from_dict(output)

df.to_csv("parsed_solar.csv")






	
