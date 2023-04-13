import geopandas 
import matplotlib.pyplot as plt
import cartopy

states_df = geopandas.read_file("/home/users/kieran/geodata/india_states_lores.zip").drop_duplicates()

print(list(set(states_df['ST_NM'].values)))

regions = {'Assam': 'NER',
		   'Dadara & Nagar Havelli': 'WR', 
		   'Maharashtra': 'WR', 
		   'Madhya Pradesh': 'WR', 
		   'Arunachal Pradesh': 'NER', 
		   'Bihar': 'ER', 
		   'Tripura': 'NER', 
		   'Lakshadweep': 'I', 
		   'Haryana': 'NR', 
		   'Daman & Diu': 'WR', 
		   'NCT of Delhi': 'NR', 
		   'Meghalaya': 'NER', 
		   'Andhra Pradesh': 'SR', 
		   'Sikkim': 'ER', 
		   'Manipur': 'NER', 
		   'Rajasthan': 'NR', 
		   'Kerala': 'SR', 
		   'Nagaland': 'NER', 
		   'Jammu & Kashmir': 'NR', 
		   'Jharkhand': 'ER', 
		   'Tamil Nadu': 'SR', 
		   'Gujarat': 'WR', 
		   'Uttar Pradesh': 'NR', 
		   'Karnataka': 'SR', 
		   'Ladakh': 'NR', 
		   'Mizoram': 'NER', 
		   'Andaman & Nicobar Island': 'I', 
		   'Goa': 'WR', 
		   'Himachal Pradesh': 'NR', 
		   'West Bengal': 'ER', 
		   'Telangana': 'SR', 
		   'Chandigarh' : 'NR', 
		   'Uttarakhand': 'NR', 
		   'Chhattisgarh': 'WR', 
		   'Odisha': 'ER', 
		   'Punjab': 'NR', 
		   'Puducherry': 'SR'}


ax = plt.subplot(1,1,1, projection=cartopy.crs.PlateCarree())

color_keys = {'NER':'lightgray','ER':'moccasin','SR':'skyblue','WR':'pink','NR':'lightgreen', 'I':'w'}

for state in regions:
	gf = states_df[states_df.ST_NM==state]
	gf.plot(ax=ax, color=color_keys[regions[state]])

plt.show()
