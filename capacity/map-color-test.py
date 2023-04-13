import numpy as np
from netCDF4 import Dataset, num2date
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cartopy
from cartopy.io import shapereader
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from cartopy.feature import ShapelyFeature
import geopandas
import pandas as pd

df = pd.read_csv("/home/users/kieran/ncas/india-renewable/capacity/data/installed_oct22.dat", sep='\t', encoding_errors='ignore')
df['hydro_density'] = 1e4*(df['Large_hydro']+df['Small_hydro'])/df['Area'] 
df['solar_density'] = 1e4*df['Solar']/df['Area'] 
df['wind_density'] = 1e4*df['Wind']/df['Area'] 
print(df)



fig = plt.figure(figsize=(8,8))

ax = plt.subplot(1,1,1, projection=cartopy.crs.PlateCarree())

states_df = geopandas.read_file("/home/users/kieran/geodata/india_states_lores.zip").drop_duplicates()

print(states_df)


print(set(states_df['ST_NM'].values))


merged = states_df.merge(df, how='left', left_on='ST_NM', right_on='State')
print(merged)

merged.plot(ax=ax, edgecolor='k', linewidth=0.25, column='wind_density')
plt.show()
