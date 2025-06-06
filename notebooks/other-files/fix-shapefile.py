import geopandas as gpd
import matplotlib.pyplot as plt

# Load the shapefile
gdf = gpd.read_file("india_states.shp")

# 1) Remove all entries with "None" geometry
gdf = gdf[gdf['geometry'].notnull()]

# Fix invalid geometries
gdf['geometry'] = gdf['geometry'].apply(lambda geom: geom if geom.is_valid else geom.buffer(0))

# 2) Merge into a single polygon/entry all lines with the same ST_NM
gdf_merged = gdf.dissolve(by='ST_NM')

# 3) Rename some states via a dictionary
rename_dict = {"Jammu & Kashmir": "Jammu and Kashmir",
               "NCT of Delhi": "Delhi",
               "Andaman & Nicobar Island": "Andaman and Nicobar Islands",
               "Dadara & Nagar Havelli": "Dadra and Nagar Haveli",
               "Daman & Diu": "Daman and Diu",
               }
gdf_merged = gdf_merged.rename(index=rename_dict)

# 4) Print out all ST_NM
print(gdf_merged.index.tolist())

# 5) Save the shapefile
#gdf_merged.reset_index().to_file("india_states_fixed.shp")

gdf_merged.plot()

plt.show()
