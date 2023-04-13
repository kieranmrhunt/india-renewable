import geopandas as gpd
import matplotlib.pyplot as plt

df = gpd.read_file("hydro.geojson")
df['geometry'] = df['geometry'].centroid

print(df)

df.plot()
plt.show()
