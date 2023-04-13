import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt
import numpy as np

df = pd.read_csv("data/tabulated-installed-by-date.csv")
df['Biomass'] = df['Bio Cogen']+df['Bio Waste']

dates = np.array([dt.strptime(t,"%d-%b-%y") for t in df['Date'].values])



fig, ax1 = plt.subplots(1,1, figsize=(5.5,5.5))

ax1.plot(dates, df['Hydro']/1000, color='b', label='Large Hydro')
ax1.plot(dates, df['Small Hydro']/1000, color='tab:blue', ls='--', label='Small Hydro')
ax1.plot(dates, df['Wind']/1000, color='tab:grey', label='Wind')
ax1.plot(dates, df['Biomass']/1000, color='tab:red', label='Biomass')
ax1.plot(dates, df['Solar']/1000, color='tab:orange', ls=':', label='Solar')


print(df['Hydro'].values[-1]/df['Hydro'].values[3])
print(df['Small Hydro'].values[-1]/df['Small Hydro'].values[3])
print(df['Wind'].values[-1]/df['Wind'].values[3])
print(df['Biomass'].values[-1]/df['Biomass'].values[3])
print(df['Solar'].values[-1]/df['Solar'].values[3])

offset=0.3

plt.text(dt(2022,8,31),df['Hydro'].values[-4]/1000+offset, 'large hydro (+4.7%)', ha='right', va='bottom', color='b')
plt.text(dt(2022,8,31),df['Small Hydro'].values[-4]/1000+offset, 'small hydro (+12%)', ha='right', va='bottom', color='tab:blue')
plt.text(dt(2022,8,31),df['Wind'].values[-4]/1000+offset, 'wind (+28%)', ha='right', va='bottom', color='tab:grey')
plt.text(dt(2022,8,31),df['Biomass'].values[-4]/1000+offset, 'biomass (+29%)', ha='right', va='bottom', color='tab:red')
plt.text(dt(2022,8,31),df['Solar'].values[-4]/1000+offset+1, 'solar (+417%)', ha='right', va='bottom', color='tab:orange')


plt.xlim([dt(2017,1,1),dt(2022,10,31)])

plt.ylabel("Installed Capacity (GW)")

plt.show()
