import pandas as pd

def convert_and_save(infile, outfile):
	df = pd.read_csv(infile, delimiter='\t')
	df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
	df.to_csv(f'data-for-export/{outfile}', index=False)


convert_and_save('data/wind-actual.dat', 'POSOCO_reported_wind_MU_daily.csv')
convert_and_save('data/solar-actual.dat','POSOCO_reported_solar_MU_daily.csv')
convert_and_save('data/hydro-actual.dat','POSOCO_reported_hydro_MU_daily.csv')
