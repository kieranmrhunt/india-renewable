from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy import signal
from statsmodels.tsa.seasonal import STL

from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import STL

import os


def load_data(state_filename):
	state_name = state_filename.replace("_", " ")
	if state_filename == "Andaman_and_Nicobar":
		state_name = "Andaman and Nicobar Islands"
	if state_filename == "NCT_of_Delhi":
		state_name = "Delhi"
	

	df = pd.read_csv(f"../../demand/regression/data/daily/{state_filename}.csv")
	df_energy = pd.read_csv("../../demand/posoco/corrected_daily_energy_met_MU.csv")	
	df_energy = df_energy[['Date', state_name]]
	df_energy.rename(columns={'Date':'date', state_name:'energy_met_MU'}, inplace=True)
	return df, df_energy

def apply_lowess_smoothing(data, frac=0.1):
	"""Apply LOESS smoothing to the data."""
	lowess_results = lowess(data, range(len(data)), frac=frac)
	return lowess_results[:, 1]

def adjust_variables(df):
	for col in df.columns:
		if 'temperature' in col:
			df[col] = df[col] - 273.15  # Convert Kelvin to Celsius
		elif 'solar' in col:
			df[col] = df[col] / 3600  # Convert J/m^2 to W/m^2
	return df


def perform_stl(column_data, period):
		stl = STL(column_data, seasonal=7, period=period)
		result = stl.fit()
		return result
	
def variance_corrected(column_data, trend, window, ref_year):
	variance_prev_12_months = column_data.rolling(window=window).std()
	variance_2023 = column_data.loc[column_data.index.year == ref_year].std()
	corrected_values = (column_data - trend) / variance_prev_12_months * variance_2023
	return corrected_values


def process_data(df, df_energy, subdiv_code, detrend=False, trim = True,
                 loess = True, variance_correct = True):
	import holidays
	df = pd.merge(df, df_energy, how='left')
	df['is_holiday'] = [0 if int(date[:4]) < 2001 else int(date in holidays.country_holidays('IN', subdiv=subdiv_code)) for date in df.date.values]
	df['date'] = pd.to_datetime(df['date'])
	df['day_of_week'] = df['date'].dt.dayofweek + 1
	df['day_of_year'] = df['date'].dt.dayofyear 
	df = df[df['date'].notna()]
	
	if trim:
		df = df[df['energy_met_MU'].notna()]
	
	if detrend:
		df['energy_met_MU'] = signal.detrend(df['energy_met_MU'].values)
		
	

	if loess:
		stl = STL(df['energy_met_MU'], seasonal=7, period=365)
		result = stl.fit()
		trend = result.trend
		df['energy_met_MU'] = df['energy_met_MU'] - trend
	
	if variance_correct:
		variance_prev_12_months = df['energy_met_MU'].rolling(window=365).var()
		variance_2023 = df.loc[df['date'].dt.year == 2023, 'energy_met_MU'].var()
		df['energy_met_MU'] = df['energy_met_MU'] / variance_prev_12_months * variance_2023


	
	for column in df.columns:
		if column.endswith("_mean"):
			new_col_name = f"rolling_avg_30d_{column}"
			df[new_col_name] = df[column].rolling(window=30).mean()
			new_col_name = f"rolling_avg_7d_{column}"
			df[new_col_name] = df[column].rolling(window=7).mean()
			
			new_col_name = f"anomaly_against_30d_{column}"
			df[new_col_name] = df[column]-df[column].rolling(window=30).mean()


	return df

def get_largest_geom(group):
	projected_group = group.to_crs(epsg=32644)  # Reproject to UTM Zone 44N
	return group.loc[projected_group.geometry.area.idxmax()]

def plot_states_by_grid(shapefile_path = "../other-files/india_states.shp", grid_x=8, grid_y=8):
	import geopandas as gpd
	# Load the shapefile with geopandas
	merged = gpd.read_file(shapefile_path)
	merged = merged.groupby('ST_NM').apply(get_largest_geom).reset_index(drop=True)
	merged['centroid_x'] = merged.geometry.centroid.x
	merged['centroid_y'] = merged.geometry.centroid.y

	# Define a grid of subplots
	x_vals = np.linspace(merged['centroid_x'].min(), merged['centroid_x'].max(), grid_x)
	y_vals = np.linspace(merged['centroid_y'].max(), merged['centroid_y'].min(), grid_y)

	fig, axes = plt.subplots(grid_y, grid_x, figsize=(9, 9))

	# Keep track of which grid positions have been used
	used_positions = []

	# Sort states by area
	merged['area'] = merged.geometry.area
	exclude_states = ["Ladakh", "Lakshadweep", "Dadara & Nagar Havelli", "Andaman & Nicobar Island", "Daman & Diu"]
	sorted_states = merged[~merged['ST_NM'].isin(exclude_states)].sort_values(by='area', ascending=True)
	
	ax_dict = {}
	# For each state, find the nearest unused subplot
	for idx, data in sorted_states.iterrows():
		distances = [(abs(data['centroid_x'] - x) + abs(data['centroid_y'] - y), (i, j))
					 for i, x in enumerate(x_vals) for j, y in enumerate(y_vals)]
		distances = sorted(distances, key=lambda x: x[0])

		for _, (i, j) in distances:
			if (i, j) not in used_positions:
				ax = axes[j, i]
				ax_dict[data['ST_NM']] = ax
				used_positions.append((i, j))
				break

		# Convert geometry to GeoSeries and plot
		#gpd.GeoSeries([data['geometry']]).plot(ax=ax, color='black')
		#ax.set_title(data['ST_NM'])
		ax.axis('off')  # Turn off the axis

	# Turn off all axes, then turn on only the used ones
	for ax_row in axes:
		for ax in ax_row:
			ax.axis('off')
	
	for state_ax in ax_dict.values():
		state_ax.axis('on')
	
	return fig, ax_dict


def evaluate_model(xg_reg, X_test, y_test, preds, state='', verbose=True, **kwargs):
	rmse = np.sqrt(mean_squared_error(y_test, preds))
	sn = rmse / np.std(y_test.values)
	_,_,r_value,_,_ = linregress(y_test.values, preds)
	
	if verbose:
		print(f"State: {state}")
		print("RMSE:", rmse)
		print("SN:", sn)
		print(linregress(y_test.values, preds))
		print('-'*50)
	
	return r_value, xg_reg.n_estimators


feature_names_full = [     '10m_u_component_of_wind_mean',
			   '10m_v_component_of_wind_mean', 
			   '2m_temperature_min', 
			   '2m_temperature_mean', 
			   '2m_temperature_max',
			   '2m_dewpoint_temperature_min', 
			   '2m_dewpoint_temperature_mean',
			   '2m_dewpoint_temperature_max', 
			   'surface_solar_radiation_downwards_mean',
			   'surface_solar_radiation_downwards_max', 
			   'total_cloud_cover_mean', 
			   'total_cloud_cover_max', 
			   'is_holiday', 
			   'day_of_week',
			   'day_of_year',
			   'rolling_avg_30d_2m_temperature_mean',
			   'rolling_avg_7d_2m_temperature_mean',
			   'rolling_avg_30d_2m_dewpoint_temperature_mean',
			   'rolling_avg_7d_2m_dewpoint_temperature_mean',
			   'rolling_avg_30d_surface_solar_radiation_downwards_mean',
			   'rolling_avg_7d_surface_solar_radiation_downwards_mean',
			   'rolling_avg_30d_total_cloud_cover_mean',
			   'rolling_avg_7d_total_cloud_cover_mean', 
			]

feature_names_xai = [      'is_holiday', 
			   'day_of_week',
			   'rolling_avg_30d_10m_u_component_of_wind_mean',
			   'anomaly_against_30d_10m_u_component_of_wind_mean',
			   'rolling_avg_30d_10m_v_component_of_wind_mean',
			   'anomaly_against_30d_10m_v_component_of_wind_mean',
			   'rolling_avg_30d_utci_mean',
			   'anomaly_against_30d_utci_mean']



def train_model(X, y, df, best_params, **kwargs):
	import xgboost as xgb
	if 'year' in kwargs:
		# Filter based on the specified year
		year_mask = df['date'].dt.year == kwargs['year']
		X_test = X_val = X[year_mask]
		y_test = y_val = y[year_mask]
		
		X_train = X[~year_mask]
		y_train = y[~year_mask]

		
	else:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
		X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.25, random_state=123)

	
	# Generate flags for train, test, and validation directly using indices
	train_flags = df.index.isin(X_train.index).astype(int)
	test_flags = df.index.isin(X_test.index).astype(int)
	val_flags = df.index.isin(X_val.index).astype(int)

	# Build the DataFrame
	data_splits_df = pd.DataFrame({
		'date': df['date'],
		'train': train_flags,
		'test': test_flags,
		'val': val_flags
	})

	# Save to CSV
	data_splits_df.to_csv('data_splits.csv', index=False)
	
	eval_set = [(X_val, y_val)]
	n_estimators = 200

	# Rename the lambda_ key to lambda for XGBoost
	if 'lambda_' in best_params:
		best_params['lambda'] = best_params.pop('lambda_')

	best_params['objective'] = 'reg:squarederror'
	best_params['max_depth'] = int(best_params['max_depth'])

	xg_reg = xgb.XGBRegressor(n_estimators=n_estimators, early_stopping_rounds=50, **best_params)
	xg_reg.fit(X_train, y_train, verbose=0, eval_set=eval_set)

	preds = xg_reg.predict(X_test)
	
	return xg_reg, X_test, y_test, preds


hyperparameters_df = pd.read_csv("outputs/best_hyperparameters.csv")
hyperparameters_dict = hyperparameters_df.set_index('state').T.to_dict()



states_subdiv_mapping = {
	#"Andaman_and_Nicobar": "AN",
	"Andhra_Pradesh": "AP",
	"Arunachal_Pradesh": "AR",
	"Assam": "AS",
	"Bihar": "BR",
	"Chhattisgarh": "CG",
	"Chandigarh": "CH",
	#"Daman_and_Diu": "DD",
	#"Dadra_and_Nagar_Haveli": "DH",
	"NCT_of_Delhi": "DL",
	"Goa": "GA",
	"Gujarat": "GJ",
	"Himachal_Pradesh": "HP",
	"Haryana": "HR",
	"Jharkhand": "JH",
	"Jammu_and_Kashmir": "JK",
	"Karnataka": "KA",
	"Kerala": "KL",
	#"Lakshadweep": "LD",
	"Maharashtra": "MH",
	"Meghalaya": "ML",
	"Manipur": "MN",
	"Madhya_Pradesh": "MP",
	"Mizoram": "MZ",
	"Nagaland": "NL",
	"Odisha": "OR",
	"Punjab": "PB",
	"Puducherry": "PY",
	"Rajasthan": "RJ",
	"Sikkim": "SK",
	"Tamil_Nadu": "TN",
	"Tripura": "TR",
	"Telangana": "TS",
	"Uttarakhand": "UK",
	"Uttar_Pradesh": "UP",
	"West_Bengal": "WB"
}
