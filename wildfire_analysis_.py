import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the dataset
data_path = '../Downloads/areaburntbywildfiresbyweek new.csv'
wildfire_data = pd.read_csv(data_path)

# Display the first few rows of the dataset to understand its structure
wildfire_data.head()

# Check for missing values in the dataset
missing_values = wildfire_data.isnull().sum()

# Display the columns with missing values and the count of missing values
print("Missing values in each column:")
print(missing_values[missing_values > 0])

# Handling missing values
# For simplicity, we will fill missing numeric values with the median of their respective columns
for column in wildfire_data.columns:
    if wildfire_data[column].dtype != 'object':
        median_value = wildfire_data[column].median()
        wildfire_data[column].fillna(median_value, inplace=True)

# Check if all missing values have been filled
print("\nMissing values after handling:")
print(wildfire_data.isnull().sum())

# Extracting columns related to area burnt by wildfires for each year
area_burnt_columns = [col for col in wildfire_data.columns if 'burnt' in col]

# Summarizing key statistics for area burnt by wildfires by year and region
summary_stats = wildfire_data.groupby('Entity')[area_burnt_columns].agg([np.mean, np.median, np.min, np.max])

# Displaying the summary statistics
print("Summary statistics for area burnt by wildfires by year and region:")
print(summary_stats)

# Extracting columns related to years and area burnt
year_columns = [col for col in wildfire_data.columns if 'burnt' in col and 'area' in col]
years = [int(col.split()[-1]) for col in year_columns]

# Creating a dataframe for trend analysis
trend_data = wildfire_data.groupby('Year')[year_columns].sum()

# Plotting trends for area burnt by wildfires over the years
plt.figure(figsize=(12, 6))
for col in year_columns:
    plt.plot(trend_data.index, trend_data[col], label=col)

plt.title('Trend of Area Burnt by Wildfires Over the Years')
plt.xlabel('Year')
plt.ylabel('Total Area Burnt')
plt.legend(title='Year Columns')
plt.grid(True)
plt.show()

# Load world shapefile for mapping
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Merge the wildfire data with the world GeoDataFrame
wildfire_geo = world.merge(wildfire_data, left_on='name', right_on='Entity', how='left')

# Sum the area burnt and PM2.5 emissions for the latest year available (2024)
wildfire_geo['Total Area Burnt 2024'] = wildfire_geo['area burnt by wildfires in 2024']
wildfire_geo['Total PM2.5 Emissions 2024'] = wildfire_geo[' burnt by wildfires in 2023']  # Assuming this column represents PM2.5 emissions

# Plotting the spatial distribution of area burnt
fig, ax = plt.subplots(1, 2, figsize=(20, 10))

wildfire_geo.plot(column='Total Area Burnt 2024', ax=ax[0], legend=True,
                  legend_kwds={'label': "Total Area Burnt by Wildfires in 2024"},
                  cmap='OrRd', missing_kwds={'color': 'lightgrey'})
ax[0].set_title('Spatial Distribution of Area Burnt by Wildfires in 2024')
ax[0].set_axis_off()

# Plotting the spatial distribution of PM2.5 emissions
wildfire_geo.plot(column='Total PM2.5 Emissions 2024', ax=ax[1], legend=True,
                  legend_kwds={'label': "Total PM2.5 Emissions in 2024"},
                  cmap='Blues', missing_kwds={'color': 'lightgrey'})
ax[1].set_title('Spatial Distribution of PM2.5 Emissions in 2024')
ax[1].set_axis_off()

plt.show()

# Selecting only the relevant columns for analysis
burnt_area_columns = [col for col in wildfire_data.columns if 'burnt' in col and 'area' in col]
pm25_columns = [col for col in wildfire_data.columns if 'burnt' in col and 'PM2.5' in col]

# Summing up the total area burnt and PM2.5 emissions for each country over the years
wildfire_data['Total Area Burnt'] = wildfire_data[burnt_area_columns].sum(axis=1)
wildfire_data['Total PM2.5 Emissions'] = wildfire_data[pm25_columns].sum(axis=1)

# Grouping data by 'Entity' to compare different countries and regions
grouped_data = wildfire_data.groupby('Entity')[['Total Area Burnt', 'Total PM2.5 Emissions']].sum()

# Sorting the data to find top countries/regions in terms of area burnt and PM2.5 emissions
top_burnt_areas = grouped_data.sort_values(by='Total Area Burnt', ascending=False).head(10)
top_pm25_emissions = grouped_data.sort_values(by='Total PM2.5 Emissions', ascending=False).head(10)

# Plotting the top countries/regions for area burnt
plt.figure(figsize=(14, 7))
sns.barplot(x=top_burnt_areas['Total Area Burnt'], y=top_burnt_areas.index)
plt.title('Top 10 Countries/Regions by Total Area Burnt')
plt.xlabel('Total Area Burnt')
plt.ylabel('Country/Region')

plt.show()

# Assuming additional columns related to temperature, precipitation, and drought conditions are present in the dataset
# Let's identify numeric columns for correlation analysis
numeric_columns = wildfire_data.select_dtypes(include=[np.number]).columns

# Compute the correlation matrix for the numeric columns
correlation_matrix = wildfire_data[numeric_columns].corr()

# Plotting the heatmap of the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix of Wildfire Data')
plt.show()

# Encode categorical variables
label_encoder = LabelEncoder()
wildfire_data['Entity_encoded'] = label_encoder.fit_transform(wildfire_data['Entity'])
wildfire_data['Code_encoded'] = label_encoder.fit_transform(wildfire_data['Code'])

# Select features and target for the model
features = ['Year', 'Entity_encoded', 'Code_encoded', 'Total Area Burnt']
target = 'Total PM2.5 Emissions'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(wildfire_data[features], wildfire_data[target], test_size=0.2, random_state=42)

# Initialize and train the XGBoost regressor
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
xgb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate the XGBoost model
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

print("XGBoost Model Evaluation:")
print(f"Mean Squared Error: {mse_xgb}")
print(f"R^2 Score: {r2_xgb}")
print(f"Mean Absolute Error: {mae_xgb}")

# Initialize and train the Gaussian Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predict on the test set
y_pred_nb = nb_model.predict(X_test)

# Evaluate the Naive Bayes model
mse_nb = mean_squared_error(y_test, y_pred_nb)
r2_nb = r2_score(y_test, y_pred_nb)
mae_nb = mean_absolute_error(y_test, y_pred_nb)

print("\nNaive Bayes Model Evaluation:")
print(f"Mean Squared Error: {mse_nb}")
print(f"R^2 Score: {r2_nb}")
print(f"Mean Absolute Error: {mae_nb}")