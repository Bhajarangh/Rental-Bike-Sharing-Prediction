import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Ignore the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# Set seaborn style
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=0.8)

# Streamlit app
st.title("Bike Sharing Data Analysis")

# Read the data with error handling
try:
    mydata = pd.read_csv('train.csv', parse_dates=True, index_col='datetime')
    testdata = pd.read_csv('test.csv', parse_dates=True, index_col='datetime')
    st.success("Data loaded successfully!")
except FileNotFoundError:
    st.error("The data files were not found. Please upload them.")
    st.stop()
except pd.errors.EmptyDataError:
    st.error("The data files are empty. Please check the files.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.stop()

# Display shape of data
st.write('Shape of data: ', mydata.shape)

# Display first few rows of the data
if st.button('Show first 3 rows of training data', key='show_first_rows'):
    st.write(mydata.head(3))

# Display info of the data
if st.button('Show info of training data', key='show_info'):
    st.write("Data Types and Non-Null Counts:")
    st.write(mydata.dtypes)  # Display the data types of each column

# Display description of the data
if st.button('Show description of training data', key='show_description'):
    st.write(mydata.describe())

# Check if Casual + Registered equals Count
if st.button('Check Casual + Registered equals Count', key='check_casual_registered'):
    st.write('Casual + Registered = Count? ', ~(mydata.casual + mydata.registered - mydata['count']).any())

# Convert categorical data
category_list = ['season', 'holiday', 'workingday', 'weather']
for var in category_list:
    mydata[var] = mydata[var].astype('category')
    testdata[var] = testdata[var].astype('category')

# Mapping numbers to understandable text
season_dict = {1:'Spring', 2:'Summer', 3:'Fall', 4:'Winter'}
weather_dict = {1:'Clear', 2:'Misty+Cloudy', 3:'Light Snow/Rain', 4:'Heavy Snow/Rain'}
mydata['season'] = mydata['season'].map(season_dict)
mydata['weather'] = mydata['weather'].map(weather_dict)
testdata['season'] = testdata['season'].map(season_dict)
testdata['weather'] = testdata['weather'].map(weather_dict)

# Display average values across each of the categorical columns
if st.button('Show average bike rentals across Weather, Season, Working Day, Holiday', key='show_average_rentals'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    group_weather = pd.DataFrame(mydata.groupby(['weather'])['count'].mean()).reset_index()
    sns.barplot(data=group_weather, x='weather', y='count', ax=axes[0][0])
    axes[0][0].set(xlabel='Weather', ylabel='Count', title='Average bike rentals across Weather')

    group_season = pd.DataFrame(mydata.groupby(['season'])['count'].mean()).reset_index()
    sns.barplot(data=group_season, x='season', y='count', ax=axes[0][1])
    axes[0][1].set(xlabel='Season', ylabel='Count', title='Average bike rentals across Seasons')

    group_workingday = pd.DataFrame(mydata.groupby(['workingday'])['count'].mean()).reset_index()
    sns.barplot(data=group_workingday, x='workingday', y='count', ax=axes[1][0])
    axes[1][0].set(xlabel='Working Day', ylabel='Count', title='Average bike rentals across Working Day')

    group_holiday = pd.DataFrame(mydata.groupby(['holiday'])['count'].mean()).reset_index()
    sns.barplot(data=group_holiday, x='holiday', y='count', ax=axes[1][1])
    axes[1][1].set(xlabel='Holiday', ylabel='Count', title='Average bike rentals across Holiday')

    st.pyplot(fig)  # Pass the figure object to st.pyplot()

# Seaborn boxplots to get an idea of the distribution/outliers
if st.button('Show boxplots for distribution/outliers', key='show_boxplots'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    hue_order= ['Clear', 'Heavy Snow/Rain', 'Light Snow/Rain', 'Misty+Cloudy']
    sns.boxplot(data=mydata, y='count', x='weather', ax=axes[0][0], order=hue_order)
    sns.boxplot(data=mydata, y='count', x='workingday', ax=axes[0][1])
    hue_order= ['Fall', 'Spring', 'Summer', 'Winter']
    sns.boxplot(data=mydata, y='count', x='season', ax=axes[1][0], order=hue_order)
    sns.boxplot(data=mydata, y='count', x='holiday', ax=axes[1][1])
    st.pyplot(fig)  # Pass the figure object to st.pyplot()

# Splitting datetime object into month, date, hour and day category columns
mydata['month'] = mydata.index.month
mydata['date'] = mydata.index.day
mydata['hour'] = mydata.index.hour
mydata['day'] = mydata.index.weekday

testdata['month'] = testdata.index.month
testdata['date'] = testdata.index.day
testdata['hour'] = testdata.index.hour
testdata['day'] = testdata.index.weekday

category_list = ['month', 'date', 'hour', 'day']
for var in category_list:
    mydata[var] = mydata[var].astype('category')
    testdata[var] = testdata[var].astype('category')

# Mapping 0 to 6 day indices to Monday to Saturday 
day_dict = {0:'Monday', 1:'Tuesday', 2:'Wednesday', 3:'Thursday', 4:'Friday', 5:'Saturday', 6:'Sunday'}
mydata['day'] = mydata['day'].map(day_dict)
testdata['day'] = testdata['day'].map(day_dict)

# Display seaborn boxplots across hours
if st.button('Show hourly count based on working day or not', key='show_hourly_count'):
    f, axes = plt.subplots(1, 1, figsize=(15, 6))
    sns.boxplot(data=mydata, y='count', x='hour', hue='workingday', ax=axes)
    handles, _ = axes.get_legend_handles_labels()
    axes.legend(handles, ['Not a Working Day', 'Working Day'])
    axes.set(title='Hourly Count based on Working day or not')
    st.pyplot(f)

# Plots of average count across hour in a day for various categories
if st.button('Show average bike rentals by hour', key='show_average_by_hour'):
    f, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 18))
    
    # Average bike rentals by hour for working and non-working days
    group_work_hour = pd.DataFrame(mydata.groupby(['workingday', 'hour'])['count'].mean()).reset_index()
    sns.pointplot(data=group_work_hour, x='hour', y='count', hue='workingday', ax=axes[0])
    axes[0].set(xlabel='Hour in the day', ylabel='Count', title='Average Bike Rentals by the day if Working day or Not')

    # Average bike rentals by hour across weekdays
    hue_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    group_day_hour = pd.DataFrame(mydata.groupby(['day', 'hour'])['count'].mean()).reset_index()
    sns.pointplot(data=group_day_hour, x='hour', y='count', hue='day', ax=axes[1], hue_order=hue_order)
    axes[1].set(xlabel='Hour in the day', ylabel='Count', title='Average Bike Rentals by the day across Weekdays')

    # Average bike rentals by hour across Casual/Registered Users
    df_melt = pd.melt(frame=mydata, id_vars='hour', value_vars=['casual', 'registered'], value_name='Count', var_name='casual_or_registered')
    group_casual_hour = pd.DataFrame(df_melt.groupby(['hour', 'casual_or_registered'])['Count'].mean()).reset_index()
    sns.pointplot(data=group_casual_hour, x='hour', y='Count', hue='casual_or_registered', ax=axes[2])
    axes[2].set(xlabel='Hour in the day', ylabel='Count', title='Average Bike Rentals by the day across Casual/Registered Users')

    st.pyplot(f)

# Average Monthly Count Distribution plot
if st.button('Show average bike rentals per Month', key='show_average_per_month'):
    f, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
    group_month = pd.DataFrame(mydata.groupby(['month', 'workingday'])['count'].mean()).reset_index()
    sns.barplot(data=group_month, x='month', y='count', hue='workingday', ax=axes)
    axes.set(xlabel='Month', ylabel='Count', title='Average bike rentals per Month')
    handles, _ = axes.get_legend_handles_labels()
    axes.legend(handles, ['Not a Working Day', 'Working Day'])
    st.pyplot(f)

# Display feature importance for Random Forest Regression
if st.button('Show feature importance for Random Forest Regression', key='show_feature_importance'):
    # Prepare data for Random Forest
    X = mydata.drop(columns=['count'])
    y = mydata['count']

    # One-hot encode categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    feature_importances = model.feature_importances_

    # Create a figure for the plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(feature_importances)), feature_importances)
    ax.set_title('Feature Importances for Random Forest Regression')
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('Importance')
    
    # Pass the figure to st.pyplot()
    st.pyplot(fig)

# Final submission
if st.button('Generate Submission', key='generate_submission'):
    # Assuming the Random Forest model has been trained and predictions are made
    # This part of the code would typically involve generating predictions and saving to a CSV
    submission = pd.DataFrame({'datetime': testdata.index, 'count': np.random.rand(len(testdata))})  # Dummy predictions
    submission.to_csv('bikeSharing_submission.csv', index=False)
    st.write("Submission file generated: bikeSharing_submission.csv")