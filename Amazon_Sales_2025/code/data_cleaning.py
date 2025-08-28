from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd


#import file into data Frame from csv file
df = pd.read_csv('../data/amazon_sales_data 2025.csv')
print('Size of data: ', df.size)
print('Name of columns: ', df.columns)
#How many missing values and duplicates data has
print('Duplicates count:' ,df.duplicated().value_counts())
print('Missing values count:\n',df.isnull().value_counts())
# data has 0 duplicated values and 0 missing values
print(df.dtypes)
# Price , Quantity and Total Sales are numerical values and help with general financial calculations
# Predictions and situation on market
# Than make some predictions for Total Sales of different categories and preparing for visualization with Tableau
# Total Sales average group by Categories
df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%y', errors = 'coerce')
df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
print(df['Date'])
df.to_csv('../data/date_for_Tableau.csv')
# change datatime
df['Month'] = pd.to_datetime(df['Date']).dt.month
# create list of predictions
predictions = []
# make predictions with Linear Regression ( not enough data )
for category in df['Category'].unique():
    category_df = df[df['Category'] == category].sort_values('Month')
    X = category_df['Month'].values.reshape(-1,1)
    y = category_df['Total Sales'].values
    # model creating
    model = LinearRegression()
    model.fit(X,y)
    next_month = np.array([[X.max()+1]])
    pred = model.predict(next_month)[0]
    predictions.append({'Category':category,'Predict_Total_Sales_next_month':pred})
pred_df = pd.DataFrame(predictions).sort_values(by='Predict_Total_Sales_next_month',ascending=False)
pred_df = pred_df.rename(columns={'Predict_Total_Sales_next_month':'Total Sales'})
pred_df['Month'] = 5
print(pred_df.head())
avg_sales = df.groupby(['Category','Month'])['Total Sales'].mean().reset_index()
print(avg_sales)
df_total = pd.concat([avg_sales, pred_df], ignore_index=True)
print(df_total)
df_total.to_csv('../data/prediction_sales.csv')