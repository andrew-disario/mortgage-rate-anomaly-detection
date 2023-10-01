<h1>Mortgage Rate Anomaly Detection</h1>

<p align="center">
<img src="https://github.com/andrew-disario/mortgage-rate-anomaly-detection/blob/main/30-year-fixed-rate-mortgage-average-in-the-united-states-last-20-years-anomaly-detection.png?raw=true" height="80%" width="80%" alt="30-year-fixed-rate-mortgage-average-in-the-united-states-last-20-years-anomaly-detection"/>
</br>
Use data observability tools that apply machine learning to identify unexpected changes in 30-year fixed-rate mortgages. 

<h2>Part I - Gather Data</h2>

<b>Set Up Notebook and Connect to FRED</b>

1. Import libraries
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date
from datetime import timedelta
import plotly.express as px
import matplotlib as mpl
from fbprophet import Prophet
from dateutil.relativedelta import relativedelta
from fredapi import Fred
```
2. Set up Fred object with FRED API key.
```
fred_key = 'b2e2ce0707e5200f8e4ed4e725d5a257'
fred = Fred(api_key=fred_key)
```

<>bSearch FRED for 30-Year Mortgage Data</b>

1. Search FRED for "mortgage rate" data and output results to dataframe.
```
fred_df = fred.search('mortgage rate')
```
2. Output titles to list
```
fred_title_list = fred_df['title'].to_list()
```
3. Output Series ID of 30-year fixed rate US mortgage averages.
```
fred_df[fred_df['title'] == '30-Year Fixed Rate Mortgage Average in the United States'].index
```
4. Get 30-year mortgage rate data from Fred using get_series and the Series ID.
```
fred_mortgage_series = fred.get_series(series_id='MORTGAGE30US')
```
5. Plot 30-year mortgage data.
```
fred_mortgage_series.plot(
    figsize=(10, 5), 
    title='30-Year Fixed Rate Mortgage Average in the United States 1971-Present', 
    lw=2
)
```
<p align="center">
<img src="https://github.com/andrew-disario/mortgage-rate-anomaly-detection/blob/main/30-year-fixed-rate-mortgage-average-in-the-united-states-1971-present.png?raw=true" height="70%" width="70%" alt="30-year-fixed-rate-mortgage-average-in-the-united-states-1971-present"/>
</br>

<h2>Part II - Clean Data</h2>


<b>Convert to Dataframe and Filter to Last 20 Years</b>

1. Set cutoff date to 20 years ago
```
cutoff_date = date.today()-relativedelta(years=20)
cutoff_date = cutoff_date.strftime("%Y-%m-%d")
```
2. Convert series to dataframe, rename columns and use cutoff date to filter data
```
fred_mortgage_df = fred_mortgage_series.to_frame().reset_index()
fred_mortgage_df = fred_mortgage_df.rename(columns={'index': 'date', 0: 'mortgage_rate'})
fred_mortgage_df = fred_mortgage_df[fred_mortgage_df['date'] >= (cutoff_date)]
```

<b>Clean Data</b>

1. Set variable column names.
```
column_1 = 'date'
column_2 = 'mortgage_rate'
```
2. Set timeseries unit.
```
increment = 'D'
```
3. Set train-test split date.
```
train_test_split_date = '2020-01-01'
```
4. Set changepoint.
```
changepoint = 0.95
```
5. Set figure and axes parameters.
```
mpl.rcParams['figure.figsize'] = (10, 8)
mpl.rcParams['axes.grid'] = False
```
6. Copy dataframe to new dataframe.
```
df = fred_mortgage_df
```
7. Convert Column 1 to datetime.
```
df[column_1] = pd.to_datetime(df[column_1])
```
8. Plot data.
```
fig = px.line(
    df.reset_index(), 
    x=column_1, 
    y=column_2, 
    title='30-Year Fixed Rate Mortgage Average in the United States (Last 20 Years)',
    labels={
        column_1: "Date",
        column_2: "Mortgage Rate"
    },
)
```
9. Set up plot view-slider
```
fig.update_xaxes(
    rangeslider_visible = True,
    rangeselector = dict(
        buttons = list([
              dict(count=1, label='1y', step="year", stepmode="backward"),
              dict(count=2, label='2y', step="year", stepmode="backward"),
              dict(count=2, label='5y', step="year", stepmode="backward")
        ])
    )
)

fig.show()
```
<p align="center">
<img src="https://github.com/andrew-disario/mortgage-rate-anomaly-detection/blob/main/30-year-fixed-rate-mortgage-average-in-the-united-states-last-20-years.png?raw=true" height="70%" width="70%" alt="30-year-fixed-rate-mortgage-average-in-the-united-states-last-20-years"/>
</br>

<h2>Part III - Run Anomaly Detection</h2>


<b>Set Up Training and Testing Data</b>

1. Copy dataframe to new dataframe with renamed columns.
```
df2 = df.reset_index()[[column_1, column_2]].rename({column_1: 'ds', column_2: 'y'}, axis='columns')
```
2. Create training data dataframe and testing data dataframe
```
train = df2[(df2['ds'] <= train_test_split_date)]
test = df2[(df2['ds'] > train_test_split_date)]
```

<b>Set Up Model</b>

1. Create Prophet model and set changepoint range
```
m = Prophet(changepoint_range=changepoint)
```
2. Fit model to training data dataframe.
```
m.fit(train)
```
3. Set periods equal to difference between testing date dateframe start date and end date.
```
f_date = test['ds'].iloc[0]
l_date = test['ds'].iloc[-1]
delta = l_date - f_date
periods = delta.days
```
4. Generate 'future' dates for the model to use a number of periods beyond training data end date.
```
future = m.make_future_dataframe(periods=periods, freq=increment)
```

<b>Use Model to Generate Predictions</b>

1. Use model to generate predictions with lower and upper bounds for 'future' dates.
```
forecast = m.predict(future)
```
2. Concatenate dataframe with renamed columns and forecasted data.
```
result = pd.concat([df2.set_index('ds')['y'], forecast.set_index('ds')[['yhat','yhat_lower','yhat_upper']]], axis=1)
```
3. Plot forecasted data with trainining data and model projections.
```
fig1 = m.plot(forecast)
```
<p align="center">
<img src="https://github.com/andrew-disario/mortgage-rate-anomaly-detection/blob/main/forecast-plot.png?raw=true" height="70%" width="70%" alt="forecast-plot"/>
</br>

4. Plot weekly, yearly and overall trends.
```
comp = m.plot_components(forecast)
```
<p align="center">
<img src="https://github.com/andrew-disario/mortgage-rate-anomaly-detection/blob/main/trends-plot.png?raw=true" height="70%" width="70%" alt="trends-plot"/>
</br>

<b>Calculate Error and Uncertainty and Determine Anomalies</b>

1. Create columns with calculations for 'error' and 'uncertainty'.
```
result['error'] = result['y'] - result['yhat']
result['uncertainty'] = result['yhat_upper'] - result['yhat_lower']
```
2. Create column and definition for 'anomaly'.
```
result['anomaly'] = result.apply(lambda x: 'Yes' if(np.abs(x['error']) > 1.5*x['uncertainty']) else 'No', axis = 1)
```

<b>Plot Results</b>

1. Plot result data while highlighting anomalies.
```
fig = px.scatter(
    result.reset_index(), 
    x='ds', 
    y='y', 
    color='anomaly', 
    title='30-Year Fixed Rate Mortgage Average in the United States (Last 20 Years)')
```
2. Set up plot view-slider.
```
fig.update_xaxes(
    rangeslider_visible = True,
    rangeselector = dict(
        buttons = list([
              dict(count=1, label='1y', step="year", stepmode="backward"),
              dict(count=2, label='3y', step="year", stepmode="backward"),
              dict(count=2, label='5y', step="year", stepmode="backward"),
              dict(step="all")
        ])
    )
)
fig.show()
```
<p align="center">
<img src="https://github.com/andrew-disario/mortgage-rate-anomaly-detection/blob/main/30-year-fixed-rate-mortgage-average-in-the-united-states-last-20-years-anomaly-detection.png?raw=true" height="70%" width="70%" alt="30-year-fixed-rate-mortgage-average-in-the-united-states-last-20-years-anomaly-detection"/>
<br />
