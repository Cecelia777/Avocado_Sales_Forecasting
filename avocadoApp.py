#%%
from xml.etree.ElementInclude import include
from holidays import IM
from matplotlib import units
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
import streamlit as st
import datetime
import base64
from PIL import Image
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
from datetime import datetime
from sklearn.metrics import mean_absolute_error
# %%
# load datasets
avocado19 = pd.read_csv('Data/2019-plu-total-hab-data.csv')
avocado20 = pd.read_csv('Data/2020-plu-total-hab-data.csv')
avocado21 = pd.read_csv('Data/2021-plu-total-hab-data.csv')
avocado22 = pd.read_csv('Data/2022-plu-total-hab-data.csv')


#%%
@st.cache
def load_data():
    data = pd.concat([avocado19, avocado20, avocado21, avocado22], axis = 0)
    data['Current Year Week Ending'] = pd.to_datetime(data['Current Year Week Ending'])
    data.rename(columns = {'Current Year Week Ending': 'Date', 'ASP Current Year': 'AvgSalePrice', 'Total Bulk and Bags Units': 'TotalSaleUnits',
                '4046 Units': '4046', '4225 Units': '4225', '4770 Units': '4770'}, inplace = True)
    return data



#%%
image = Image.open('logo1.jpg')

st.image(image, width = 500)

#%%
st.title('Hass Avocado Explorer And Sales Forecasting')

st.markdown("""
This App performs data of hass avocado from different areas in the United States; 
The App also use Facebook Prophet forecasting to predict for future Total Avocado Sale Units for curtain type and geography area.
The App also explore different funtionality of Facebook Prophet
""")

#%%
expander_bar = st.expander("About")
expander_bar.markdown("""
* **Python Libraries:** streamlit, Prophet, pandas, base64, numpy, seaborn, matplotlib, sklearn
* **Data Source:** [https://hassavocadoboard.com](https://hassavocadoboard.com/)
""")

#%%
st.sidebar.header('User Input Features')


# Load the data
avocado = load_data()
avocado.fillna(0)


# %%
# sidebar -- Geography
sorted_area = sorted(avocado.Geography.unique())
selected_area = st.sidebar.selectbox('Geography', sorted_area)

# sidebar -- Type
unique_type = ['Conventional', 'Organic']
selected_type = st.sidebar.selectbox('Type', unique_type)




# %%
# Filtering the data
df_selected_areas = avocado[(avocado.Geography == selected_area) & (avocado.Type == selected_type)]



st.header('Display Hass Avocado Data')
st.write('Data Dimension: ' + str(df_selected_areas.shape[0]) + ' rows and ' + str(df_selected_areas.shape[1]) + ' columns')
# st.dataframe(df_selected_areas)   # this line of code is displaying the dataframe on streamlit
st.write(df_selected_areas) #Try another method to display the dataframe





# %%
# Option to download the Avocado data based on selections
def filedownload(df):
    csv = df.to_csv(index = False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="">Download CSV File</a>'
    return href

st.markdown(filedownload(df_selected_areas), unsafe_allow_html=True)

#%%
# Creating a heatmap(hidden)
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_numeric = df_selected_areas.select_dtypes(include = np.number)
    df_numeric.to_csv('output.csv', index = False)
    df = pd.read_csv('output.csv')

    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True

    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7,5))
        ax = sns.heatmap(corr, mask = mask, vmax = 1, square = True)
    st.pyplot(f)



#%%
# Another way to plot the graphs
df_selected_areas = df_selected_areas.set_index('Date').sort_index()
def plot_TotalSaleUnits_data():
    fig = go.Figure() 
    fig.add_trace(go.Scatter(x = df_selected_areas.index, y = df_selected_areas['TotalSaleUnits'], name = 'Avocado Total Sale Units Overtime in Chosen Geography and Type'))
    fig.layout.update(xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

st.subheader("Time Series Plot -- Total Sale Units")
plot_TotalSaleUnits_data()

# st.subheader("Time Series")
def plot_AvgSalePrice_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = df_selected_areas.index, y = df_selected_areas['AvgSalePrice'], name = 'Avgerage Sale Price Overtime in Chosen Geography and Type')) 
    fig.layout.update(xaxis_rangeslider_visible = True)
    st.plotly_chart(fig)

st.subheader("TIme Series Plot -- Average Sale Price")
plot_AvgSalePrice_data()




#%%
#######################################################
#        Set Up Data for Prophet Forecasting          #
#######################################################
saleUnits = avocado[(avocado.Geography == selected_area) & (avocado.Type == selected_type)][['Date', 'TotalSaleUnits']].rename(columns={'Date': 'ds',
                                                                                           'TotalSaleUnits': 'y'})


### Prophet Model
# We can specify the desired range of our uncertainty interval by setting the 
# interval_width parameter
# I set uncertainty interbal to 95% (the Prophet default is 80%)
my_model = Prophet(interval_width=0.95)

# Fit the model
my_model.fit(saleUnits)


# In order to obtain forecasts of our time series, we must provide Prophet with a new DataFrame containing a ds column that holds the dates for which we want predictions.
# We instructed Prophet to generate 90 datestamps in the future

# siderbar -- Prediction Horizon
n_seasons = st.slider("Number of Seasons Prediction: ", 1, 8)
period = n_seasons * 90
future_dates = my_model.make_future_dataframe(periods=period)




# %%
# The dataframe of future dates is then used as input to the predict method of fitted model
# Prophet returns a large dataframe with many interesting columns
# ds: The datestampp of the forecasted value
# yhat: the forecasted value of our metric (in Statistics, yhat is a notation traditionally used to represent the predicted value a value y)
# yhat_lower: The lower bound of our forecasts
# yhat_upperL The upper bound of our forecasts


forecast = my_model.predict(future_dates)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head()

st.subheader('Forecast Data For Total Sales Units')
st.write(forecast.tail())



#%%
# Prophet also provides a convenient function to quickly plot the results of our forecasts as follows:
# We can see that prophet plots the observed values of our time series(the black dots), the forecasted values(blue line)
# and the uncertainty intervals of our forecasts(the blue shaded regions)
st.subheader('Graph For Total Sales Units Forecast')
fig1 = plot_plotly(my_model, forecast)
st.plotly_chart(fig1)





# %%
# Prophet also can return the components of our forecasts
# The first plot shows that the monthly total sale units of Atlanta conventional avocado
# The second plot highlights the fact that the weekly conventional avocado sold units peaks during the weekdays and Saturday.
# The third plot shows that the most conventional avocado sold peaks in Feburury 
st.subheader("Forecast Components")
fig2 = my_model.plot_components(forecast)
st.write(fig2)


# %%
# Diagnostics- Cross valudation 
# Prophet includes functionality for time series cross validation to meansure forecast error using historical data
# This is done by selecting cutoff points in the history, and for each of them fitting the model using data only up to that cutoff point.

from prophet.diagnostics import cross_validation
saleUnits_cv = cross_validation(my_model, 
                                   initial = '120 days',
                                   period = '15 days',
                                   horizon = '30 days')

# %%
# Write in the cross validation table
st.subheader("Cross validation Metric Table")
st.write(saleUnits_cv.tail())


# %%
# The performance_metrics utility can be used to compute some useful statistics of the prediction performance as a function of the distance 
# from the cutoff. 
from prophet.diagnostics import performance_metrics
df_p = performance_metrics(saleUnits_cv)
# st.subheader("Performance Matrix")
# st.write(df_p.head())

# Cross validation performance metrics can be visualized with plot_cross_validation_metrix, here shown for MAE.
# Dots show the Mean Absolute Error for each prediction in saleUnits_cv. 
# The blue line shows the MAE, where the mean is taken over a rolling window of the dots. 
from prophet.plot import plot_cross_validation_metric
fig3 = plot_cross_validation_metric(saleUnits_cv, metric='mae')
st.subheader("Cross Validation Metric Plot -- MAE")
st.write(fig3)





# %%
# Adding changePoints to Prophet

# Changepoints are the datetime points where the time series have abrupt changes in the trajectory
# By default, prophet adds 25 changespoints to the initial 80% of the dataset

from prophet.plot import add_changepoints_to_plot
fig4 = my_model.plot(forecast)
a = add_changepoints_to_plot(fig4.gca(), my_model, forecast)

st.write(fig4)

# We can view the dates where the changepoints occurred
# my_model.changepoints  # By default, Prophet specifies 25 potential changepoints which are uniformly placed 
                       # in the first 80% of the time series. 
                       # The vertical lines in the figure indicate where the potential changepoints were placed





#%%
# Adjusting Trend 
# Prophet allows us to adjust the trend in case there is an overfit or underfit
# Being overfit(too much flecibility), being underfit(not enough flexibility)
# changepoint_prior_scale helps adjust the strength of the trend
# Default value for changepoint_prior_scale is 0.05
# Decrease the value to make the trend less flexible
# Increase the value of changepoint_prior)scake to make the trend more flexible
# n_changepoints = 20, changepoint_prior_sclae = 0.08
pro_change= Prophet(n_changepoints=30, yearly_seasonality=True, changepoint_prior_scale=0.85)
forecast2 = pro_change.fit(saleUnits).predict(future_dates)
fig5= pro_change.plot(forecast2);
a2 = add_changepoints_to_plot(fig5.gca(), pro_change, forecast2)

st.write(fig5)

# %%
# Specifying the location of the changepoints
loc_change = Prophet(changepoints=['2020-01-06', '2020-08-23', '2021-01-04'])
forecast3 = loc_change.fit(saleUnits).predict(future_dates)
fig6 = loc_change.plot(forecast3)
a3 = add_changepoints_to_plot(fig6.gca(),loc_change, forecast3)

st.write(fig6)

#%%
# Uncertainty in seasonality
# By dedault Prophet will only return uncertainty in the trend and observation noise. To get uncertainy in seasonality, you must
# do full Bayesian sample. This is done using the parameter mcmc.samples(which defaults to 0).
m = Prophet(mcmc_samples=184)
forecast = m.fit(saleUnits, show_progress=False).predict(future_dates)
fig6 = m.plot_components(forecast)
st.write(fig6)



# %%
# Split the data

split_date = '2021-08-29'
split_date = datetime.strptime(split_date, '%Y-%m-%d')

# Plot tran and test

saleUnits_train = saleUnits.loc[saleUnits.ds <= split_date].copy()
saleUnits_test = saleUnits.loc[saleUnits.ds > split_date].copy()

# saleUnits_train = saleUnits_train.set_index('ds')
# saleUnits_test = saleUnits_test.set_index('ds')

# saleUnits_train\
#     .rename(columns = {'y': 'Train Set'}) \
#     .join(saleUnits_test.rename(columns = {'y': 'Test Set'}), how = 'outer').plot(figsize = (15,5), stype = '.')
# plt.show()


model = Prophet()
model.fit(saleUnits_train)

saleUnits_test_forecast = model.predict(saleUnits_test)

st.subheader("Predicted with test set")
st.write(saleUnits_test_forecast.head())


#%%
# Plot the forecast
f, ax = plt.subplots()
f.set_figheight(5)
f.set_figwidth(15)
fig7 = model.plot(saleUnits_test_forecast,
                 ax=ax)
st.subheader("Plot with test set forecast")
st.write(fig7)

# %%
# Compare Forecast to Actuals
f1, ax1 = plt.subplots(1)
f1.set_figheight(5)
f1.set_figwidth(15)
ax1.scatter(saleUnits_test['ds'], saleUnits_test['y'], color='r')
fig7 = model.plot(saleUnits_test_forecast, ax=ax1)
st.write(fig7)

#%%
st.write("Mean Absolute Error without Regressor: ")
st.write(mean_absolute_error(y_true=saleUnits_test['y'],
                   y_pred=saleUnits_test_forecast['yhat']))

#%%
# Additional regressors
# Additional regressors can be added to the linear part of the model using the add_regressor method or function. 
# A column with the regressor value will need to be present in both the fitting and prediction dataframes. 

# Cite: Regressor value must be known in the past and in the future, this is how it helps Prophet to adjust the forecast.
# The future value must be either predefined and known
# For example. a specific event happening in certain dates, or it should be forecasted elsewhere. 


### salePrice = avocado[(avocado.Geography == selected_area) & (avocado.Type == selected_type)][['Date', 'AvgSalePrice']]
saleUnits_withPrice = avocado[(avocado.Geography == selected_area) & (avocado.Type == selected_type)][['Date', 'TotalSaleUnits', 'AvgSalePrice']].rename(columns={'Date': 'ds',
                                                                                           'TotalSaleUnits': 'y'})

# Split the data

split_date = '2021-08-29'
split_date = datetime.strptime(split_date, '%Y-%m-%d')

saleUnits_withPrice_train = saleUnits_withPrice.loc[saleUnits_withPrice.ds <= split_date].copy()
saleUnits_withPrice_test = saleUnits_withPrice.loc[saleUnits_withPrice.ds > split_date].copy()

model_with_regressor = Prophet()
model_with_regressor.add_regressor('AvgSalePrice')
model_with_regressor.fit(saleUnits_withPrice_train)
saleUnits_withPrice_test_forecast = model_with_regressor.predict(saleUnits_withPrice_test)

st.subheader("Predicted with AvgSalePrice as regressor for test set")
st.write(saleUnits_withPrice_test_forecast.head())

#%%
# Compare Forecast to Actuals
f2, ax2 = plt.subplots(1)
f2.set_figheight(5)
f2.set_figwidth(15)
ax2.scatter(saleUnits_withPrice_test['ds'], saleUnits_withPrice_test['y'], color='r')
fig8 = model_with_regressor.plot(saleUnits_withPrice_test_forecast, ax=ax2)
st.write(fig8)

#%%
st.write("Mean Absolute Error with Regressor: ")
st.write(mean_absolute_error(y_true=saleUnits_withPrice_test['y'],
                   y_pred=saleUnits_withPrice_test_forecast['yhat']))

# %%
