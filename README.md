# Avocado_Sales_Forecasting

I love Avocado foe many reasons, it is an awesome nutrient source of vitamins, omega-3 fatty acids, antiocidant and anti-inflammatory compounds; it also has a unique texture, delicaous to the taste combined with anything, etc. You name it :)

However, the price of avocado got unbelievable high a couple months back to me; I did not consume avocado like I would like to; Until recent when I revisit Costco and found the price the avocado went back to "normal" that I can afford. 

I thought it would be interested to analyzing avocado data and its sales trend. I found a reliable data source from hassavocadoboard.com and script 4 years of data to begin my analysis.

This repository is to forecast time series avocado sales from different areas in United States using Facebook Prophet Library; as well as build an application to automate forecasting based on user's input geography area and avocado type. 

The application contains hass avocado data that you could download, time series plots for total sale units and average sale price; forecast data and it's graph; As well as the Prophet model performance metric table; and error matrix comparison for with and without added regressor.

To serve the application locally, use the command:
streamlit run avocadoApp.py

