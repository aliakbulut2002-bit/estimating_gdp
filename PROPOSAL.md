Accurate and timely information on a country’s Gross Domestic Product (GDP) is essential for 
understanding economic performance and designing effective public policies. GDP data guide
decisions on investment, infrastructure, and poverty reduction, and they are crucial for evaluating
whether development strategies are effective. However, in many underdeveloped or fragile states, 
such data are often incomplete, outdated, or unavailable. Without reliable economic indicators, 
face serious challenges in identifying priorities or measuring progress toward prosperity.

Eritrea represents one of the clearest examples of this data scarcity. On the World Bank’s 
database, GDP figures are available only up to 2011, and before 1993 there are no official records. 
The International Monetary Fund does not publish GDP data for Eritrea, leaving a gap of more than a 
decade with no accessible information. This lack of statistics makes it almost impossible to track 
the country’s economic trajectory or assess the impact of its policies.

This project aims to address this gap by estimating Eritrea’s GDP per capita for the period 
2012–2021 using machine learning and non-economic proxies. The approach relies on widely available 
data sources such as satellite-derived night-time light intensity from the VIIRS instrument, 
together with demographic and geographic covariates including population, land area, and the urban
population rate. These variables provide indirect yet informative signals about economic activity.

The analysis uses a balanced country–year panel of 115 countries between 2012 and 2021. Four 
regression algorithms are trained and compared: Linear Regression, k-Nearest Neighbors, Random 
Forests, and Gradient Boosted Trees. All models are evaluated out-of-sample using the Root Mean 
Squared Error (RMSE) and R² metrics. The main goal is not to recover exact GDP levels but to 
capture general macroeconomic trends, indicating whether Eritrea’s economy expanded, contracted, or 
stabilized over time. The project will be considered successful if the models show good predictive 
performance on held-out countries and if the estimated series for Eritrea follows a plausible 
trajectory consistent with other indirect indicators of development.
