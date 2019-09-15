# Vancouver Air bnb analysis Udacity project 1
 
# Analysis of Vancouver Air Bnb market


### Table of Contents

1. [Libraries used for the project](#libraries)
2. [Objective](#motivation)
3. [File Descriptions](#files)
4. [Summary Of Conclusions](#results)
5. [Acknowledgements](#acknowledgements)

## Libraries used for the project <a name="libraries"></a>

Following python libraries:

1. Collections
2. Matplotlib 
3. NLTK
4. NumPy
5. Pandas
6. Seaborn
7. Sklearn

I used the Anconda python distribution with python 3.0

## Objective<a name="motivation"></a>

We reviewed the Vancouver Air bnb data set to analyze the pricing trends, occupancy rate vs reviewes, and the sentiment analaysis. 

Pricing correlation:
* How does price correlates with seasons of year?
* How the type of property impacts the listing price in Vancouver??
* Dependance of listing price on the neighbourhoods in Vancouver?



Analysis of Reviews:
- Understanding how reviews impact the occupancy rate
- Get a correlation of the reviews with the neighbourhoods in vancouver
- Can we explore some of the worst reviews for additional insights?


Price Prediction
Can we predict he price of the listing based on property type, neighbourhood, and availabilty?


## File Descriptions <a name="files"></a>

Vancouver folder contains five files

- calendar.csv: availability data for listings
- listings.csv: information about listings
- reviews.csv: reviews by users
- neighbourhoods.csv  (not used)
- neighbourhoods.geojson (not used)

## Summary Of Conclusions<a name="results"></a>

The following key findings from the analysis are summarized below:

1. It was found that the peak season in Seattle is during the summer months from June to August, with the absolute peak being in July. 
2. The "Southeast Magnolia" neighborhood was the priciest neighborhood in Seattle, followed by Portage Bay. Rainier Beach was the cheapest.
3. Looking further at neighborhoods and property types, I found out that houses in Portage Bay are the most expensive followed by houses in West Queen Anne and Westlake. 
4. With the help of SentimentIntensityAnalyzer, I was able to map the reviews to their respective sentiments of positive, negative or neutral. I found out that 97.2% of reviews were mostly positive, with 1% negative reviews and 1.8% of reviews that were neutral.
5. By exploring review sentiments by neighborhoods, I found out that Roxhill, Cedar Park and Pinehurst were the neighborhoods with the most positive reviews, while University District, Holly Park and View Ridge ranked lower.
6. By exploring the worst reviews, I found out that SentimentIntensityAnalyzer associate non-English reviews with negative sentiments. 
7. Using LinearRegression, I was able to predict price based on a prepped and cleaned dataset, with an r2score of 0.62 on both training and test datasets.
8. It was found that the features that had the most impact on price were a combination of host details as well as descriptive information about the listing.

## Acknowledgements<a name="acknowledgements"></a>

- Credit to the AirBnB dataset published by AirBnB and Kaggle for hosting it, the dataset here: https://www.kaggle.com/airbnb/
- Remove the plot border: https://stackoverflow.com/questions/925024/how-can-i-remove-the-top-and-right-axis-in-matplotlib
- Annotations:  https://robertmitchellv.com/blog-bar-chart-annotations-pandas-mpl.html
- Gradient Color: https://www.pythonprogramming.in/bar-chart-with-different-color-of-bars.html
- Remove the $ symbol: https://stackoverflow.com/questions/22588316/pandas-applying-regex-to-replace-values
- Subplots: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.subplots.html
- SentimentIntensityAnalyzer:https://stackoverflow.com/questions/39462021/nltk-sentiment-vader-polarity-scorestext-not-working
- Dropping multiple columns: https://stackoverflow.com/questions/28538536/deleting-multiple-columns-based-on-column-names-in-pandas                                  -  https://stackoverflow.com/questions/17838752/how-to-delete-multiple-columns-in-one-pass
- Dtype:  https://stackoverflow.com/questions/21271581/selecting-pandas-columns-by-dtype
- Replacing True, False: https://stackoverflow.com/questions/23307301/replacing-column-values-in-a-pandas-dataframe
