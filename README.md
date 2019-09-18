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


Influence of parameters on availability:

- Cancellation policy
- Room type
- Number of guests
- Guests picture


## File Descriptions <a name="files"></a>

Vancouver folder contains five files

- calendar.csv: availability data for listings
- listings.csv: information about listings
- reviews.csv: reviews by users
- neighbourhoods.csv  (not used)
- neighbourhoods.geojson (not used)

## Summary Of Conclusions<a name="results"></a>

Key findings from the analysis are summarized below:

1. It was found that approximately 70% of the hosts in metro vancouver respond with in an hour. 
2. Approximately 35% of the properties listed on Vancouver Air bnb are houses and 26% are apartments.
3. Listing price is higher for the number of reviews between 200-300, as the number of reviews increases, price drops, which suggests that more customer tend to review places which are economical, however places with 200-300 tend to have higher listing price. 

4. Downtown, Kistilano are the most expensive neighbourhoods, whereas Killarnay is the cheapest.
5. We didnt see any storng relationship with the occupancy rate and the review scores.
6. We also found that Kistilano and Downtown east neighbourhoods receivee the most positive reviews. 
7. We also found that there is a storng relationship between property type, room type, number of guests with the occupancy rate.


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
