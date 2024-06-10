# %% [markdown]
# # Introduction

# %% [markdown]
# One of the main aspects of a Data Scientist's work involves exploratory data analysis and data cleaning. To exercise this skill, I utilized the dataset provided by the Airbnb company, which contains information about rooms in the state of Rio de Janeiro.

# %%
import folium
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from IPython.display import display

# %%
ad = pd.read_excel('database\airbnb_database.xlsx')
ad.head()

# %% [markdown]
# # Data cleaning
# 
# 

# %% [markdown]
# Before carrying out an analysis, it is necessary to clean our database. For this process, I will remove unnecessary columns to reduce the size of the database.

# %%
ad.drop(columns=['name', 'host_name', 'id', 'host_id', 'neighbourhood_group', 'last_review', 'license'], inplace=True)
ad.describe()

# %% [markdown]
# ## Observations
# 
# 
# *  75% of the values in "minimum_nights" are below four nights, with a maximum value of 1125 nights.
# * The maximum amount someone can pay for an Airbnb is R$595,793.00 (which is highly discrepant from the Brazilian reality and our dataset).
# 

# %%
ad.minimum_nights.plot(kind='box', vert=False, figsize=(10,3))

# %% [markdown]
# By plotting a graph to check how the 'minimum_nights' information is distributed, it is possible to visualize a significant number of outliers. Because of that, I will remove situations where 'minimum_nights' is greater than 30 days (1 month) 

# %%
ad.drop(ad[ad.minimum_nights > 30].index, axis=0, inplace=True)

# %%
ad.price.plot(kind='box', vert=False, figsize=(10,3))

# %% [markdown]
# By plotting a graph to check how the 'price' information is distributed. Since there is a big quantity of outliers, I will remove all the advertisement which the price is above $3.000 reais.

# %%
ad.drop(ad[ad.price > 3000].index, axis=0, inplace=True)

# %% [markdown]
# ## Fetching empty values
# 
# 

# %%
print('Amount of empty information in each column:')
print(ad.isnull().sum())

# %% [markdown]
# Since there is only one column with missing data, we will remove it and work with other information.

# %%
ad.drop(columns=['reviews_per_month'], inplace=True)

# %% [markdown]
# # Exploratory Data Analysis

# %% [markdown]
# For an initial analysis, we will seek to understand how our data would be organized spatially (according to latitude and longitude)

# %%
ad.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)

# %% [markdown]
# By plotting a longitude x latitude graph with color variation according to rental price variation, we recieve:

# %%
ad.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, c=ad['price'], cmap=plt.get_cmap('jet'))

# %% [markdown]
# As a result, it can be seen that there is no variation in prices according to location, so it is possible to find cheap rentals very close to more expensive rentals.
# 
# Let's investigate further.

# %%
correlacao = ad.corr()
correlacao['price'].sort_values(ascending=False)

# %% [markdown]
# It can be seen in the table above that there is little correlation between the price variable and the other numerical variables.
# 
# Next, we will investigate how they can be related to categorical variables.

# %%
print(ad['room_type'].value_counts())

# %% [markdown]
# There is a wide range of rentals for entire houses/apartments in the city of Rio de Janeiro

# %%
print(ad.groupby('room_type')['price'].mean())

# %% [markdown]
# Therefore, when visiting Rio de Janeiro, if you want to rent an Airbnb you can expect shared rooms to be, on average, the cheapest as well as those with the most options.

# %% [markdown]
# To check which regions have the most expensive average price, let's create a simple bar chart

# %%
neighborhood_mean_value = ad.groupby('neighbourhood')['price'].mean().reset_index()
top_10_neighborhood = neighborhood_mean_value.nlargest(10, 'price')

# %%
#Seaborn Aesthetic Settings
sb.set(style="whitegrid")
pastel_palette = sb.color_palette("pastel")
sb.set_palette(pastel_palette)

#Plotting
sb.barplot(x='price', y='neighbourhood', data=top_10_neighborhood, orient='h')
plt.xlabel('Average Price')
plt.ylabel('Neighborhoods')
plt.title('Neighborhoods with the Highest Average Price in Rio de Janeiro.')
plt.show()

# %% [markdown]
# If you want to see the average price of all the neighborhoods in Rio de Janeiro, use the iterative map below

# %%
#Create a map centered on Rio de Janeiro.
rio_map = folium.Map(location=[-22.9068, -43.1729], zoom_start=12)

#Add markers for each neighborhood with average price information
for index, row in neighborhood_mean_value.iterrows():
    bairro = row['neighbourhood']
    preço_medio = row['price']

    #Check if the neighborhood is among the 10 with the highest prices
    if bairro in top_10_neighborhood['neighbourhood'].values:
        icone = folium.Icon(color='red', icon='info-sign')  #Red icon for top 10
    else:
        icone = folium.Icon(color='blue')  #Blue icon for others

    popup_text = f"Neighbourhood: {bairro}<br>Average price: R$ {preço_medio:.2f}"
    folium.Marker(location=(ad[ad['neighbourhood'] == bairro]['latitude'].iloc[0],
                  ad[ad['neighbourhood'] == bairro]['longitude'].iloc[0]),
                  popup=popup_text,
                  icon=icone).add_to(rio_map)

#Display the map on the screen
display(rio_map)

# %% [markdown]
# With this in mind, we will seek to understand the opinion of contracting customers and where they tend to stay most often and where the best-rated airbnbs are found.

# %%
mean_review_per_neighborhood = ad.groupby('neighbourhood')['availability_365'].mean()
top_5_neighborhood = mean_review_per_neighborhood.nsmallest(5)
print("Neighborhoods with lower average availability throughout the year:")
print('')
for neighborhood, mean in top_5_neighborhood.items():
    print(f"{neighborhood}: {mean:.0f} days available, on average")

# %% [markdown]
# # Conclusion

# %% [markdown]
# Based on the explored data, it is evident that there is no strict price stratification by region in the city of Rio de Janeiro. In other words, affordable and luxurious rooms or apartments can be found in various locations.
# 
# However, entire apartments tend to have the highest prices on average compared to other categories, and rentals typically last for an average of 5 days.
# 
# In addition to this, neighborhoods such as Acari, Magalhães Bastos, and Manguinhos emerge as the preferred choices among platform users. Interestingly, none of these neighborhoods were among the top 10 in terms of the highest rental prices.
# 
# Given the low correlation between all numerical variables, we chose not to employ machine learning techniques for predictive purposes.