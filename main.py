import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
from sklearn.linear_model import LinearRegression

pd.options.display.float_format = "{:,.2f}".format
register_matplotlib_converters()

data = pd.read_csv("cost_revenue_dirty.csv")
# print(data.shape)
# print(data.sample(10))

check_nan = data.isna().values.any()
# print(check_nan)

nan_rows = data[data.isna().any(axis=1)]
# print(len(nan_rows))
# print(f"Any NaN values among the data? {data.isna().values.any()}")
# print(f"Number of NaN values: {len(nan_rows)}")

check_dup = data.duplicated().values.any()
# print(check_dup)

duplicated_rows = data[data.duplicated()]
# print(len(duplicated_rows))
# print(f"Any duplicates? {data.duplicated().values.any()}")
# print(f"Number of duplicates: {len(duplicated_rows)}")

# data.info()

"""Convert the USD_Production_Budget, USD_Worldwide_Gross, and USD_Domestic_Gross columns to a numeric format by removing $ signs and ,."""
# solution #1
# data["USD_Production_Budget"] = data["USD_Production_Budget"].astype(str).str.replace("$", "", regex=False).str.replace(",", "").astype(int)
# data["USD_Worldwide_Gross"] = data["USD_Worldwide_Gross"].astype(str).str.replace("$", "", regex=False).str.replace(",", "").astype(int)
# data["USD_Domestic_Gross"] = data["USD_Domestic_Gross"].astype(str).str.replace("$", "", regex=False).str.replace(",", "").astype(int)
#
# print(data[["USD_Production_Budget", "USD_Worldwide_Gross", "USD_Domestic_Gross"]])

# solution #2
# for x in data.columns[-3:]:
#     data[x] = data[x].astype(str).str.replace(r"[$,]", "", regex=True).astype(int)
#
# print(data[["USD_Production_Budget", "USD_Worldwide_Gross", "USD_Domestic_Gross"]])

# solution #3
# char_to_replce = ["$", ","]
# col_to_clean = ["USD_Production_Budget", "USD_Worldwide_Gross", "USD_Domestic_Gross"]
#
# for x in col_to_clean:
#     for i in char_to_replce:
#         data[x] = data[x].astype(str).str.replace(i, "", regex=False)
#     data[x] = pd.to_numeric(data[x])
#
# print(data[["USD_Production_Budget", "USD_Worldwide_Gross", "USD_Domestic_Gross"]])

# solution #4
# char_to_replce = ["$", ","]
# col_to_clean = ["USD_Production_Budget", "USD_Worldwide_Gross", "USD_Domestic_Gross"]
#
# for x in col_to_clean:
#     for i in char_to_replce:
#         data[x] = data[x].astype(str).str.replace(i, "", regex=False)
#     data[x] = data[x].astype(int)
#
# print(data[["USD_Production_Budget", "USD_Worldwide_Gross", "USD_Domestic_Gross"]])

# solution #5
col_to_clean = ["USD_Production_Budget", "USD_Worldwide_Gross", "USD_Domestic_Gross"]

for x in col_to_clean:
    data[x] = data[x].astype(str).str.replace(r"[$,]", "", regex=True).astype(int)

# print(data[["USD_Production_Budget", "USD_Worldwide_Gross", "USD_Domestic_Gross"]])

"""Convert the Release_Date column to a Pandas Datetime type."""
data["Release_Date"] = pd.to_datetime(data["Release_Date"])
# print(data.head())
# data.info()

"""Descriptive Statistics"""
# print(data.describe())

"""How much revenue did the lowest and highest budget films make?"""
lowest_budget = data[data["USD_Production_Budget"] == 1100.00]
# print(lowest_budget)

highest_budget = data[data["USD_Production_Budget"] == 425000000.00]
# print(highest_budget)

"""How many films grossed $0 domestically (i.e., in the United States)? What were the highest budget films that grossed nothing?"""
zero_domestic = data[data["USD_Domestic_Gross"] == 0]
# print(f"Number of films that grossed $0 domestically {len(zero_domestic)}")
sort_zero_domestic = zero_domestic.sort_values("USD_Production_Budget", ascending=False)
# print(sort_zero_domestic)

"""How many films grossed $0 worldwide? What are the highest budget films that had no revenue internationally?"""
zero_worldwide = data[data["USD_Worldwide_Gross"] == 0]
# print(f"Number of films that grossed $0 worldwide {len(zero_worldwide)}")
sort_zero_worldwide = zero_worldwide.sort_values("USD_Production_Budget", ascending=False)
# print(sort_zero_worldwide)

"""Filtering on Multiple Conditions"""
bool_list1 = [True, True, False, False]
bool_list2 = [False, True, True, False]
bool_lists = np.array(bool_list1) & np.array(bool_list2)
# print(bool_lists)

"""Create a subset for international releases that had some worldwide gross revenue, but made zero revenue in the United States."""
international_releases_loc = data.loc[(data["USD_Domestic_Gross"] == 0) & (data["USD_Worldwide_Gross"] != 0)]
# print(f"Number of international releases: {len(international_releases_loc)}")
# print(international_releases_loc.head())

"""Use the .query() function to accomplish the same thing."""
international_releases_query = data.query("USD_Domestic_Gross == 0 and USD_Worldwide_Gross != 0")
# print(f"Number of international releases: {len(international_releases_loc)}")
# print(international_releases_loc.tail())

"""How many films are included in the dataset that have not yet had a chance to be screened in the box office?"""
# Date of Data Collection
scrape_date = pd.Timestamp("2018-5-1")

future_releases = data[data["Release_Date"] > scrape_date]
# print(f"Number of unreleased movies: {len(future_releases)}")
# print(future_releases)

"""Create another DataFrame called data_clean that does not include these films."""
data_clean = data.drop(future_releases.index)

"""What is the percentage of films where the production costs exceeded the worldwide gross revenue?"""
money_losing_loc = data_clean.loc[data_clean["USD_Production_Budget"] > data_clean["USD_Worldwide_Gross"]]
# print(len(money_losing_loc))
# print(money_losing_loc)

money_losing_query = data_clean.query("USD_Production_Budget > USD_Worldwide_Gross")
# print(len(money_losing_query))
# print(money_losing_query)

"""Seaborn for Data Viz: Bubble Charts"""
# Default Chart
# plt.figure(figsize=(8, 4), dpi=200)
#
# ax = sns.scatterplot(data=data_clean, x="USD_Production_Budget", y="USD_Worldwide_Gross")
#
# ax.set(ylim=(0, 3000000000),
#        xlim=(0, 450000000),
#        ylabel='Revenue in $ billions',
#        xlabel='Budget in $100 millions')
#
# plt.show()

# Customized Chart
# plt.figure(figsize=(8, 4), dpi=200)
#
# with sns.axes_style("darkgrid"):
#
#     ax = sns.scatterplot(data=data_clean,
#                          x="USD_Production_Budget",
#                          y="USD_Worldwide_Gross",
#                          hue='USD_Worldwide_Gross',
#                          size='USD_Worldwide_Gross')
#
#     ax.set(ylim=(0, 3000000000),
#            xlim=(0, 450000000),
#            ylabel='Revenue in $ billions',
#            xlabel='Budget in $100 millions')
#
# plt.show()

"""Plotting Movie Releases over Time"""
# plt.figure(figsize=(8, 4), dpi=200)
#
# with sns.axes_style("darkgrid"):
#
#     ax = sns.scatterplot(data=data_clean,
#                          x="Release_Date",
#                          y="USD_Production_Budget",
#                          hue='USD_Worldwide_Gross',
#                          size='USD_Worldwide_Gross')
#
#     ax.set(ylim=(0, 450000000),
#            xlim=(data_clean.Release_Date.min(), data_clean.Release_Date.max()),
#            ylabel='Budget in $100 millions',
#            xlabel='Year')
#
# plt.show()

"""Converting Years to Decades Trick"""
# Create a DatetimeIndex object from the Release_Date column.
dt_index = pd.DatetimeIndex(data_clean.Release_Date)

# Grab all the years from the DatetimeIndex object using the .year property.
years = dt_index.year
# print(type(years))

# Use floor division // to convert the year data to the decades of the films.
decades = years // 10 * 10
# print(type(decades))
# print(decades)

# Add the decades as a Decade column to the data_clean DataFrame.
data_clean["Decade"] = decades
# print(data_clean.head())

"""Separate the "old" (before 1969) and "New" (1970s onwards) Films"""
old_films = data_clean[data_clean["Decade"] < 1970]
# print(len(old_films))
sort_old_films = old_films.sort_values("USD_Production_Budget", ascending=False)
# print(sort_old_films.head())

new_films = data_clean[data_clean["Decade"] >= 1970]
# print(len(new_films))
sort_new_films = new_films.sort_values("USD_Production_Budget", ascending=False)
# print(sort_new_films.head())

"""Seaborn Regression Plots"""
# plt.figure(figsize=(8, 4), dpi=200)
#
# with sns.axes_style("whitegrid"):
#
#     sns.regplot(data=old_films,
#                 x="USD_Production_Budget",
#                 y="USD_Worldwide_Gross",
#                 scatter_kws={"alpha": 0.4},
#                 line_kws={"color": 'black'})
#
# plt.show()

"""Use Seaborn's .regplot() to show the scatter plot and linear regression line against the new_films. """
# plt.figure(figsize=(8, 4), dpi=200)
#
# with sns.axes_style("darkgrid"):
#
#     ax = sns.regplot(data=new_films,
#                      x="USD_Production_Budget",
#                      y="USD_Worldwide_Gross",
#                      color="#2f4b7c",
#                      scatter_kws={"alpha": 0.3},
#                      line_kws={"color": "#ff7c43"})
#
#     ax.set(ylim=(0, 3000000000),
#            xlim=(0, 450000000),
#            ylabel="Revenue in $ billions",
#            xlabel="Budget in $ millions")
#
# plt.show()

"""Run Your Own Regression with scikit-learn (𝑅𝐸𝑉𝐸̂𝑁𝑈𝐸 = 𝜃0 + 𝜃1𝐵𝑈𝐷𝐺𝐸𝑇)"""
regression = LinearRegression()

# Regression line structure:
# Revenue = Y-Intercept + Slope * BUDGET

# Explanatory Variable(s) or Feature(s)
X = pd.DataFrame(new_films, columns=['USD_Production_Budget'])

# Response Variable or Target
y = pd.DataFrame(new_films, columns=['USD_Worldwide_Gross'])

# Find the best-fit line
regression.fit(X, y)

# Theta zero (Y-Intercept)
# intercept = regression.intercept_
# print(f"Y-Intercept: {intercept}. If a movie budget is $0, the estimated movie revenue is -$8.65 million.")

# Theta one (Slope)
# slope = regression.coef_
# print(f"Slope: {slope}. For every extra $1 in the budget, movie revenue increases by $3.1.")

# R-squared
# r_squared = regression.score(X, y)
# print(f"R-squared: {r_squared}. Our model explains about 56% of the variance in movie revenue.")

""" Run a linear regression for the old_films. Calculate the intercept, slope and r-squared. How much of the variance in movie revenue does the linear model explain in this case?"""
# Explanatory Variable(s) or Feature(s)
X = old_films[["USD_Production_Budget"]]

# Response Variable or Target
y = old_films[["USD_Worldwide_Gross"]]

# Find the best-fit line
regression.fit(X, y)

# Theta zero (Y-Intercept)
intercept = regression.intercept_[0]
# print(f"The intercept is: {intercept}")

# Theta one (Slope)
slope = regression.coef_[0]
# print(f"The slope coefficient is: {slope}")

# R-squared
r_squared = regression.score(X, y)
# print(f"The r-squared is: {r_squared}")

"""How much global revenue does our model estimate for a film with a budget of $350 million?"""
# 𝑅𝐸𝑉𝐸̂𝑁𝑈𝐸 = 𝜃0 + 𝜃1𝐵𝑈𝐷𝐺𝐸𝑇
old_films_revenue = regression.intercept_[0] + regression.coef_[0, 0] * 350000000
print(old_films_revenue)

budget = 350000000
revenue_estimate = regression.intercept_[0] + regression.coef_[0, 0]*budget
revenue_estimate = round(revenue_estimate, -6)
print(f'The estimated revenue for a $350 film is around ${revenue_estimate:,.0f}.')
