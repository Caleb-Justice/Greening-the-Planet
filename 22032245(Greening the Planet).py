# -*- coding: utf-8 -*-
"""
Created on Thu May 11 12:52:14 2023

@author: Vite UH
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import scipy.optimize as opt
import errors as err


def read_data(a):
    """
    Reads and imports files from comma seperated values, to a python DataFrame

    Arguments:
    a: string, The name of the csv file which is to be read
    b: integer, indicates the number of rows on the csv file to be
    skipped

    Returns:
    data: A pandas dataframe with all values from the excel file
    transposed_data: The transposed pandas dataframe
    """
    data = pd.read_csv(a, skiprows=4)
    data = data.drop(['Country Code', 'Indicator Code'], axis=1)
    transposed_data = data.set_index(
        data['Country Name']).T.reset_index().rename(columns={'index': 'Year'})
    transposed_data = transposed_data.set_index('Year').dropna(axis=1)
    transposed_data = transposed_data.drop(['Country Name'])
    return data, transposed_data


co2_data = read_data("API_EN.ATM.CO2E.PC_DS2_en_csv_v2_5455265.csv")
co2_data = co2_data[0]
# print(co2_data.head())
rnew_ene = read_data("API_EG.FEC.RNEW.ZS_DS2_en_csv_v2_5457050.csv")
rnew_ene = rnew_ene[0]

# ------------------------------------------------------------------------
# Extracting the Desired Column
co2_emission = co2_data.iloc[:, [0, 59]]
rnew_energy = rnew_ene.iloc[:, [0, 59]]
co2_emission = co2_emission.round(3)

# Combine the dataframes using concat
desired_data = pd.concat([rnew_energy, co2_emission["2017"]], axis=1)

# List of countries/regions not needed
regions = ['Arab World',
           'Caribbean small states',
           'Central Europe and the Baltics',
           'Early-demographic dividend',
           'East Asia & Pacific (excluding high income)',
           'Euro area',
           'Europe & Central Asia (excluding high income)',
           'European Union',
           'Fragile and conflict affected situations',
           'Heavily indebted poor countries (HIPC)',
           'High income',
           'Latin America & Caribbean (excluding high income)',
           'Latin America & the Caribbean (IDA & IBRD countries)',
           'Least developed countries: UN classification',
           'Low & middle income',
           'Low income',
           'Lower middle income',
           'Middle East & North Africa (excluding high income)',
           'Middle income',
           'North America',
           'OECD members',
           'Other small states',
           'Pacific island small states',
           'Small states',
           'South Asia (IDA & IBRD)',
           'Sub-Saharan Africa (excluding high income)',
           'Sub-Saharan Africa (IDA & IBRD countries)',
           'Upper middle income',
           'World']

# Remove the countries from the DataFrame
desired_data = desired_data[~desired_data.index.isin(regions)]
print(desired_data)


def clean_data(desired_data, a, b):
    """
    This function takes a dataframe as input and performs several
    cleaning tasks on it:
    1. Sets the index to 'Country Name'
    2. Renames the columns to 'Renewable Energy' and 'CO2 Emission'
    3. Prints information about the dataframe
    4. Drops rows with missing values

    Parameters:
    desired_data (pandas dataframe): the dataframe to be cleaned

    Returns:
    pandas dataframe: the cleaned dataframe
    """
    # set index to 'Country Name'
    desired_data = desired_data.set_index('Country Name')

    # rename columns
    desired_data.columns.values[0] = a
    desired_data.columns.values[1] = b

    # print information about the dataframe
    # print(data.info())

    # drop rows with missing values
    desired_data = desired_data.dropna(axis=0)

    # return cleaned data
    return desired_data


main_df = clean_data(desired_data, "Renewable Energy", "CO2 Emission")
print(main_df)


# Define function to plot a scatter plot
def scatter_plot(data, x_col, y_col, title, xlabel, ylabel):
    """
    This function takes a dataframe and plots a scatter plot with the
    specified columns for x and y values, as well as a title, x-axis label,
    and y-axis label.

    Parameters:
    data (pandas dataframe): the dataframe to be plotted
    x_col (int or string):  index of x-axis column.
    y_col (int or string):  index of y-axis column.
    title (string): the title of the plot
    xlabel (string): the label for the x-axis
    ylabel (string): the label for the y-axis
    """
    # extract x and y values from the dataframe
    x = data.iloc[:, x_col]
    y = data.iloc[:, y_col]

    # create scatter plot
    plt.scatter(x, y)

    # add title, legend, x-axis label, and y-axis label
    plt.title(title, fontweight='bold')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # save plot figure
    plt.savefig('General Scatter Plot')

    # show plot
    plt.tight_layout()
    plt.show()


# Activate scatter plot function
scatter_plot(main_df, 0, 1, "Scatter Plot of Renewable Energy vs CO2 Emission",
             "Renewable Energy", "CO2 Emission")

print(main_df['CO2 Emission'].describe())

# Filter the dataframe to extract the desired countries
df1 = main_df[main_df.iloc[:, 1] <= 4.186]  # countries below world average
print(df1['CO2 Emission'].mean())
df2 = main_df[(main_df.iloc[:, 1] > 4.186) & (main_df.iloc[:, 1] < 5.75)]
print(df2)
df3 = main_df[main_df.iloc[:, 1] >= 30]
print(df3)

"""
Plot a scatter plot of countries with a CO2 emission
below the world average, i.e < 4.186
"""

# Activate scatter plot function
scatter_plot(df1, 0, 1,
             "Countries with CO2 Emission Below 4.186(World Average)",
             "Renewable Energy", "CO2 Emission")

# Normalise the data
scaler = preprocessing.MinMaxScaler()
df1_norm = scaler.fit_transform(df1)
# print(df1_norm)


# Define a function to determine the number of effective clusters for KMeans
def optimise_k_num(data, max_k):
    """
    A function to determine the optimal number of clusters for k-means
    clustering on a given dataset. The function plots the relationship between
    the number of clusters and the inertia, and displays the plot.

    Parameters:
    - data (array-like): the dataset to be used for clustering
    - max_k (int): the maximum number of clusters to test for

    Returns: None
    """

    means = []
    inertias = []

    for k in range(1, max_k):
        kmeans = KMeans(n_clusters=k, init='k-means++',
                        max_iter=1000)
        kmeans.fit(data)

        means.append(k)
        inertias.append(kmeans.inertia_)

    # Generate the elbow plot
    fig = plt.subplots(figsize=(8, 5))
    plt.plot(means, inertias, "r-")
    plt.xlabel("Number of Clusters", fontsize=12)
    plt.ylabel("Inertia", fontsize=12)
    plt.title("Elbow Method Showing Optimal Number of K", fontsize=16,
              fontweight='bold')
    plt.grid(True)
    plt.savefig("K-Means Elbow Plot.png")
    plt.show()

    return fig


# Activate the optimum k function to get the number of effective clusters
optimise_k_num(df1_norm, 10)

# Create a function to run the KMeans model on the dataset


def kmeans_model(data, n_clusters):
    """
    Applies K-Means clustering on the data and returns the cluster labels.

    Parameters:
        data (numpy array or pandas dataframe) : The data to be clustered
        n_clusters (int) : The number of clusters to form.

    Returns:
        numpy array : The cluster labels for each data point
        numpy array : The cluster centers
        float : The inertia of the model
    """
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=100,
                    random_state=0)
    kmeans.fit(data)
    clusters = kmeans.predict(data)
    centroids = kmeans.cluster_centers_
    inertia = kmeans.inertia_

    return clusters, centroids, inertia


# Activate KMeans clustering function
clusters, centroids, inertia = kmeans_model(df1, 4)
print("Clusters: ", clusters)
print("Centroids: ", centroids)
print("Inertia: ", inertia)

# Calculate the silhouette score for the number of clusters
sil_0 = silhouette_score(df1, clusters)
print("Silhouette Score:" + str(sil_0))

df1['Clusters'] = clusters
print(df1)


# Define a function that plots the clusters
def plot_clusters(df, cluster, centroids):
    """
    Plot the clusters formed from a clustering algorithm.

    Parameters:
    df: DataFrame containing the data that was clustered.
    cluster: Array or Series containing the cluster labels for each
    point in the data.
    centroids: Array or DataFrame containing the coordinates of the
    cluster centroids.
    """

    df.iloc[:, 1]
    df.iloc[:, 0]
    cent1 = centroids[:, 1]
    cent2 = centroids[:, 0]
    plt.scatter(df.iloc[cluster == 0, 1], df.iloc[cluster == 0, 0], s=50,
                c='orange', label='Cluster 0')
    plt.scatter(df.iloc[cluster == 1, 1], df.iloc[cluster == 1, 0], s=50,
                c='blue', label='Cluster 1')
    plt.scatter(df.iloc[cluster == 2, 1], df.iloc[cluster == 2, 0], s=50,
                c='green', label='Cluster 2')
    plt.scatter(df.iloc[cluster == 3, 1], df.iloc[cluster == 3, 0], s=50,
                c='purple', label='Cluster 3')
    # Centroid plot
    plt.scatter(cent1, cent2, c='red', s=100, label='Centroid')
    plt.title('Cluster of the Countries',
              fontweight='bold')
    plt.ylabel('Renewable Energy', fontsize=12)
    plt.xlabel('CO2 Emission', fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig("Clusters.png")
    plt.show()


# Activate the function to plot the clusters
plot_clusters(df1, clusters, centroids)

# Carry out cluster analysis by plotting a bar chart showing the country
# distribution in each cluster
sns.countplot(x='Clusters', data=df1)
plt.savefig('Cluster distribution.png')
plt.title('Cluster Distribution of Countries with CO2 Emission Below 4.186'
          ' (World Average)', fontweight='bold')
plt.show()


# Define a polynomial function to plot a curve fit curve
def fit_polynomial(x, a, b, c, d):
    """
    Fit a polynomial of degree 3 to a given set of data points.

    Parameters:
    x: x-coordinates of the data points.
    a,b,c,d: function coefficients.

    Returns: Optimal values for the coefficients of the polynomial.
    """
    # popt, pcov = curve_fit(fit_polynomial, x, y)
    return a*x**3 + b*x**2 + c*x + d


# Initialise variables
x_axis = df1.values[:, 1]
y_axis = df1.values[:, 0]

# Instantiate the curvefit function
popt, pcov = opt.curve_fit(fit_polynomial, x_axis, y_axis)
a, b, c, d = popt
print('y = %.5f * x^3 + %.5f * x^2 + %.5f * x + %.5f' % (a, b, c, d))
print(pcov)

# Generate the curvefit line variables
d_arr = df1.values[:, 1]  # convert data to an array
x_line = np.arange(min(d_arr), max(d_arr)+1, 1)  # a random range of points
y_line = fit_polynomial(x_line, a, b, c, d)  # generate y-axis variables
plt.scatter(x_axis, y_axis, label="Countries")  # scatterplot
# plot the curvefit line
plt.plot(x_line, y_line, '-', color='red', linewidth=2, label="Curvefit")
plt.title('Cluster of Countries showing Prediction Line (Curvefit)',
          fontweight='bold')
plt.ylabel('Renewable Energy (%)', fontsize=12)
plt.xlabel('CO2 Emission (kg/cap)', fontsize=12)
plt.legend(loc='lower right')
plt.annotate('y = 0.00671x + 58.308', (3000, 55), fontweight='bold')
plt.savefig("Scatterplot Prediction Line.png")
plt.show()

# Generate the confidence interval and error range
sigma = np.sqrt(np.diag(pcov))
low, up = err.err_ranges(d_arr, fit_polynomial, popt, sigma)
print(low, up)

ci = 1.95 * np.std(y_axis)/np.sqrt(len(x_axis))
lower = y_line - ci
upper = y_line + ci
print(f'Confidence Interval, ci = {ci}')

# Plot showing best fitting function and the error range
plt.scatter(x_axis, y_axis, label="Countries")
plt.plot(x_line, y_line, '-', color='black', linewidth=2,
         label="Curvefit")
plt.fill_between(x_line, lower, upper, alpha=0.3, color='purple',
                 label="Error range")
plt.title('Cluster Showing Prediction Line (Curvefit) & Error Range',
          fontweight='bold')
plt.ylabel('Renewable Energy (%)', fontsize=12)
plt.xlabel('CO2 Emission (kg/cap)', fontsize=12)
plt.annotate(f'C.I = {ci.round(3)}', (7800, 60), fontweight='bold')
plt.legend(loc='upper right')
plt.savefig("Scatterplot Prediction Line.png")
plt.show()
