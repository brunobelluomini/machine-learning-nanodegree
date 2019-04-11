#!/usr/bin/env python
# coding: utf-8

# # Machine Learning Engineer Nanodegree
# ## Unsupervised Learning
# ## Project: Creating Customer Segments

# Welcome to the third project of the Machine Learning Engineer Nanodegree! In this notebook, some template code has already been provided for you, and it will be your job to implement the additional functionality necessary to successfully complete this project. Sections that begin with **'Implementation'** in the header indicate that the following block of code will require additional functionality which you must provide. Instructions will be provided for each section and the specifics of the implementation are marked in the code block with a `'TODO'` statement. Please be sure to read the instructions carefully!
# 
# In addition to implementing code, there will be questions that you must answer which relate to the project and your implementation. Each section where you will answer a question is preceded by a **'Question X'** header. Carefully read each question and provide thorough answers in the following text boxes that begin with **'Answer:'**. Your project submission will be evaluated based on your answers to each of the questions and the implementation you provide.  
# 
# >**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

# ## Getting Started
# 
# In this project, you will analyze a dataset containing data on various customers' annual spending amounts (reported in *monetary units*) of diverse product categories for internal structure. One goal of this project is to best describe the variation in the different types of customers that a wholesale distributor interacts with. Doing so would equip the distributor with insight into how to best structure their delivery service to meet the needs of each customer.
# 
# The dataset for this project can be found on the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Wholesale+customers). For the purposes of this project, the features `'Channel'` and `'Region'` will be excluded in the analysis — with focus instead on the six product categories recorded for customers.
# 
# Run the code block below to load the wholesale customers dataset, along with a few of the necessary Python libraries required for this project. You will know the dataset loaded successfully if the size of the dataset is reported.

# In[1]:


# Import libraries necessary for this project
import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames

# Import supplementary visualizations code visuals.py
import visuals as vs

# Pretty display for notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print("Wholesale customers dataset has {} samples with {} features each.".format(*data.shape))
except:
    print("Dataset could not be loaded. Is the dataset missing?")


# ## Data Exploration
# In this section, you will begin exploring the data through visualizations and code to understand how each feature is related to the others. You will observe a statistical description of the dataset, consider the relevance of each feature, and select a few sample data points from the dataset which you will track through the course of this project.
# 
# Run the code block below to observe a statistical description of the dataset. Note that the dataset is composed of six important product categories: **'Fresh'**, **'Milk'**, **'Grocery'**, **'Frozen'**, **'Detergents_Paper'**, and **'Delicatessen'**. Consider what each category represents in terms of products you could purchase.

# In[2]:


# Display a description of the dataset
display(data.describe())


# ### Implementation: Selecting Samples
# To get a better understanding of the customers and how their data will transform through the analysis, it would be best to select a few sample data points and explore them in more detail. In the code block below, add **three** indices of your choice to the `indices` list which will represent the customers to track. It is suggested to try different sets of samples until you obtain customers that vary significantly from one another.

# In[3]:


# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [10, 20, 42]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(
    data.loc[indices], 
    columns = data.keys(),
    index=indices
)
print("Chosen samples of wholesale customers dataset:")
display(samples)


# ### Question 1
# Consider the total purchase cost of each product category and the statistical description of the dataset above for your sample customers.  
# 
# * What kind of establishment (customer) could each of the three samples you've chosen represent?
# 
# **Hint:** Examples of establishments include places like markets, cafes, delis, wholesale retailers, among many others. Avoid using names for establishments, such as saying *"McDonalds"* when describing a sample customer as a restaurant. You can use the mean values for reference to compare your samples with. The mean values are as follows:
# 
# * Fresh: 12000.2977
# * Milk: 5796.2
# * Grocery: 7951.3
# * Frozen: 3071.932
# * Detergents_paper: 2881.4
# * Delicatessen: 1524.8
# 
# Knowing this, how do your samples compare? Does that help in driving your insight into what kind of establishments they might be? 
# 

# In[4]:


# Percentile of chosen samples
import seaborn as sns
import matplotlib.pyplot as plt

percentiles_data = 100 * data.rank(pct=True)
percentiles_samples = percentiles_data.iloc[indices]
plt.figure(figsize=(14, 4))

_ = sns.heatmap(percentiles_samples, annot=True, cmap='RdBu_r', center=0)


# As the distribution is highly right-skewed (many outliers with high value) it's better to analyze the sample above in terms of percentile instead of the mean value.

# **Answer:**
# 
# The customer #10 has low percentile value on `Fresh` category and high percentiles on the `Grocery` one. It can represent someone in a category such as `supermarket`.
# 
# The second customer, #20, spends more on the `Fresh` category (percentile 77 on this category) and less on `Grocery`. It might be an example of someone who likes to eat at restaurants or at fresh markets.
# 
# Finally, customer #42 is also a heavy spender on the `Grocery` and `Detergents_Paper` categories and it can also represents someone who goes often in a supermarket.

# ### Implementation: Feature Relevance
# One interesting thought to consider is if one (or more) of the six product categories is actually relevant for understanding customer purchasing. That is to say, is it possible to determine whether customers purchasing some amount of one category of products will necessarily purchase some proportional amount of another category of products? We can make this determination quite easily by training a supervised regression learner on a subset of the data with one feature removed, and then score how well that model can predict the removed feature.
# 
# In the code block below, you will need to implement the following:
#  - Assign `new_data` a copy of the data by removing a feature of your choice using the `DataFrame.drop` function.
#  - Use `sklearn.cross_validation.train_test_split` to split the dataset into training and testing sets.
#    - Use the removed feature as your target label. Set a `test_size` of `0.25` and set a `random_state`.
#  - Import a decision tree regressor, set a `random_state`, and fit the learner to the training data.
#  - Report the prediction score of the testing set using the regressor's `score` function.

# In[5]:


data.head(2)


# In[6]:


# TODO: Make a copy of the DataFrame, using the 'drop' function to drop the given feature
feature_to_predict = 'Delicatessen'
new_data = data.drop(columns=[feature_to_predict])

# TODO: Split the data into training and testing sets(0.25) using the given feature as the target
# Set a random state.
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    new_data, 
    data[feature_to_predict], 
    test_size=0.25, 
    random_state=42
)

# TODO: Create a decision tree regressor and fit it to the training set
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)

# TODO: Report the score of the prediction using the testing set
score = regressor.score(X_test, y_test)
print(score)


# ### Question 2
# 
# * Which feature did you attempt to predict? 
# * What was the reported prediction score? 
# * Is this feature necessary for identifying customers' spending habits?
# 
# **Hint:** The coefficient of determination, `R^2`, is scored between 0 and 1, with 1 being a perfect fit. A negative `R^2` implies the model fails to fit the data. If you get a low score for a particular feature, that lends us to believe that that feature point is hard to predict using the other features, thereby making it an important feature to consider when considering relevance.

# **Answer:**
# 
# I tried to predict the `Delicatessen` category and my $R^2$ score resulted in `-2.25`. Statistically it means that my regressor has performed worse than the Null hypothesis as its value is _negative_. 
# 
# We will repeat the experiment for the rest of features and check their resulting `R^2`:

# In[7]:


np.random.randint(low=0, high=50, size=10)


# In[8]:


r2_median_scores = []

features_to_predict = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicatessen']

for feature_to_predict in features_to_predict:
    r2_scores = []
    new_data = data.drop(columns=[feature_to_predict])
    
    # Calculating R² score for multiple samples with different random statess
    random_states = np.random.randint(0, 100, 100)
    for random_state in random_states:
        X_train, X_test, y_train, y_test = train_test_split(
            new_data, 
            data[feature_to_predict], 
            test_size=0.25, 
            random_state=random_state
        )

        regressor = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
        r2_scores.append(regressor.score(X_test, y_test))
    r2_median_scores.append(np.median(r2_scores))

pd.DataFrame(r2_median_scores, index=features_to_predict, columns=['R2 Score']).T


# According to the results, there are two features which we could be predicted (therefore the most likely to be removed): `Grocery` and `Detergents_Paper`, which have obtained a quite high `R^2` score of almost 0.70. 
# 
# All the other features either obtained a low or negative `R^2` value, thus they are not predictable with the other ones, and so cannot be removed.

# ### Visualize Feature Distributions
# To get a better understanding of the dataset, we can construct a scatter matrix of each of the six product features present in the data. If you found that the feature you attempted to predict above is relevant for identifying a specific customer, then the scatter matrix below may not show any correlation between that feature and the others. Conversely, if you believe that feature is not relevant for identifying a specific customer, the scatter matrix might show a correlation between that feature and another feature in the data. Run the code block below to produce a scatter matrix.

# In[9]:


# Produce a scatter matrix for each pair of features in the data
# pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde'); Deprecated, so I will use `pandas.plotting.scatter_matrix` instead

pd.plotting.scatter_matrix(data, alpha = 1.0, figsize = (16, 10), diagonal = 'kde');


# In[10]:


# calculate the correlation matrix
corr = data.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(16, 8))

# plot the heatmap
sns.set_style({'axes.edgecolor': '1.'})
_ = sns.heatmap(corr, linewidths=.25, mask=mask, center=0.5, cmap='RdBu_r')


# ### Question 3
# * Using the scatter matrix as a reference, discuss the distribution of the dataset, specifically talk about the normality, outliers, large number of data points near 0 among others. If you need to sepearate out some of the plots individually to further accentuate your point, you may do so as well.
# * Are there any pairs of features which exhibit some degree of correlation? 
# * Does this confirm or deny your suspicions about the relevance of the feature you attempted to predict? 
# * How is the data for those features distributed?
# 
# **Hint:** Is the data normally distributed? Where do most of the data points lie? You can use [corr()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.corr.html) to get the feature correlations and then visualize them using a [heatmap](http://seaborn.pydata.org/generated/seaborn.heatmap.html)(the data that would be fed into the heatmap would be the correlation values, for eg: `data.corr()`) to gain further insight.

# **Answer:**
# 
# All features are _log-normal and right-skewed distributed_ as most of data is highly concentrated around 25% of the spread of values (_i.e._ the range from _min_ and _max_ values) and the frequency start dropping after. There is some outliers in all features. A good idea for a feature engineering is to apply some math _log_ scale (like using numpy's [`log10` method](https://docs.scipy.org/doc/numpy/reference/generated/numpy.log10.html)) to transform them into something more like a normal distribution.
# 
# According to the heatmap above the most correlated pair of features is `Grocery` and `Detergents_Paper` with a correlation of almost 1.0. Notice that `Milk` also has a mid-correlation with those (~0.60).
#     
# `Milk` and `Detergents_Paper` are examples of things you can find in a grocery store, so one explanation is that you can find items like those together most of times in a shop cart.
# 
# Also, notice that `Delicatessen` feature, the one with the lowest `R^2` from the exercise above, has overall the lowest correlation among them all, reinforcing its importance in clustering customers further on this project.

# ## Data Preprocessing
# In this section, you will preprocess the data to create a better representation of customers by performing a scaling on the data and detecting (and optionally removing) outliers. Preprocessing data is often times a critical step in assuring that results you obtain from your analysis are significant and meaningful.

# ### Implementation: Feature Scaling
# If data is not normally distributed, especially if the mean and median vary significantly (indicating a large skew), it is most [often appropriate](http://econbrowser.com/archives/2014/02/use-of-logarithms-in-economics) to apply a non-linear scaling — particularly for financial data. One way to achieve this scaling is by using a [Box-Cox test](http://scipy.github.io/devdocs/generated/scipy.stats.boxcox.html), which calculates the best power transformation of the data that reduces skewness. A simpler approach which can work in most cases would be applying the natural logarithm.
# 
# In the code block below, you will need to implement the following:
#  - Assign a copy of the data to `log_data` after applying logarithmic scaling. Use the `np.log` function for this.
#  - Assign a copy of the sample data to `log_samples` after applying logarithmic scaling. Again, use `np.log`.

# In[11]:


# TODO: Scale the data using the natural logarithm
log_data = np.log(data)

# TODO: Scale the sample data using the natural logarithm
data_samples = data.sample(10) # Using 10 examples arbitrarily as the exercise did not specified the sample size
log_samples = np.log(data_samples) 

# Produce a scatter matrix for each pair of newly-transformed features
pd.plotting.scatter_matrix(log_data, alpha = 1.0, figsize = (16, 10), diagonal = 'kde');


# ### Observation
# After applying a natural logarithm scaling to the data, the distribution of each feature should appear much more normal. For any pairs of features you may have identified earlier as being correlated, observe here whether that correlation is still present (and whether it is now stronger or weaker than before).
# 
# Run the code below to see how the sample data has changed after having the natural logarithm applied to it.

# In[12]:


# Display the log-transformed sample data
display(log_samples)


# ### Implementation: Outlier Detection
# Detecting outliers in the data is extremely important in the data preprocessing step of any analysis. The presence of outliers can often skew results which take into consideration these data points. There are many "rules of thumb" for what constitutes an outlier in a dataset. Here, we will use [Tukey's Method for identfying outliers](http://datapigtechnologies.com/blog/index.php/highlighting-outliers-in-your-data-with-the-tukey-method/): An *outlier step* is calculated as 1.5 times the interquartile range (IQR). A data point with a feature that is beyond an outlier step outside of the IQR for that feature is considered abnormal.
# 
# In the code block below, you will need to implement the following:
#  - Assign the value of the 25th percentile for the given feature to `Q1`. Use `np.percentile` for this.
#  - Assign the value of the 75th percentile for the given feature to `Q3`. Again, use `np.percentile`.
#  - Assign the calculation of an outlier step for the given feature to `step`.
#  - Optionally remove data points from the dataset by adding indices to the `outliers` list.
# 
# **NOTE:** If you choose to remove any outliers, ensure that the sample data does not contain any of these points!  
# Once you have performed this implementation, the dataset will be stored in the variable `good_data`.

# In[13]:


log_data.describe()


# In[14]:


a = np.array([1, 2, 3])
b = np.array([1, 4, 5])


# In[15]:


np.unique(np.append(a, b))


# In[16]:


all_outliers  = []
extreme_outliers = []

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    
    # TODO: Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    
    # TODO: Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    
    # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3 - Q1)
    extreme_outlier_step = 3 * (Q3 - Q1)
    
    # Display the outliers
    print("Data points considered outliers for the feature '{}':".format(feature))
    outliers_upper_threshold = log_data[feature] >= Q3 + step
    outliers_lower_threshold = log_data[feature] <= Q1 - step
    outliers_mask = (outliers_upper_threshold) | (outliers_lower_threshold)
    all_outliers.append(list(log_data[outliers_mask].index))
    display(log_data[outliers_mask])
    
    # Extreme outliers which will be removed later
    extreme_outliers_upper_threshold = log_data[feature] >= Q3 + extreme_outlier_step
    extreme_outliers_lower_threshold = log_data[feature] <= Q1 - extreme_outlier_step
    extreme_outliers_mask = (extreme_outliers_upper_threshold) | (extreme_outliers_lower_threshold)
    extreme_outliers.append(list(log_data[extreme_outliers_mask].index))
    
# OPTIONAL: Select the indices for data points you wish to remove
flat_outliers = [item for sublist in all_outliers for item in sublist]
outliers_in_more_than_one_category = np.unique(
    np.array([x for x in flat_outliers if flat_outliers.count(x) > 1])
)

print(f'Outliers in more than one category: {outliers_in_more_than_one_category}')

# Remove the outliers, if any were specified
extreme_outliers_idx = np.array([idx for feature in extreme_outliers for idx in feature])
outliers_idx = np.unique(np.append(extreme_outliers_idx, outliers_in_more_than_one_category))

good_data = log_data.drop(
    log_data.index[outliers_idx]
).reset_index(drop=True)
number_of_outliers = log_data.shape[0] - good_data.shape[0]

print(f'Removing outliers: {number_of_outliers} rows removed')


# ### Question 4
# * Are there any data points considered outliers for more than one feature based on the definition above? 
# * Should these data points be removed from the dataset? 
# * If any data points were added to the `outliers` list to be removed, explain why.
# 
# ** Hint: ** If you have datapoints that are outliers in multiple categories think about why that may be and if they warrant removal. Also note how k-means is affected by outliers and whether or not this plays a factor in your analysis of whether or not to remove them.

# **Answer:**
# There are some datapoints considered outliers for more than one feature. Here are their indexes: `65`, `66`, `75`, `128` and `154`. They need to be removed as some clustering algorithm are way sensitive to outliers, such as K-Means and datapoints with outliers in more than one category could lead to misconclusions.
# 
# 
# We will also remove extreme outliers out of the range of 3 * IQR, resulting in 11 removed rows.

# ## Feature Transformation
# In this section you will use principal component analysis (PCA) to draw conclusions about the underlying structure of the wholesale customer data. Since using PCA on a dataset calculates the dimensions which best maximize variance, we will find which compound combinations of features best describe customers.

# ### Implementation: PCA
# 
# Now that the data has been scaled to a more normal distribution and has had any necessary outliers removed, we can now apply PCA to the `good_data` to discover which dimensions about the data best maximize the variance of features involved. In addition to finding these dimensions, PCA will also report the *explained variance ratio* of each dimension — how much variance within the data is explained by that dimension alone. Note that a component (dimension) from PCA can be considered a new "feature" of the space, however it is a composition of the original features present in the data.
# 
# In the code block below, you will need to implement the following:
#  - Import `sklearn.decomposition.PCA` and assign the results of fitting PCA in six dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

# In[17]:


from sklearn.decomposition import PCA

# TODO: Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA(n_components=6, random_state=42).fit(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Generate PCA results plot
pca_results = vs.pca_results(good_data, pca)


# ### Question 5
# 
# * How much variance in the data is explained* **in total** *by the first and second principal component? 
# * How much variance in the data is explained by the first four principal components? 
# * Using the visualization provided above, talk about each dimension and the cumulative variance explained by each, stressing upon which features are well represented by each dimension(both in terms of positive and negative variance explained). Discuss what the first four dimensions best represent in terms of customer spending.
# 
# **Hint:** A positive increase in a specific dimension corresponds with an *increase* of the *positive-weighted* features and a *decrease* of the *negative-weighted* features. The rate of increase or decrease is based on the individual feature weights.

# In[18]:


print(pca_results['Explained Variance'].cumsum())


# **Answer:**
# 
# As the first principal component explains `0.4764` of the variance and the second one `0.2527` then the total of the first two principal components explains `0.7291` of the total variance.
# 
# The first four principal components explains `0.9350` of the total variance.
# 
# The first dimension focus more on explaining customers which prefers supermarkets or grocery stores instead of restaurants and it's responsible for almost 50% of the variance explanation. 
# 
# The second dimension does the oposite and can explain more about customers who spends more on restaurants and likes some fresh food. Along with the first dimension we sum up `0.7291` of the total variance. 
# 
# The third dimension results in a high negative weight for the `Fresh` feature and a high positive weight for `Delicatessen` and `Frozen`. The total explained variance goes to a value of `0.8467` when this dimension is added-up to the other first two.
# 
# The four dimension results results in a high positive weight for the `Frozen` category and a somewhat high negative `Delicatessen`. Along with the first three dimensions the total explained variance is of `0.9350`.

# ### Observation
# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it in six dimensions. Observe the numerical value for the first four dimensions of the sample points. Consider if this is consistent with your initial interpretation of the sample points.

# In[19]:


# Display sample log-data after having a PCA transformation applied
display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))


# ### Implementation: Dimensionality Reduction
# When using principal component analysis, one of the main goals is to reduce the dimensionality of the data — in effect, reducing the complexity of the problem. Dimensionality reduction comes at a cost: Fewer dimensions used implies less of the total variance in the data is being explained. Because of this, the *cumulative explained variance ratio* is extremely important for knowing how many dimensions are necessary for the problem. Additionally, if a signifiant amount of variance is explained by only two or three dimensions, the reduced data can be visualized afterwards.
# 
# In the code block below, you will need to implement the following:
#  - Assign the results of fitting PCA in two dimensions with `good_data` to `pca`.
#  - Apply a PCA transformation of `good_data` using `pca.transform`, and assign the results to `reduced_data`.
#  - Apply a PCA transformation of `log_samples` using `pca.transform`, and assign the results to `pca_samples`.

# In[20]:


# TODO: Apply PCA by fitting the good data with only two dimensions
pca = PCA(n_components=2, random_state=42).fit(good_data)

# TODO: Transform the good data using the PCA fit above
reduced_data = pca.transform(good_data)

# TODO: Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])


# ### Observation
# Run the code below to see how the log-transformed sample data has changed after having a PCA transformation applied to it using only two dimensions. Observe how the values for the first two dimensions remains unchanged when compared to a PCA transformation in six dimensions.

# In[21]:


# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns = ['Dimension 1', 'Dimension 2']))


# ## Visualizing a Biplot
# A biplot is a scatterplot where each data point is represented by its scores along the principal components. The axes are the principal components (in this case `Dimension 1` and `Dimension 2`). In addition, the biplot shows the projection of the original features along the components. A biplot can help us interpret the reduced dimensions of the data, and discover relationships between the principal components and original features.
# 
# Run the code cell below to produce a biplot of the reduced-dimension data.

# In[22]:


# Create a biplot
vs.biplot(good_data, reduced_data, pca)


# ### Observation
# 
# Once we have the original feature projections (in red), it is easier to interpret the relative position of each data point in the scatterplot. For instance, a point the lower right corner of the figure will likely correspond to a customer that spends a lot on `'Milk'`, `'Grocery'` and `'Detergents_Paper'`, but not so much on the other product categories. 
# 
# From the biplot, which of the original features are most strongly correlated with the first component? What about those that are associated with the second component? Do these observations agree with the pca_results plot you obtained earlier?

# **Answer**: 
# 
# `Detergents_Paper`, `Grocery` and `Milk` are the three original features more correlated with the first component as their projections are close. As for the second component the `Frozen` and `Fresh` features are the most correlated.

# ## Clustering
# 
# In this section, you will choose to use either a K-Means clustering algorithm or a Gaussian Mixture Model clustering algorithm to identify the various customer segments hidden in the data. You will then recover specific data points from the clusters to understand their significance by transforming them back into their original dimension and scale. 

# ### Question 6
# 
# * What are the advantages to using a K-Means clustering algorithm? 
# * What are the advantages to using a Gaussian Mixture Model clustering algorithm? 
# * Given your observations about the wholesale customer data so far, which of the two algorithms will you use and why?
# 
# ** Hint: ** Think about the differences between hard clustering and soft clustering and which would be appropriate for our dataset.

# **Answer:**
# 
# Some advantages on using a *K-Means clustering*:
#     * Simple to understand and easy to implement
#     * It's a fast algorithm
#     * It does a very good job on grouping spherical shape clusters
#     
# Some advantages on using a *Gaussian Mixture Model (GMM)* clustering:
#     * Each datapoint result in a probability to result in each cluster found, giving more flexibility in solving certain problems where categories can overlap.
#     * More robust to outliers as the algorithm doesn't use Elucidian distance
#     
# Giving that the dataset is somewhat sparsed without some spherical shape nodes and contains some outliers or distant points, I will use the **Gaussian Mixture Model** to cluster.

# ### Implementation: Creating Clusters
# Depending on the problem, the number of clusters that you expect to be in the data may already be known. When the number of clusters is not known *a priori*, there is no guarantee that a given number of clusters best segments the data, since it is unclear what structure exists in the data — if any. However, we can quantify the "goodness" of a clustering by calculating each data point's *silhouette coefficient*. The [silhouette coefficient](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html) for a data point measures how similar it is to its assigned cluster from -1 (dissimilar) to 1 (similar). Calculating the *mean* silhouette coefficient provides for a simple scoring method of a given clustering.
# 
# In the code block below, you will need to implement the following:
#  - Fit a clustering algorithm to the `reduced_data` and assign it to `clusterer`.
#  - Predict the cluster for each data point in `reduced_data` using `clusterer.predict` and assign them to `preds`.
#  - Find the cluster centers using the algorithm's respective attribute and assign them to `centers`.
#  - Predict the cluster for each sample data point in `pca_samples` and assign them `sample_preds`.
#  - Import `sklearn.metrics.silhouette_score` and calculate the silhouette score of `reduced_data` against `preds`.
#    - Assign the silhouette score to `score` and print the result.

# In[23]:


from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score

# TODO: Apply your clustering algorithm of choice to the reduced data 
clusterer_with_two_components = GaussianMixture(n_components=2, n_init=10, max_iter=1000, random_state=42).fit(reduced_data)

# TODO: Predict the cluster for each data point
preds_with_two_components = clusterer_with_two_components.predict(reduced_data)

# TODO: Find the cluster centers
centers = clusterer_with_two_components.means_

# TODO: Predict the cluster for each transformed sample data point
sample_preds = clusterer_with_two_components.predict(pca_samples)

# TODO: Calculate the mean silhouette coefficient for the number of clusters chosen
score = silhouette_score(reduced_data, preds_with_two_components)


# ### Question 7
# 
# * Report the silhouette score for several cluster numbers you tried. 
# * Of these, which number of clusters has the best silhouette score?

# In[24]:


silhouette_scores = []
n_clusters = list(range(2, 11))

for n_cluster in n_clusters:
    clusterer = GaussianMixture(n_components=n_cluster, n_init=10, max_iter=1000, random_state=42).fit(reduced_data)
    preds = clusterer.predict(reduced_data)
    silhouette_scores.append(silhouette_score(reduced_data, preds))


# In[25]:


plt.figure(figsize=(16, 6))
plt.plot(n_clusters, silhouette_scores)
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score by Number of Clusters using Gaussian Mixture Model')
plt.grid(True)

plt.show()


# **Answer:**
# The best silhouette score was resulted with 2 clusters.

# ### Cluster Visualization
# Once you've chosen the optimal number of clusters for your clustering algorithm using the scoring metric above, you can now visualize the results by executing the code block below. Note that, for experimentation purposes, you are welcome to adjust the number of clusters for your clustering algorithm to see various visualizations. The final visualization provided should, however, correspond with the optimal number of clusters. 

# In[26]:


# Display the results of the clustering from implementation
vs.cluster_results(reduced_data, preds_with_two_components, centers, pca_samples)


# ### Implementation: Data Recovery
# Each cluster present in the visualization above has a central point. These centers (or means) are not specifically data points from the data, but rather the *averages* of all the data points predicted in the respective clusters. For the problem of creating customer segments, a cluster's center point corresponds to *the average customer of that segment*. Since the data is currently reduced in dimension and scaled by a logarithm, we can recover the representative customer spending from these data points by applying the inverse transformations.
# 
# In the code block below, you will need to implement the following:
#  - Apply the inverse transform to `centers` using `pca.inverse_transform` and assign the new centers to `log_centers`.
#  - Apply the inverse function of `np.log` to `log_centers` using `np.exp` and assign the true centers to `true_centers`.
# 

# In[27]:


# TODO: Inverse transform the centers
log_centers = pca.inverse_transform(centers)

# TODO: Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)


# ### Question 8
# 
# * Consider the total purchase cost of each product category for the representative data points above, and reference the statistical description of the dataset at the beginning of this project(specifically looking at the mean values for the various feature points). What set of establishments could each of the customer segments represent?
# 
# **Hint:** A customer who is assigned to `'Cluster X'` should best identify with the establishments represented by the feature set of `'Segment X'`. Think about what each segment represents in terms their values for the feature points chosen. Reference these values with the mean values to get some perspective into what kind of establishment they represent.

# **Answer:**

# First let's recap both mean and standard deviation values of all features:

# In[28]:


display(data.describe().loc[['mean', 'std'], :])


# There are three features which their values for each segment diverges in being above or below the mean: `Milk`, `Grocery` and `Detergents_Paper`. 
# 
# The `Segment 0` has those three values **below** the mean whereas `Segment 1` has those values **above** the mean.
# 
# We may find out that `Segment 0` can represent people who goes less in a grocery store but spend more on fresh markets or restaurants and `Segment 1` people who are heavily spenders on grocery stores or supermarkets.

# ### Question 9
# 
# * For each sample point, which customer segment from* **Question 8** *best represents it? 
# * Are the predictions for each sample point consistent with this?*
# 
# Run the code block below to find which cluster each sample point is predicted to be.

# In[29]:


data_samples['cluster_pred'] = sample_preds
data_samples


# In[30]:


# Display the predictions
for i, pred in enumerate(sample_preds):
    print("Sample point", i, "predicted to be in Cluster", pred)


# **Answer:**
# Most of predictions seems consistent with the definitions made on exercise 8. Good examples for each cluster can be found at customer #115 for the Cluster 0 (fresh markets or restaurants) and customer #159 for the Cluster 1 (grocery storey/supermarket person).
# 
# However there are some predictions in which I don't totally agree with, such as #265, #181 and #352. 
# 
# The first one was probably put into Cluster 0 because of the low value on `Detergents_Paper` as if we look at `Milk` and `Grocery` values are higher than the average. 
# 
# The second one is a heavy spender at all, with a high spend on `Fresh` but also a considerable spend on the other categories. Due to its high spend on `Fresh` and `Frozen` I would put this customer on Cluster 0 instead.
# 
# Finally the third one has low values on all features overall but focusing their spends on `Milk` and `Grocery`. Probably because of a low spend on `Detergents_Paper` this customer was classified as `Cluster 0`. If I had to make this decisions I'd put this customer on `Cluster 1` instead.

# ## Conclusion

# In this final section, you will investigate ways that you can make use of the clustered data. First, you will consider how the different groups of customers, the ***customer segments***, may be affected differently by a specific delivery scheme. Next, you will consider how giving a label to each customer (which *segment* that customer belongs to) can provide for additional features about the customer data. Finally, you will compare the ***customer segments*** to a hidden variable present in the data, to see whether the clustering identified certain relationships.

# ### Question 10
# Companies will often run [A/B tests](https://en.wikipedia.org/wiki/A/B_testing) when making small changes to their products or services to determine whether making that change will affect its customers positively or negatively. The wholesale distributor is considering changing its delivery service from currently 5 days a week to 3 days a week. However, the distributor will only make this change in delivery service for customers that react positively. 
# 
# * How can the wholesale distributor use the customer segments to determine which customers, if any, would react positively to the change in delivery service?*
# 
# **Hint:** Can we assume the change affects all customers equally? How can we determine which group of customers it affects the most?

# **Answer:**
# 
# As the wholesale wants to lower the rate from 5 days/week to 3 days/week this change may impact negatively customers who wants fresh things or businesses which can't stock supplies for extra 2 days. 
# 
# Given that, thinking about our clusters 0 and 1 from **Question 8**, I'd say customers from `Cluster 1` can react more positively compared to the `Cluster 0`.
# 
# In order to test this hypothesis I would separate the customers in four groups:
# 
# |Test Iteration |Group | Description                     | Delivery Service Rate | Satisfying Rate (%) |
# |---------------|------|---------------------------------|-----------------------|---------------------|
# | 1             | 1    | 50% of customers from Cluster 0 | 5 days/week           | - |
# | 1             | 2    | 50% of customers from Cluster 0 | 3 days/week           | - |
# | 2             | 3    | 50% of customers from Cluster 1 | 5 days/week           | - |
# | 2             | 4    | 50% of customers from Cluster 1 | 3 days/week           | - |
# 
# This A/B test would be split into two iterations, each testing one cluster segment, and check for their satisfying rates (a hypothetic metric which will guide our decision of *positive* or *negative* reaction).
# 
# After run this test until data reaches a statistical significance, let's say we have the following result:
# 
# |Test Iteration |Group | Description                     | Delivery Service Rate | Satisfying Rate (%) |
# |---------------|------|---------------------------------|-----------------------|---------------------|
# | 1             | 1    | 50% of customers from Cluster 0 | 5 days/week           | 80.7 |
# | 1             | 2    | 50% of customers from Cluster 0 | 3 days/week           | 71.5 |
# | 2             | 3    | 50% of customers from Cluster 1 | 5 days/week           | 88.3 |
# | 2             | 4    | 50% of customers from Cluster 1 | 3 days/week           | 88.2 |
# 
# For Cluster 0 customers, the lower in rate affected *negatively* as the satisfying rate has dropped from 80.7% to 71.5%. 
# 
# For the other cluster, although the satisfying rate has remained almost the same, it also has dropped from 88.3% to 88.2%. Assuming statistical significance, we may also concludes that the lower in rate affected those customers negatively*.

# ### Question 11
# Additional structure is derived from originally unlabeled data when using clustering techniques. Since each customer has a ***customer segment*** it best identifies with (depending on the clustering algorithm applied), we can consider *'customer segment'* as an **engineered feature** for the data. Assume the wholesale distributor recently acquired ten new customers and each provided estimates for anticipated annual spending of each product category. Knowing these estimates, the wholesale distributor wants to classify each new customer to a ***customer segment*** to determine the most appropriate delivery service.  
# * How can the wholesale distributor label the new customers using only their estimated product spending and the **customer segment** data?
# 
# **Hint:** A supervised learner could be used to train on the original customers. What would be the target variable?

# **Answer:**
# 
# The more straightforward strategy would simply using our customer clusterer `clusterer_with_two_components` and the PCA pre-processor trained above with the new data in order to label the new 10 customers.
# 
# We may also label our original data with the clusterer output and train a supervised learning algorithm using the cluster segment as target.

# ### Visualizing Underlying Distributions
# 
# At the beginning of this project, it was discussed that the `'Channel'` and `'Region'` features would be excluded from the dataset so that the customer product categories were emphasized in the analysis. By reintroducing the `'Channel'` feature to the dataset, an interesting structure emerges when considering the same PCA dimensionality reduction applied earlier to the original dataset.
# 
# Run the code block below to see how each data point is labeled either `'HoReCa'` (Hotel/Restaurant/Cafe) or `'Retail'` the reduced space. In addition, you will find the sample points are circled in the plot, which will identify their labeling.

# In[31]:


# Display the clustering results based on 'Channel' data
vs.channel_results(reduced_data, outliers_in_more_than_one_category, pca_samples)


# ### Question 12
# 
# * How well does the clustering algorithm and number of clusters you've chosen compare to this underlying distribution of Hotel/Restaurant/Cafe customers to Retailer customers? 
# * Are there customer segments that would be classified as purely 'Retailers' or 'Hotels/Restaurants/Cafes' by this distribution? 
# * Would you consider these classifications as consistent with your previous definition of the customer segments?

# **Answer:**
# 
# The cluster model did a pretty decent job on find a separation both two clusters. The are some `HoReCa` category points more on the left cluster which were misclassified as `Retailer` ones but most are somewhat close to the separation line (something between -1 and -.5 on `Dimension 1`).
# 
# Maybe the furthest points from the separation line could be more likely to be "purely" _Retailers_ on the bottom left or _HoReCa_ on the top right.
# 
# These classifications are consistents with the previouly defined segments as _Retailers_ ones were defined as _heavily spenders on grocery stores or supermarkets_ and _HoReCa_ were the ones who _spend more on fresh markets or restaurants_.

# > **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  
# **File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.
