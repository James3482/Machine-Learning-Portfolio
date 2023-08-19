### K-means clustering and Dataset
The code uses K-means clustering to group iris flowers into clusters based on their feature measurements. Clustering is a technique in machine learning and data analysis that involves grouping similar data points together. In this context, clustering aims to find groups of flowers that are similar to each other in terms of measurements. The dataset is the famous Iris dataset which contains measurements of various features of three species of iris flowers. Using clustering we will group similar flowers together within a cluster to display trends within flower measurements.

### Code Steps
1. Import the necessary libraries required to perform clustering, visualization, and machine learning. Set up the size of the graph that will be used to display plotted points and visualise the data:
```
# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

import sklearn
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import scale

import sklearn.metrics as am
from sklearn import datasets
from sklearn.metrics import confusion_matrix,classification_report
     
# Graph setup
%matplotlib inline
rcParams['figure.figsize'] = 7,4
```
2. We load the Iris data set. The feature data (X) is scaled to standardize the features. The target values are stored in a DataFrame (y), and the feature names are saved in variable_names. We then display the first 10 rows of the scaled feature data:
```
iris = datasets.load_iris()

X = scale(iris.data)

y = pd.DataFrame(iris.target)
variable_names = iris.feature_names
X[0:10,]
```
3. Restrict the algorithm to only one thread in order to avoid memory leakage warnings and issues:
```
import os
os.environ['OMP_NUM_THREADS'] = '1'
```
4. The K-Means clustering algorithm is applied to the scaled feature data. n_clusters=3 indicates that we want to find 3 clusters in the data. Random_state=5 ensures the reproducibility of the results, and n_init=10 specifies the number of times the algorithm will be run with different initial centroid placement:
```
clustering = KMeans(n_clusters=3, random_state=5, n_init=10) # sets the number of centroids and randomizes centroid positions
clustering.fit(X)
```
5. The data is prepared for visualisation in the form of graphs. Two scatter plots are created, one showing the true classification based on petal length and petal width, and the other showing the classification obtained through K-Means clustering:
```
iris_df = pd.DataFrame(iris.data)
iris_df.columns = ["sepal_Length", "Sepal_Width", "Petal_Length", "Petal_Width"]
y.columns = ["Targets"]

color_theme = np.array(["darkgray", "lightsalmon", "powderblue"])

plt.subplot(1,2,1)

# Petal_Length along the x-axis and Petal_Width along the y-axis
plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[iris.target], s=50)
plt.title("Ground Truth Classification")

plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[clustering.labels_], s=50)
plt.title("K-Means Classification")
```
6. The K-Means cluster labels are relabeled to better align with the ground truth classes. The scatter plots are updated to show the comparison between the ground truth and K-Means classifications after relabeling:
```
relabel = np.choose(clustering.labels_, [2,0,1]).astype(np.int64)

plt.subplot(1,2,1)

# Petal_Length along the x-axis and Petal_Width along the y-axis
plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[iris.target], s=50)
plt.title("Ground Truth Classification")

plt.subplot(1,2,2)
plt.scatter(x=iris_df.Petal_Length, y=iris_df.Petal_Width, c=color_theme[clustering.labels_], s=50)
plt.title("K-Means Classification")
```
7. Calculate and print a classification report. This report provides various metrics like precision, recall, F1-score, and support for each class. It summarizes the performance of the clustering algorithm in terms of its ability to assign data points to the correct clusters:
```
print(classification_report(y, relabel))
```

### Conclusion
In conclusion, the algorithm demonstrates the method of K-Means clustering, an unsupervised machine learning algorithm, by using the well-known Iris datasheet. K-Means clustering aims to group data points into clusters based on their similarities, without the need for labelled target values. In this specific model, we are using the method of clustering to determine flower species based on attributes present in the dataset (measurements of the flowers).
