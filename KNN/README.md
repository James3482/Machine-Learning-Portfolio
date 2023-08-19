### K-Nearest Neighbour and Dataset
#### The code will use K-Nearest Neighbour to predict whether or not a person has diabetes. K-NN algorithm explained: 

K is the hyperparameter or the number of members who can vote.

K (Number of Neighbours): The number of closest neighbours to take into account while creating predictions for a new data point is defined by the 'K' hyperparameter. For classification problems, for instance, if k=3, the algorithm will take into account the three closest neighbours to the query point and generate a prediction based on the majority class among those three neighbours. The algorithm's performance can be considerably impacted by the selection of 'k'.

The dataset consists of many different attributes, each revolving around health, with one of these attributes being whether or not someone has diabetes. We will use this dataset to train our model through K-NN machine learning to determine whether or not someone has diabetes based on the attributes within the dataset

### Code Steps
1. Import the libraries that our model will rely on to predict a value and display data:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
```
2. Read the dataset that we will be evaluating and training our model using and print the number of rows as well as the columns of the data:
```
data = pd.read_csv("diabetes.csv")
print (len(data))
print(data.head())
```
3. Pre proccess the data and make sure there are no zeros, since it would interfere with the models accuracy. This is done by replacing any zeroes with 'NaN' and then calculating the mean number of each row and replacing anywhere we put 'NaN' with the mean. We then print the mean of Glucose to ensure the code is working:
```
zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for column in zero_not_accepted:
    data[column] = data[column].replace(0, np.NaN)
    mean = int(data[column].mean(skipna=True))
    data[column].fillna(mean, inplace=True)

print(data['Glucose'])
print(mean)
```
4. The code separates the dataset into x (containing columns 0 to 7) and y (Outcome column, whether or not the person has diabetes). It then divides the data into training and testing sets, where 80% is for training and 20% is for testing:
```
x = data.iloc[:, 0:8] # keeps all rows, exclude column 8
y = data.iloc[:, 8] # label - column 8, whether or not you have diabetes
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)
```
5. The feature data is standardized to have a mean of 0 and a standard deviation of 1, resulting in features that are on a similar scale, making the KNN model more effective:
```
sc_x = StandardScaler() # sets all data between -1 and 1
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
```
6. To select an appropriote K value for our model we take the length of the 'y_test' array which corresponds to the number of instances in the set test. We then calculate the square root of the length of 'y_test' to provide with a good K value that will provide a reasonable balance between capturing local patterns and preventing overfitting:
```
import math
print (len(y_test))
math.sqrt(len(y_test))
```
7. The KNeighborsClassifier is initialized with hyperparameters like the number of neighbors (n_neighbors), power parameter (p), and distance (metric). The model is then trained on the standardized training data:
```
classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
classifier.fit(x_train, y_train)
```
8. The trained KNN model is applied to the standardized test data (x_test) to predict the labels. The predicted labels are stored in 'y_pred' and we then display 'y_pred' which is a array of raw values (either 1 or 0):
```
y_pred = classifier.predict(x_test)
y_pred
```
9. We then evaluate the data and display it in a confusion matrix to interpret the number of positive diabetes results that our model predicted correctly and the number of negative diabetes results that our model predicted correctly
```
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f1_score(y_test, y_pred))
```
