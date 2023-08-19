### Linear Regression
The code uses linear regression to predict the final grades that students from a dataset received. Linear regression is a machine-learning algorithm that is used to interpret and analyse data to make a prediction. In this case, we will take information about the students (which is the dataset) and then implement linear regression to use the information and deduce what the final grade of the students will be.

### Code Steps
1. We begin by importing the necessary libraries that will be needed to implement machine learning and display the result data in a graph:
```
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as pyplot
import pickle
from sklearn import linear_model
from sklearn.utils import shuffle
from matplotlib import style
```
2. We use the Pandas library to display the dataset that will be used:
```
data = pd.read_csv("student-mat.csv", sep=";")
print (data.head())  
```
3. We filter out the relevant data (attributes) that we will use for the prediction and then display it:
```
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]
print(data.head())
     
```
4. The target variable that we are attempting to predict is set to "G3" while the attributes we will be using to predict are stored in the array x, and the actual "G3" results (which are our labels) are stored in the array y:
```
predict = "G3"

x = np.array(data.drop(predict, axis=1))
y = np.array(data[predict])
```
5. We split the data into 'train' and 'test' sets which will be used to train the model via machine learning in the next step:
```
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)
```
6. The model is trained on the training data, making a prediction of the student's final grade, its accuracy score is then calculated (using the testing data) and printed using the true value of the student's final grade so we can gauge how accurate the model is:
```
linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)
print(acc)
```
7. The trained model is saved to a file named "studentmodel.pickle" using the pickle library. This allows the model to be reused without retraining:
```
with open("studentmodel.pickle", "wb") as f:
    pickle.dump(linear, f)

pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)
```
8. The coefficients, which represent the weights assigned to each attribute by the linear regression model, are printed. These coefficients indicate how much each attribute affects the final prediction, with some attributes being more influential than others. The intercept is also printed which provides the model with a starting point for the prediction which is then adjusted based on the influence of the attributes:
```
print('Coefficient: \n', linear.coef_)
print('Intercept: \n', linear.intercept_)
```
9. The results are printed, displaying the model's prediction alongside the attributes that were used as well as the true result of the final exam:
```
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])
```
10. The results are then visualised in a graph, the code below is used to show the relationship between the students "G2" and "G3" but it can be altered to display any other trends within the data
```
style.use("ggplot")

# Set up a scatter pyplot
p = "G2"
pyplot.scatter(data[p], data["G3"])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade")
pyplot.show()
```
