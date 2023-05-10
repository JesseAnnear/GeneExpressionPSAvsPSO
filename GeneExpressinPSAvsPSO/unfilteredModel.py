import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import random

# both paths for our gene set and the significant genes, although we will only use path2 for the unfiltered model
path = "significant_genes.csv"
path2 = "GSE61281.csv"

# Read the gene file
ex = pd.read_csv(path2,index_col=0)

# Get rid of the control set as we are only trying to differentiate between PSO and PSA
# The first twenty is the PSA group and the last twenty is the PSO group
ex = ex.iloc[:,:40]

# Turn into numpy array for the machine learning models
X = ex.values  

# Add zeros for the PSA group and ones for the PSO group 
y = np.concatenate((np.zeros(20), np.ones(20)))
y = np.round(y).astype(int)

import random

# Create array's to track the model accuracies for each of our models
accuracy = []
accuracy2 = []
accuracy3 = []
accuracy4 = []

# test each model 50 times, each time randomizing the random state
for i in range(0,50):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.3, random_state=random.randint(0,99))

    # Create and train a machine learning model 
    model = LogisticRegression()
    model2 = KNeighborsClassifier()
    model3 = RandomForestClassifier()
    model4 = MLPClassifier()

    # Fit our models
    model.fit(X_train, y_train)
    model2.fit(X_train,y_train)
    model3.fit(X_train,y_train)
    model4.fit(X_train,y_train)
    
    # Evaluate the model
    accuracy.append(model.score(X_test, y_test))
    accuracy2.append(model2.score(X_test,y_test))
    accuracy3.append(model3.score(X_test,y_test))
    accuracy4.append(model4.score(X_test,y_test))

# Calculate the mean of each of our model accuracy scores
mean = sum(accuracy)/len(accuracy)
mean2 = sum(accuracy2)/len(accuracy2)
mean3 = sum(accuracy3)/len(accuracy3)
mean4 = sum(accuracy4)/len(accuracy4)

# Plot our model accuracies
plt.plot(accuracy, label = "Logistic Regression", color = 'r')
plt.plot(accuracy2, label = "KNN",color = 'b')
plt.plot(accuracy3,label = "Random Forest",color = 'g')
plt.plot(accuracy4, label = "MLP",color = 'purple')

# Create text to write the mean of each of our models onto the plot
plt.text(0.1, mean, f'Mean Linear Regression: {mean:.3f}', color='r', ha='left', va='center')
plt.text(0.1, mean2, f'Mean KNN: {mean2:.3f}', color='b', ha='left', va='center')
plt.text(0.1, mean3, f'Mean Random Forest: {mean3:.3f}', color='g', ha='left', va='center')
plt.text(0.1, mean4, f'Mean MLP: {mean4:.3f}', color='purple', ha='left', va='center')

# Plot our title and show our plot
plt.title("Differentiating PSA and PSO Model Accuracies")
plt.show()