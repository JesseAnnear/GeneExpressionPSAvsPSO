import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import random
# paths to both our entire gene set and also the list of significant genes
path = "significant_genes.csv"
path2 = "GSE61281.csv"

# Read both of the files
significant = pd.read_csv(path,index_col=0)
ex = pd.read_csv(path2,index_col=0)

# filter out the non-significant genes
significant = significant[(significant['PSAvC'] != 0) | (significant['PSAvPSO'] != 0)]
ex = ex[ex.index.isin(significant.index)]

# Get rid of the control group as we are only interesting differentiate between PSO only and PSA
# First twenty is PSA and the tail twenty is the PSO group
ex = ex.iloc[:,:40]

# turn in numpy array rather than dataframe for machine learning models
X = ex.values  

# Add zeros for the PSA group and ones for the PSO group 
y = np.concatenate((np.zeros(20), np.ones(20)))
y = np.round(y).astype(int)


#create arrays to track the accurracies of the four different models
accuracy = []
accuracy2 = []
accuracy3 = []
accuracy4 = []

# test each model 50 times, each time randomizing the random state
for i in range(0,50):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X.T, y, test_size=0.3, random_state=random.randint(0,99))

    # Create and train a machine learning model (e.g., logistic regression)
    model = LogisticRegression()
    model2 = KNeighborsClassifier()
    model3 = RandomForestClassifier()
    model4 = MLPClassifier()

    model.fit(X_train, y_train)
    model2.fit(X_train,y_train)
    model3.fit(X_train,y_train)
    model4.fit(X_train,y_train)
    # Evaluate the model
    accuracy.append(model.score(X_test, y_test))
    accuracy2.append(model2.score(X_test,y_test))
    accuracy3.append(model3.score(X_test,y_test))
    accuracy4.append(model4.score(X_test,y_test))

# Calculate the mean of eah of the models
mean = sum(accuracy)/len(accuracy)
mean2 = sum(accuracy2)/len(accuracy2)
mean3 = sum(accuracy3)/len(accuracy3)
mean4 = sum(accuracy4)/len(accuracy4)

# plot each of the models
plt.plot(accuracy, label = "Logistic Regression", color = 'r')
plt.plot(accuracy2, label = "KNN",color = 'b')
plt.plot(accuracy3,label = "Random Forest",color = 'g')
plt.plot(accuracy4, label = "MLP",color = 'purple')

# Write the mean of each of the models in text
plt.text(0.1, mean, f'Mean Linear Regression: {mean:.3f}', color='r', ha='left', va='center')
plt.text(0.1, mean2, f'Mean KNN: {mean2:.3f}', color='b', ha='left', va='center')
plt.text(0.1, mean3, f'Mean Random Forest: {mean3:.3f}', color='g', ha='left', va='center')
plt.text(0.1, mean4, f'Mean MLP: {mean4:.3f}', color='purple', ha='left', va='center')

# Plot our title and show our plots
plt.title("Differentiating PSA and PSO Model Accuracies")
plt.show()