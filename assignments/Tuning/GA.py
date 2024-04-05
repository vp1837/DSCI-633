import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from geneticalgorithm import geneticalgorithm as ga


# import my_evaluation correctly.
from my_evaluation import my_evaluation

# Load the Breast Cancer dataset
df = pd.read_csv('C:\\Users\\vpark\\Vee\\DSCI\\DSCI-633\\assignments\\data\\breast_cancer.csv')  

# Initialize LabelEncoder
le = LabelEncoder()

# Encode categorical columns
for col in df.columns:
    if df[col].dtype == 'object':  # If the column is categorical
        df[col] = le.fit_transform(df[col])  # Transform it to numerical

X = df.drop('Class', axis=1)  # Features
y = df['Class']  # Target

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def f(X):
    # Convert GA solution to model parameters
    criterion = 'gini' if X[0] == 0 else 'entropy'
    max_depth = int(X[1])

    # Train DecisionTreeClassifier
    model = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
    model.fit(X_train, y_train)
    
    # Predictions
    predictions = model.predict(X_test)
    
    # Evaluate model using macro F1 score
    evaluator = my_evaluation(predictions, y_test)
    f1_score = evaluator.f1(average='macro')
    
    # The GA is maximizing the objective, so return the negative F1 score
    return -f1_score

# Setup the genetic algorithm
varbound = np.array([[0,1], [1,12]])  # Boundaries correctly set for criterion (gini, entropy) and max_depth

# Initializing the genetic algorithm with the correct variable type and boundaries
model = ga(function=f, dimension=2, variable_type='int', variable_boundaries=varbound)

# Run the genetic algorithm
model.run()
