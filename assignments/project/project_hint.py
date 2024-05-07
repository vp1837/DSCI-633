import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from geneticalgorithm import geneticalgorithm as ga
from my_evaluation import my_evaluation

class my_model():

    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english', norm='l2', use_idf=True, smooth_idf=True)
        self.clf = None

    def obj_func(self, X):
        # Decode hyperparameters
        n_estimators = int(X[0])
        max_depth = int(X[1]) if X[1] > 0 else None  # None is unlimited depth

        # Model training
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        XX = self.vectorizer.transform(self.X_train)
        model.fit(XX, self.y_train)

        # Evaluation using F1 score
        XX_test = self.vectorizer.transform(self.X_test)
        predictions = model.predict(XX_test)
        eval = my_evaluation(predictions, self.y_test)
        return -eval.f1()  # Minimize negative F1 score

    def fit(self, X, y):
        # Splitting data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Fit vectorizer on training data
        self.vectorizer.fit(self.X_train)

        # Define variable boundaries for hyperparameters
        varbound = np.array([[10, 200],  # n_estimators range
                             [0, 50]])  # max_depth range, where 0 means no limit

        # Genetic algorithm parameters
        algorithm_param = {
            'max_num_iteration': 100,
            'population_size': 10,
            'mutation_probability': 0.1,
            'elit_ratio': 0.01,
            'crossover_probability': 0.5,
            'parents_portion': 0.3,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': None
        }

        # Initialize GA
        model = ga(function=self.obj_func, dimension=2, variable_type='int', variable_boundaries=varbound, algorithm_parameters=algorithm_param)

        # Run GA
        model.run()

        # Best solution from GA
        best_solution = model.output_dict['variable']
        self.clf = RandomForestClassifier(n_estimators=int(best_solution[0]), max_depth=int(best_solution[1]) if best_solution[1] > 0 else None, random_state=42)
        XX_train = self.vectorizer.transform(self.X_train)
        self.clf.fit(XX_train, self.y_train)

    def predict(self, X):
        XX = self.vectorizer.transform(X)
        return self.clf.predict(XX)
