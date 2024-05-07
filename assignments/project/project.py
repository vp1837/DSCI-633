import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
from geneticalgorithm import geneticalgorithm as ga
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

nltk.download('wordnet')
nltk.download('stopwords')

class my_model():
    def __init__(self):
        custom_stopwords = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.preprocessor = TfidfVectorizer(stop_words=list(self.processed_stopwords(custom_stopwords)),
                                            tokenizer=self.tokenize, norm='l2', use_idf=True, smooth_idf=True,
                                            ngram_range=(1,3), max_features=10000)

    def processed_stopwords(self, stopwords):
        return {self.lemmatizer.lemmatize(word) for word in stopwords}

    def tokenize(self, text):
        words = self.tokenizer.tokenize(text.lower())
        return [self.lemmatizer.lemmatize(word) for word in words]

    def obj_func(self, predictions, actuals):
        return f1_score(actuals, predictions)

    def fit(self, X, y):
        # Reset index to avoid indexing errors in cross-validation
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        XX = self.preprocessor.fit_transform(X["description"])
        selector = SelectKBest(f_classif, k=min(10000, XX.shape[1]))
        XX = selector.fit_transform(XX, y)

        def fitness_function(parameters):
            penalty = 'l2' if parameters[0] < 0.5 else 'l1'
            alpha = parameters[1]
            model = SGDClassifier(loss='log_loss', penalty=penalty, alpha=alpha, class_weight='balanced', random_state=42)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            f1_scores = []
            for train_idx, test_idx in cv.split(XX, y):
                X_train, X_test = XX[train_idx], XX[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                f1_scores.append(self.obj_func(predictions, y_test))
            return -np.mean(f1_scores)

        varbound = np.array([[0, 1], [0.00001, 0.05]])
        algorithm_param = {
            'max_num_iteration': 200,
            'population_size': 50,
            'mutation_probability': 0.2,
            'elit_ratio': 0.05,
            'crossover_probability': 0.8,
            'parents_portion': 0.4,
            'crossover_type': 'uniform',
            'max_iteration_without_improv': 40
        }

        model = ga(function=fitness_function, dimension=2, variable_type='real', variable_boundaries=varbound, algorithm_parameters=algorithm_param)
        model.run()

        best_parameters = {'penalty': 'l2' if model.output_dict['variable'][0] < 0.5 else 'l1', 'alpha': model.output_dict['variable'][1], 'class_weight': 'balanced'}
        self.clf = SGDClassifier(**best_parameters)
        self.clf.fit(XX, y)

    def predict(self, X):
        XX = self.preprocessor.transform(X["description"])
        return self.clf.predict(XX)


