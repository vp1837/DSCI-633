from my_evaluation import my_evaluation
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

if __name__ == "__main__":
    # Load training data
    data_train = pd.read_csv("../data/breast_cancer.csv")

    # Separate independent variables and dependent variables
    independent = ["age","menopause","tumor-size","inv-nodes","node-caps","deg-malig",\
                   "breast","breast-quad","irradiat"]
    # use one-hot encoding so discrete features can be used with DecisionTreeClassifier
    X = pd.get_dummies(data_train[independent], drop_first=True)
    y = data_train["Class"]

    # Define parameter combinations
    criterions = ["gini", "entropy"]
    max_depths = [2, 3, 4, 5]

    # Iterate over parameter combinations
    for criterion in criterions:
        for max_depth in max_depths:
            # Fit model
            clf = DecisionTreeClassifier(criterion=criterion, max_depth=max_depth)
            clf.fit(X, y)

            # Predict on training data
            predictions = clf.predict(X)

            # Evaluate results
            metrics = my_evaluation(predictions, y)
            result = {}
            for target in clf.classes_:
                result[target] = {}
                result[target]["prec"] = metrics.precision(target)
                result[target]["recall"] = metrics.recall(target)
                result[target]["f1"] = metrics.f1(target)

            # Print results
            print(f"Criterion: {criterion}, Max Depth: {max_depth}")
            print(result)
            print()
