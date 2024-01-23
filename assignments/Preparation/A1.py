from sklearn.tree import DecisionTreeClassifier
import pandas as pd

if __name__ == "__main__":
    #  Load training data
    data_train = pd.read_csv("../data/Iris_train.csv")
    
    # Explore the loaded pandas dataframe
    # Print out the 21st training data point
    print(data_train.loc[11])
    # Print out the column "SepalWidthCm"
    print(data_train["SepalWidthCm"])
    # Print out the data points with "SepalWidthCm" < 2.5
    print(data_train[data_train["SepalWidthCm"] < 2.5])

    # Separate independent variables and dependent variables
    independent = ["SepalLengthCm",	"SepalWidthCm",	"PetalLengthCm", "PetalWidthCm"]
    X = data_train[independent]
    Y = data_train["Species"]

    # Train model
    clf = DecisionTreeClassifier()
    clf.fit(X, Y)

    # clf = GaussianNB()
    # clf.fit(X,Y)

    # Load testing data
    data_test = pd.read_csv("../data/Iris_test.csv")
    X_test = data_test[independent]

    # Predict probabilities
    predictions = clf.predict(X_test)
    # Predict probabilities directly for Decision Tree Classifier
    probs = clf.predict_proba(X_test)
    # Print results
    for i, pred in enumerate(predictions):
        print("%s\t%f" % (pred, max(probs[i])))