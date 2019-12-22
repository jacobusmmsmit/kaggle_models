from pathlib import Path

import pandas as pd
from plotnine import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

data_file = Path("./data/mushrooms.csv")

data = pd.read_csv(data_file)

# Create a boolean column of whether the mushroom is poisonous
data["poisonous"] = [x == "p" for x in data["class"]]

# Define features
y = data["poisonous"]
X = data.drop(columns=["poisonous", "class"])

# One-hot-encode the features
encoder = OneHotEncoder()
encoder.fit(X)
X_ohe = encoder.transform(X).toarray()

# Split into test/train datasets
X_train, X_test, y_train, y_test = train_test_split(
    X_ohe, y, test_size=0.2, train_size=0.8
)


def print_accuracy(acc: float, model_name: str) -> None:
    """Prints the accuracy of a model in a nice format."""
    print(f"accuracy of {model_name} = {round(100* acc,2)}%")


# Train a logistic regression model
logistic_model = LogisticRegression().fit(X=X_train, y=y_train)
y_pred_log_reg = logistic_model.predict(X_test)
y_pred_log_reg = [i > 0.01 for i in y_pred_log_reg]
accuracy_log_reg = accuracy_score(
    y_test, y_pred_log_reg, normalize=True, sample_weight=None
)

# Train an artificial neural network model
ann_model = MLPClassifier().fit(X=X_train, y=y_train)
y_pred_ann = ann_model.predict(X_test)
accuracy_ann = accuracy_score(y_test, y_pred_ann, normalize=True, sample_weight=None)

# Print accuracies
print_accuracy(accuracy_log_reg, "logistic regression")
print_accuracy(accuracy_ann, "neural network")
