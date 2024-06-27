"""
Python project: Medical engineering final project;


@author: Florian Zwicker

"""

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector as selector

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt


def get_data():
    # Change the Path in the line below according to your own path
    dataset = pd.read_csv('G:\\Users\\tinys\PycharmProjects\Medical_Engineering_final_Project\src\\acquiredDataset.csv')
    dataset.head()
    print(dataset.head())
    X = dataset.drop('classification', axis=1)
    y = dataset['classification']
    print("Data Balance: " + str(y.value_counts()))
    plot_data_distribution(dataset)
    # Map target variable to numerical values using label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Words in the data need to be changed to numerical values
    numeric_features = selector(dtype_exclude="object")
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_features = selector(dtype_include="object")
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ])

    # Define the classifier
    classifier = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(random_state=42))
    ])

    # Train the model
    classifier.fit(X_train, y_train)

    # Make predictions
    y_pred = classifier.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")

    # Reverse map numerical predictions to original categories for the classification report
    reverse_mapping = {v: k for k, v in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
    y_pred_labels = [reverse_mapping[val] for val in y_pred]
    y_test_labels = [reverse_mapping[val] for val in y_test]

    # Print classification report
    print("Classification Report:\n", classification_report(y_test_labels, y_pred_labels))


def plot_data_distribution(dataset):
    """plots the distribution of the target classes"""
    plt.figure(figsize = (8, 6))
    sns.countplot(x = 'classification', data = dataset)
    plt.title('Distribution of Target Variable')
    plt.xlabel('Obesity Class')
    plt.ylabel('Count')
    plt.show()


if __name__ == "__main__":
    get_data()