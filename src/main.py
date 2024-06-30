"""
Python project: Medical engineering final project;


@author: Florian Zwicker

"""
from PIL._imaging import display
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector as selector



import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt


def get_data():
    # Change the Path in the line below according to your own path
    dataset = pd.read_csv('G:\\Users\\tinys\\PycharmProjects\\Medical_Engineering_final_Project\\src\\acquiredDataset.csv')
    dataset.head()
    print(dataset.head())
    create_heatmap(dataset) # show heatmap to visualize correlation
    show_feature_relation(dataset) # show relation between the features
    #do_logistic_regression(dataset) # logistic_regression model
    X = dataset.drop('classification', axis=1)
    y = dataset['classification']
    print("Data Balance: " + str(y.value_counts()))
    plot_data_distribution(dataset)
    # Map target variable to numerical values using label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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


def get_metadata(dataframe):
    '''
    A function that fetches all the Metadata Information about the Dataframe
    This function can be reused for all Pandas Dataframe
    '''
    print("\nBASIC INFORMATION\n")
    print(dataframe.info())
    print("=" * 100)
    print("STATISTICAL INFORMATION")
    display(dataframe.describe(include='all'))
    print("=" * 100)
    print("Dataframe Shape\n", dataframe.shape)
    print("=" * 100)
    print("Number of Duplicate Rows\n", dataframe.duplicated().sum())
    print("=" * 100)
    print("NULL Values Check")
    print(dataframe.isnull().sum())
    print("=" * 100)

def create_heatmap(data):
    plt.figure(figsize = (20, 8))
    sns.heatmap(data.corr(), cmap = "YlGnBu", annot = True)


def show_feature_relation(data):
    # Calculate the 70th percentile for each wave frequency band
    delta_70th = data['delta'].quantile(0.7)
    theta_70th = data['theta'].quantile(0.7)
    lowAlpha_70th = data['lowAlpha'].quantile(0.7)
    highAlpha_70th = data['highAlpha'].quantile(0.7)
    lowBeta_70th = data['lowBeta'].quantile(0.7)
    highBeta_70th = data['highBeta'].quantile(0.7)
    lowGamma_70th = data['lowGamma'].quantile(0.7)
    highGamma_70th = data['highGamma'].quantile(0.7)

    # Filter the data to include only data points above the 70th percentile for each band
    filtered_data = data[(data['delta'] >= delta_70th) &
                         (data['theta'] >= theta_70th) &
                         (data['lowAlpha'] >= lowAlpha_70th) &
                         (data['highAlpha'] >= highAlpha_70th) &
                         (data['lowBeta'] >= lowBeta_70th) &
                         (data['highBeta'] >= highBeta_70th) &
                         (data['lowGamma'] >= lowGamma_70th) &
                         (data['highGamma'] >= highGamma_70th)]

    # Create the pair plot with the filtered data
    sns.pairplot(filtered_data, hue = 'classification',
                 vars = ['delta', 'theta', 'lowAlpha', 'highAlpha',
                         'lowBeta', 'highBeta', 'lowGamma', 'highGamma'],
                 markers = ["o", "s"])

    # Add a legend
    plt.legend(loc = 'upper right', labels = ['Sleep', 'Awake'])

    # Set plot labels and title
    plt.xlabel('Brain Waves')
    plt.ylabel('Brain Waves')
    plt.suptitle('Pair Plot of Brain Waves vs. Awake State (70th percentile)')

    # Show the plot
    plt.show()


def do_logistic_regression(dataset):
    # Create a Logistic Regression model
    logistic_regression = LogisticRegression(max_iter = 1000)
    X = dataset.drop('classification', axis = 1)
    y = dataset['classification']
    param_grid = {
        'penalty': ['l1', 'l2'],  # Regularization penalty
        'C': [0.001, 0.01, 0.1, 1, 10],  # Inverse of regularization strength
        'solver': ['liblinear', 'saga']  # Optimization algorithm
    }
    # Create a GridSearchCV object
    grid_search = GridSearchCV(logistic_regression, param_grid, cv = 5, scoring = 'accuracy', n_jobs = -1)

    # Perform the grid search to find the best hyperparameters
    grid_search.fit(X, y)

    # Print the best hyperparameters and corresponding accuracy
    print("Best Hyperparameters: ", grid_search.best_params_)
    print("Best Accuracy: {:.2f}%".format(grid_search.best_score_ * 100))

    # Evaluate the model on the test set using the best hyperparameters
    best_logistic_regression = grid_search.best_estimator_
    test_accuracy = best_logistic_regression.score(X, y)
    print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))


if __name__ == "__main__":
    get_data()