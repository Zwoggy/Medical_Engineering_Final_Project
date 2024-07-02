"""
Python project: Medical engineering final project;


@author: Florian Zwicker
Foe project describtion read readme.md

"""
from PIL._imaging import display
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from keras.models import Sequential
from keras.layers import Dense
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector as selector

import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline, make_pipeline

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt


def get_data():
    """
    This function loads the dataset from a specified path, performs data preprocessing (including encoding categorical variables, scaling numerical variables, handling missing values), visualizes the correlation heatmap and feature relationship, splits the dataset into training and testing data, and applies different classifiers after resampling.

    It first imports data from a CSV file, then calls functions to create a heatmap and show feature relations.

    The function maps the target variable to numerical values, splits the data, and applies a preprocessing transformation.

    Then, it applies multiple classifiers (including GridSearchCV, GradientBoostingClassifier, LogisticRegression, RandomForestClassifier, SVC, KNeighborsClassifier), and for each classifier resampling is applied if specified.

    Model training is performed, followed by making predictions and evaluating the model. The evaluation metrics include accuracy and the detailed classification report.

    Note: The classifier and resampling methods are predefined in the code and could be customized according to specific requirements.

    The function returns none but it prints out the output of the intermediate steps as well as the model performance measures.

    Parameters:
    None

    Returns:
    None
    """
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

    pca = PCA(n_components = 2)
    logistic_regression = LogisticRegression(max_iter = 1000)
    X = dataset.drop('classification', axis = 1)
    y = dataset['classification']
    param_grid = {
        'penalty': ['l1', 'l2'],  # Regularization penalty
        'C': [0.001, 0.01, 0.1, 1, 10],  # Inverse of regularization strength
        'solver': ['liblinear', 'saga']
    }
    classifiers = [
        ("GradientBoostingClassifier", GradientBoostingClassifier(random_state = 42)),
        ("GridSearchCV", GridSearchCV(LogisticRegression(random_state = 42, max_iter = 1000), param_grid, cv = 5, scoring = 'accuracy', n_jobs = -1)),
        ("LogisticRegression", LogisticRegression(random_state = 42)),
        ("RandomForestClassifier", RandomForestClassifier(random_state = 42)),
        ("SVC", SVC(random_state = 42, degree=3, kernel='rbf')), # SVM classifier
        ("knn", KNeighborsClassifier(n_neighbors = 3))
    ]
    """resampling for comparison if required"""
    resampling = {
        'None': None,

        #'OverSampling': SMOTE(sampling_strategy = 'minority'),
        #'UnderSampling': RandomUnderSampler(sampling_strategy = 'majority')

    }


    for classifier_name, classifier in classifiers:
        for resample_name, resampler in resampling.items():
            if resampler is None:
                model = Pipeline(steps = [
                    ("preprocessor", preprocessor),
                    #("pca", pca),
                    ("classifier", classifier)
                ])
            else:
                model = Pipeline(steps = [
                    ("preprocessor", preprocessor),
                    ("resampling", resampler),
                    ("pca", pca),
                    ("classifier", classifier)
                ])

            # Train the model
            model.fit(X_train, y_train)

            # Make predictions
            y_pred = model.predict(X_test)

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy:.2f}; " + str(classifier_name))

            # Reverse map numerical predictions to original categories for the classification report
            reverse_mapping = {v: k for k, v in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
            y_pred_labels = [reverse_mapping[val] for val in y_pred]
            y_test_labels = [reverse_mapping[val] for val in y_test]

            # Print classification report
            print("Classification Report:\n", classification_report(y_test_labels, y_pred_labels))
    train_dnn(X_train, y_train, X_test, y_test)

def plot_data_distribution(dataset):
    """plots the distribution of the target classes"""
    plt.figure(figsize = (8, 6))
    sns.countplot(x = 'classification', data = dataset)
    plt.title('Distribution of Target Variable')
    plt.xlabel('Drowsiness')
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
    """
    Creates and displays a heatmap for the correlation matrix of the given data.

    Parameters:
    data : DataFrame
        The data used to generate the correlation matrix and heatmap.

    Returns:
    None
    """
    plt.figure(figsize = (20, 8))
    sns.heatmap(data.corr(), cmap = "YlGnBu", annot = True)


def show_feature_relation(data):
    """
    Visualizes pairwise relationships in a dataset.
    For each pair of features, filters the data to include only data points
    that are above the 70th percentile for both features. Then generates a pair plot
    from these data points, separated by classification.

    Parameters:
    data : DataFrame
        The data used to generate the pair plot.
        It is assumed to contain 'classification' column, and all other columns
        are considered numeric features.

    Returns:
    None
    """
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
    """
    Trains and evaluates a Logistic Regression classifier on the provided dataset.
    Optimizes the hyperparameters of the model using grid search with 5-fold cross-validation.
    The function will print the optimal hyperparameters, the best cross-validation accuracy, and the test set accuracy.
    The classification target column in the dataset should be named 'classification'.

    Parameters:
    dataset : DataFrame
        The dataset for training, validation and test.
        Should contain a 'classification' column which will be used as the target variable.

    Returns:
    None
    """
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


def train_dnn(X_train, y_train, X_test, y_test):
    """
    Trains a Deep Neural Network model on provided training data and evaluates it on test data.

    Parameters:
    X_train, X_test: DataFrame
        Training and test feature data. The number of columns should be the same.
    y_train, y_test: Series or array-like
        Training and test target data. The length should correspond to the number of rows in X_train and X_test.

    Returns:
    history: History
        Output of the Keras fit method. Contains details about the training process, including training and validation loss and accuracy at each epoch.

    Note:
    - This function assumes that binary cross-entropy loss is suitable for the given problem, which is only true for binary classification tasks. It may not be suitable for regression or multi-class classification.
    - The DNN model architecture is defined within this function, and is fixed to two hidden layers with 32 and 16 nodes respectively. Depending upon the complexity of the dataset, you may need to adjust this architecture.
    """

    dnn_model = Sequential()
    dnn_model.add(Dense(8, input_dim = len(X_train.columns), activation = 'relu'))  # Input layer
    dnn_model.add(Dense(4, activation = 'relu'))  # Hidden layer
    dnn_model.add(Dense(1, activation = 'sigmoid'))  # Output layer for binary classification

    # Compile the model
    dnn_model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    history = dnn_model.fit(X_train, y_train, epochs = 50, batch_size = 30, validation_data = (X_test, y_test))

    return history


if __name__ == "__main__":
    get_data()