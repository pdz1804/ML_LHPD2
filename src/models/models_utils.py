"""
build_features.py

Author: Nguyen Quang Phu
Date: 2025-02-03
Updated: 2025-02-10
"""

import os
import joblib
import pandas as pd
import numpy as np
from tqdm import tqdm
from statistics import mean
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import hmmlearn.hmm
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn_crfsuite import CRF
from sklearn.metrics import log_loss, hinge_loss

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

def build_cnn_model(hp, input_shape):
    model = keras.Sequential()
    
    # CNN Layer 1
    model.add(layers.Conv1D(
        filters=hp.Int('filters_1', min_value=32, max_value=128, step=32),
        kernel_size=hp.Choice('kernel_size_1', values=[3, 5]),
        activation="relu",
        input_shape=input_shape
    ))
    model.add(layers.MaxPooling1D(pool_size=2))
    
    # CNN Layer 2
    model.add(layers.Conv1D(
        filters=hp.Int('filters_2', min_value=64, max_value=256, step=64),
        kernel_size=hp.Choice('kernel_size_2', values=[3, 5]),
        activation="relu"
    ))
    model.add(layers.MaxPooling1D(pool_size=2))
    
    model.add(layers.Flatten())

    # Dense Layer
    model.add(layers.Dense(
        units=hp.Int('dense_units', min_value=64, max_value=256, step=64),
        activation="relu"
    ))
    
    # Dropout
    model.add(layers.Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    
    # Output Layer
    model.add(layers.Dense(1, activation="sigmoid"))
    
    # Compile Model
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-3, 1e-4])),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model

def build_lstm_model(hp, input_shape):
    model = keras.Sequential()
    
    # First LSTM Layer
    model.add(layers.LSTM(
        units=hp.Int('lstm_units_1', min_value=64, max_value=256, step=64),
        return_sequences=True,
        input_shape=input_shape
    ))
    
    # Second LSTM Layer
    model.add(layers.LSTM(
        units=hp.Int('lstm_units_2', min_value=32, max_value=128, step=32),
        return_sequences=False
    ))
    
    # Dense Layer
    model.add(layers.Dense(
        units=hp.Int('dense_units', min_value=64, max_value=256, step=64),
        activation="relu"
    ))
    
    # Dropout
    model.add(layers.Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    
    # Output Layer
    model.add(layers.Dense(1, activation="sigmoid"))
    
    # Compile Model
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-3, 1e-4])),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    
    return model


# Creating a reusable function for churning through all five binary classification algorithms
def generate_binary_classification_model(X, y, model_algorithm, hyperparameters, needs_scaled = False, model_save_path="best_model.pkl", img_save_path=None, img_loss_path=None):
    """
    Generating everything required for training and validation of a binary classification model

    Args:
        - X (Pandas DataFrame): Training features
        - y (Pandas DataFrame): Target values
        - model_algorithm (object): Model algorithm to train
        - hyperparameters (dict): Hyperparameters for tuning
        - needs_scaled (Boolean): Whether to scale the dataset
        - model_save_path (str): Path to save the best model
        - img_save_path (str): Path to save validation performance plot
        - img_loss_path (str): Path to save training loss plot
    """
    # Check if the model already exists
    if os.path.exists(model_save_path):
        print(f"🔄 Loading existing model from {model_save_path}...")
        model_algorithm = joblib.load(model_save_path)
        return model_algorithm
    
    print(f"🚀 Training new model: {model_algorithm.__class__.__name__}...")
    # Performing a scaling on the data if required
    if needs_scaled:
        
        # Instantiating the StandardScaler
        scaler = StandardScaler()
        
        # Performing a fit_transform on the dataset
        scaled_features = scaler.fit_transform(X)
        
        # Transforming the StandardScaler output back into a Pandas DataFrame
        X = pd.DataFrame(scaled_features, index = X.index, columns = X.columns)
        
    # Instantiating a GridSearch object with the inputted model algorithm and hyperparameters
    gridsearchcv = GridSearchCV(estimator = model_algorithm,
                                param_grid = hyperparameters)
    
    # Fitting the training data to the GridSearch object
    gridsearchcv.fit(X, y)
    
    # Printing out the best hyperparameters
    print(f'Best hyperparameters: {gridsearchcv.best_params_}')
    
    # Instantiating a new model object with the ideal hyperparameters from the GridSearch job
    model_algorithm.set_params(**gridsearchcv.best_params_)
    
    # Creating a container to hold each set of validation metrics
    accuracy_scores, roc_auc_scores, f1_scores, precision_scores, recall_scores = [], [], [], [], []
    training_losses, validation_losses = [], []
    
    # Instantiating the K-Fold cross validation object
    k_fold = KFold(n_splits = 5)
    
    print("\n🎯 Running K-Fold Cross-Validation...")
    for train_index, val_index in tqdm(k_fold.split(X), total=k_fold.get_n_splits(), desc="K-Fold Progress"):

        # Splitting the training set from the validation set for this specific fold
        X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Fitting the X_train and y_train datasets to the model algorithm
        model_algorithm.fit(X_train, y_train)
        
        # Compute losses
        train_loss = get_training_loss(model_algorithm, X_train, y_train)
        val_loss = get_training_loss(model_algorithm, X_val, y_val)

        training_losses.append(train_loss)
        validation_losses.append(val_loss)

        # Getting inferential predictions for the validation dataset
        val_preds = model_algorithm.predict(X_val)

        # Generating validation metrics by comparing the inferential predictions (val_preds) to the actuals (y_val)
        val_accuracy = accuracy_score(y_val, val_preds)
        val_roc_auc_score = roc_auc_score(y_val, val_preds)
        val_f1_score = f1_score(y_val, val_preds)
        val_precision_score = precision_score(y_val, val_preds)
        val_recall_score = recall_score(y_val, val_preds)
        
        # Appending the validation scores to the respective validation metric container
        accuracy_scores.append(val_accuracy)
        roc_auc_scores.append(val_roc_auc_score)
        f1_scores.append(val_f1_score)
        precision_scores.append(val_precision_score)
        recall_scores.append(val_recall_score)
        
    # Print average validation scores
    print(f'📊 Average Accuracy: {int(mean(accuracy_scores) * 100)}%')
    print(f'📊 Average ROC AUC: {int(mean(roc_auc_scores) * 100)}%')
    print(f'📊 Average F1 Score: {int(mean(f1_scores) * 100)}%')
    print(f'📊 Average Precision: {int(mean(precision_scores) * 100)}%')
    print(f'📊 Average Recall: {int(mean(recall_scores) * 100)}%')

    # Save the trained model
    joblib.dump(model_algorithm, model_save_path)
    print(f'💾 Model saved to {model_save_path}')
    
    # If img_save_path is provided, plot the validation scores
    if img_save_path:
        plt.figure(figsize=(10, 6))

        # Plot Accuracy
        plt.plot(range(1, len(accuracy_scores) + 1), accuracy_scores, label="Accuracy", marker='o')

        # Plot ROC AUC
        plt.plot(range(1, len(roc_auc_scores) + 1), roc_auc_scores, label="ROC AUC", marker='o')

        # Add labels and title
        plt.title("Validation Performance Across K-Folds")
        plt.xlabel("Fold Number")
        plt.ylabel("Score")
        plt.legend()

        # Save the plot to the specified path
        plt.savefig(img_save_path)
        plt.close()
        print(f"📈 Plot saved to {img_save_path}")
        
    # Plot loss curves
    if img_loss_path:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(training_losses) + 1), training_losses, label="Training Loss", marker='o')
        plt.plot(range(1, len(validation_losses) + 1), validation_losses, label="Validation Loss", marker='o')
        plt.title("Training & Validation Loss Across K-Folds")
        plt.xlabel("Fold Number")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(img_loss_path)
        plt.close()
        print(f"📉 Loss plot saved to {img_loss_path}")
    
    # ---- Additional Decision Tree Analysis ---- #
    # if isinstance(model_algorithm, DecisionTreeClassifier):
    #     print("\n🌳 Performing Decision Tree Analysis...")

    #     # Plot the decision tree
    #     plt.figure(figsize=(20, 10))
    #     plot_tree(model_algorithm, filled=True, class_names=[str(label) for label in model_algorithm.classes_], rounded=True)
    #     plt.title("Decision Tree Visualization")
    #     plt.show()

    #     # Print decision path for a sample
    #     sample_id = 0  # Change for other samples if needed
    #     node_indicator = model_algorithm.decision_path(X)
    #     leaf_id = model_algorithm.apply(X)

    #     # print(f"\n📝 Rules used to predict sample {sample_id}: {X_train[sample_id]}")
    #     print(f"\n📝 Rules used to predict sample {sample_id}:")
    #     node_index = node_indicator.indices[node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]]

    #     for node_id in node_index:
    #         continue

    #     threshold_sign = "<=" if X.iloc[sample_id, model_algorithm.tree_.feature[node_id]] <= model_algorithm.tree_.threshold[node_id] else ">"
    #     print(f"🔹 Decision node {node_id}: (X[{sample_id}, {model_algorithm.tree_.feature[node_id]}] = "
    #             f"{X.iloc[sample_id, model_algorithm.tree_.feature[node_id]]}) {threshold_sign} {model_algorithm.tree_.threshold[node_id]}")

    return model_algorithm



def get_training_loss(model, X_train, y_train):
    """
    Compute training loss based on model type.
    """
    # Models that expose their loss during training
    if hasattr(model, "best_score_"):  # XGBoost
        return -model.best_score_

    if hasattr(model, "loss_"):  # Perceptron (Hinge loss)
        return model.loss_

    # Probabilistic models (e.g., HMM, Naive Bayes)
    if hasattr(model, "score"):  
        return -model.score(X_train, y_train)  # Negative log-likelihood

    # Support Vector Machines (hinge loss)
    if isinstance(model, SVC):
        y_pred = model.decision_function(X_train)
        return np.mean(np.maximum(0, 1 - y_train * y_pred))  # Hinge loss

    # Logistic Regression (log loss)
    if isinstance(model, LogisticRegression):
        y_proba = model.predict_proba(X_train)
        return log_loss(y_train, y_proba)

    # Decision Tree, Random Forest: No direct loss, use log loss
    if isinstance(model, (DecisionTreeClassifier, RandomForestClassifier)):
        y_proba = model.predict_proba(X_train)
        return log_loss(y_train, y_proba)

    return None  # Loss not available

# Example usage:
# Setting the hyperparameter grid for the Logistic Regression algorithm
# logistic_reg_params = {
#     'penalty': ['l1', 'l2'],
#     'C': np.logspace(-4, 4, 20),
#     'solver': ['lbfgs', 'liblinear']
# }

# logistic_reg_algorithm = LogisticRegression()

# generate_binary_classification_model(X = X,
#                                      y = y,
#                                      model_algorithm = logistic_reg_algorithm,
#                                      hyperparameters = logistic_reg_params)
