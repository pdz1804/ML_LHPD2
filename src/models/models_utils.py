import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from statistics import mean
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Creating a reusable function for churning through all five binary classification algorithms
def generate_binary_classification_model(X, y, model_algorithm, hyperparameters, needs_scaled = False, model_save_path="best_model.pkl", img_save_path=None):
    """
    Generating everything required for training and validation of a binary classification model

    Args:
        - X (Pandas DataFrame): A DataFrame containing the cleaned training data
        - y (Pandas DataFrame): A DataFrame containing the target values correlated to the X training data
        - model_algorithm (object): A model algorithm that will be trained against the X and y data
        - hyperparameters (dict): A dictionary containing all the hyperparameters to test the model with
        - needs_scaled (Boolean): A boolean value that indicates whether or not the input dataset
        - model_save_path (str): Path to save the best model
        - img_save_path (str): Path to save the validation performance plot (optional)
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
    accuracy_scores, roc_auc_scores, f1_scores = [], [], []
    
    # Instantiating the K-Fold cross validation object
    k_fold = KFold(n_splits = 5)
    
    print("\n🎯 Running K-Fold Cross-Validation...")
    for train_index, val_index in tqdm(k_fold.split(X), total=k_fold.get_n_splits(), desc="K-Fold Progress"):

        # Splitting the training set from the validation set for this specific fold
        X_train, X_val = X.iloc[train_index, :], X.iloc[val_index, :]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        # Fitting the X_train and y_train datasets to the model algorithm
        model_algorithm.fit(X_train, y_train)

        # Getting inferential predictions for the validation dataset
        val_preds = model_algorithm.predict(X_val)

        # Generating validation metrics by comparing the inferential predictions (val_preds) to the actuals (y_val)
        val_accuracy = accuracy_score(y_val, val_preds)
        val_roc_auc_score = roc_auc_score(y_val, val_preds)
        val_f1_score = f1_score(y_val, val_preds)
        
        # Appending the validation scores to the respective validation metric container
        accuracy_scores.append(val_accuracy)
        roc_auc_scores.append(val_roc_auc_score)
        f1_scores.append(val_f1_score)
        
    # Print average validation scores
    print(f'📊 Average Accuracy: {int(mean(accuracy_scores) * 100)}%')
    print(f'📊 Average ROC AUC: {int(mean(roc_auc_scores) * 100)}%')
    print(f'📊 Average F1 Score: {int(mean(f1_scores) * 100)}%')

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
    
    # # ---- Additional Decision Tree Analysis ---- #
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
    #         if leaf_id[sample_id] == node_id:
    #             continue

    #         threshold_sign = "<=" if X.iloc[sample_id, model_algorithm.tree_.feature[node_id]] <= model_algorithm.tree_.threshold[node_id] else ">"
    #         print(f"🔹 Decision node {node_id}: (X[{sample_id}, {model_algorithm.tree_.feature[node_id]}] = "
    #               f"{X.iloc[sample_id, model_algorithm.tree_.feature[node_id]]}) {threshold_sign} {model_algorithm.tree_.threshold[node_id]}")

    return model_algorithm

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
