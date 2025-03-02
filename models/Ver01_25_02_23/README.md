# Model Version: Ver01_25_02_23

This directory contains the artifacts for the machine learning models trained on February 25, 2023 (Version 1). It includes trained models, performance visualizations, and training logs.

---

## Directory Structure

```
📂 models/                     # Contains training results, trained models, and logs
│── 📂 img/                    # Stores images related to training/validation loss, accuracy, and k-fold cross-validation results
│   │── 📂 results_DT_LR/      # Images for Decision Tree + Logistic Regression results
│   │── 📂 results_per_bayes/  # Images for Perceptron + Naive Bayes results
│   │── 📂 results_RF/         # Images for Random Forest results
│   │── 📂 results_svm/        # Images for SVM results
│   │── 📂 ...       
│
│── 📂 trained/                # Stores fully trained models for later use
│   │── 📂 results_DT_LR/      # Trained models for Decision Tree + Logistic Regression
│   │── 📂 results_per_bayes/  # Trained models for Perceptron + Naive Bayes results
│   │── 📂 results_RF/         # Trained models for Random Forest
│   │── 📂 results_svm/        # Trained models for SVM
│   │── 📂 ...
│
│── 📂 training_log/           # Logs generated during the training process
│   │── 📜 app_DT_LR.log       # Training log for Decision Tree + Logistic Regression
│   │── 📜 app_per_bayes.log   # Training log for Perceptron + Naive Bayes results
│   │── 📜 app_RF.log          # Training log for Random Forest
│   │── 📜 app_svm.log         # Training log for SVM
│   │── 📜 ...
│
│── 📜 README.md               # Documentation for models directory
```

---

## Purpose and Contents

This directory is organized to maintain a clear separation of model-related assets.  Each subdirectory within `Ver01_25_02_23` corresponds to a specific stage or aspect of the model development process.

### `img/` - Performance Visualizations

This directory contains visualizations illustrating the performance of the different models during training and validation.  Expect to find:

*   **Loss Curves:**  Plots of training and validation loss over epochs.  Useful for identifying overfitting or underfitting.
*   **Accuracy Metrics:**  Graphs showing training and validation accuracy.
*   **K-Fold Cross-Validation Results:**  Box plots or similar visualizations showing the performance of each model across different folds of cross-validation. This helps assess the generalization ability of the model.

Subdirectories are named to match the model type they represent (e.g., `results_DT_LR` for Decision Tree + Logistic Regression).  Images are typically in `.png` or `.jpg` format.

### `trained/` - Serialized Trained Models

This directory holds the serialized, fully trained machine learning models.  These models are ready to be loaded and used for making predictions on new data.

Subdirectories mirror the `img/` directory structure, ensuring that each trained model is easily associated with its performance visualizations.

### `training_log/` - Training Logs

This directory contains log files generated during the model training process.  These logs provide a detailed record of the training process, including:

*   **Hyperparameters:** The specific hyperparameter values used for each model.
*   **Epoch-Level Metrics:**  Performance metrics (loss, accuracy, etc.) recorded for each epoch of training.
*   **Training Time:**  The time taken to train each model.
*   **Warnings/Errors:**  Any warnings or errors encountered during training.

These logs can be invaluable for debugging, understanding model behavior, and reproducing results.

---

## Usage

1.  **Select a Model:**  Determine the model type that best suits your needs.
2.  **Load the Trained Model:**  Use the appropriate loading function (as described above) to load the serialized model file from the `trained/` directory.  *Example:* `model = joblib.load('trained/results_RF/my_random_forest_model.joblib')`
3.  **Preprocess Your Data:** Ensure your input data is preprocessed in the same way as the training data.  Consult the training logs in `training_log/` for details on preprocessing steps.
4.  **Make Predictions:**  Use the loaded model to make predictions on your data.
5.  **Evaluate Performance:**  Refer to the visualizations in `img/` and the metrics in `training_log/` to assess the model's expected performance.

---

## Notes

*   This directory represents a specific version of the trained models (`Ver01_25_02_23`).  Subsequent versions may contain different models, architectures, or training data.
*   Always consult the `training_log/` for detailed information about the training process and model configuration.
*   Ensure that you have the necessary libraries installed (e.g., scikit-learn, joblib) to load and use the trained models.

---
