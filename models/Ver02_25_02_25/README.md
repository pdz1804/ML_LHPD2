# Model Version: Ver02_25_02_25

This directory contains the artifacts for the machine learning models trained on February 25, 2025 (Version 2). It includes trained models, performance visualizations, training logs, and additional supporting files.

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

This directory is organized to maintain a clear separation of model-related assets.  Each subdirectory within `Ver02_25_02_25` corresponds to a specific stage or aspect of the model development process.  This version introduces the `other/` directory for additional supporting files.


### `img/` - Performance Visualizations

This directory contains visualizations illustrating the performance of the different models during training and validation. Expect to find:

*   **Loss Curves:** Plots of training and validation loss over epochs. Useful for identifying overfitting or underfitting.
*   **Accuracy Metrics:** Graphs showing training and validation accuracy.
*   **K-Fold Cross-Validation Results:** Box plots or similar visualizations showing the performance of each model across different folds of cross-validation. This helps assess the generalization ability of the model.

Subdirectories are named to match the model type they represent (e.g., `results_DT_LR` for Decision Tree + Logistic Regression). Images are typically in `.png` or `.jpg` format.

### `other/` - Supporting Files

This directory contains supplementary files that are relevant to this specific model version. This includes another type of Bayesian Network Training strategies from Loc.


### `trained/` - Serialized Trained Models

This directory holds the serialized, fully trained machine learning models. These models are ready to be loaded and used for making predictions on new data.

*   **File Format:** Models are saved in [Specify the file format here. E.g., `joblib` format (`.joblib` files) for scikit-learn models].
*   **Loading Models:** To load a model, use the appropriate function from your machine learning library [Specify loading instructions here. E.g., "Use `joblib.load('path/to/model.joblib')` to load a model."].

Subdirectories mirror the `img/` directory structure, ensuring that each trained model is easily associated with its performance visualizations.

### `training_log/` - Training Logs

This directory contains log files generated during the model training process. These logs provide a detailed record of the training process, including:

*   **Hyperparameters:** The specific hyperparameter values used for each model.
*   **Metrics:** Performance metrics (loss, accuracy, etc.) recorded during training.
*   **Training Time:** The time taken to train each model.
*   **Warnings/Errors:** Any warnings or errors encountered during training.

These logs can be invaluable for debugging, understanding model behavior, and reproducing results.

---

## Changes from Ver01_25_02_23

This version includes the following changes compared to `Ver01_25_02_23`:

*  In this new version, we have used the new preprocessing data for training
*  We have added new trained models such as GA, Bayesian Network, HMM and LSTM.

---

## Usage

1.  **Select a Model:** Determine the model type that best suits your needs.
2.  **Load the Trained Model:** Use the appropriate loading function (as described above) to load the serialized model file from the `trained/` directory. *Example:* `model = joblib.load('trained/results_RF/my_random_forest_model.joblib')`
3.  **Preprocess Your Data:** Ensure your input data is preprocessed in the same way as the training data. *Crucially, use the preprocessing scripts located in the `other/` directory to ensure consistency.* Consult the training logs in `training_log/` for further details on preprocessing steps.
4.  **Make Predictions:** Use the loaded model to make predictions on your data.
5.  **Evaluate Performance:** Refer to the visualizations in `img/` and the metrics in `training_log/` to assess the model's expected performance.

---

## 📧 **Contact**
For any questions or contributions, please contact:

📩 Email: phu.nguyenquang2004@hcmut.edu.vn

🔗 GitHub: https://github.com/pdz1804/

---

<h2 align="center">✨💟 Contributors 💟✨</h2>

<p align="center">
  💖 <strong>We fairly contribute to this repository with dedication and teamwork!</strong> 💖
</p>

<div align="center">
  <a href="https://github.com/pdz1804"><img src="https://avatars.githubusercontent.com/u/123137268?v=4" title="pdz1804" width="50" height="50"></a>
  <a href="https://github.com/MarkX04"><img src="https://avatars.githubusercontent.com/u/105540317?v=4" title="MarkX04" width="50" height="50"></a>
  <a href="https://github.com/DatNguyen1402"><img src="https://avatars.githubusercontent.com/u/137872945?v=4" title="DatNguyen1402" width="50" height="50"></a>
  <a href="https://github.com/hungyle123"><img src="https://avatars.githubusercontent.com/u/138371452?v=4" title="hungyle123" width="50" height="50"></a>
  <a href="https://github.com/nguyen1oc"><img src="https://avatars.githubusercontent.com/u/131537455?v=4" title="nguyen1oc" width="50" height="50"></a>
</div>

--- 