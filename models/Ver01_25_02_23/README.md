This is the models folder.
The meaning of the structure of the folder is:

```
📂 models/                     # Contains training results, trained models, and logs
│── 📂 img/                    # Stores images related to training/validation loss, accuracy, and k-fold cross-validation results
│   │── 📂 results_DT_LR/      # Images for Decision Tree + Logistic Regression results
│   │── 📂 results_per_bayes/  # Images for Perceptron + Naive Bayes results
│   │── 📂 results_RF/         # Images for Random Forest results
│   │── 📂 results_svm/        # Images for SVM results
│
│── 📂 trained/                # Stores fully trained models for later use
│   │── 📂 results_DT_LR/      # Trained models for Decision Tree + Logistic Regression
│   │── 📂 results_per_bayes/  # Trained models for Perceptron + Naive Bayes results
│   │── 📂 results_RF/         # Trained models for Random Forest
│   │── 📂 results_svm/        # Trained models for SVM
│
│── 📂 training_log/           # Logs generated during the training process
│   │── 📜 app_DT_LR.log       # Training log for Decision Tree + Logistic Regression
│   │── 📜 app_per_bayes.log   # Training log for Perceptron + Naive Bayes results
│   │── 📜 app_RF.log          # Training log for Random Forest
│   │── 📜 app_svm.log         # Training log for SVM
│
│── 📜 README.md               # Documentation for models directory
```