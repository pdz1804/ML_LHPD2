# Models Directory

This directory serves as the central repository for all trained machine learning models and their associated artifacts. It contains different versions of trained models, along with supporting data like training logs and performance visualizations.

---

## Directory Structure

```
📂 models/              # Contains all model versions, training data and logs
│── 📂 report_info_ass1/ 
│ │── 📂 img/           # Stores images related to training/validation
│ │── 📂 log/           # Logs generated during the training process
│
│── 📂 report_info_ass2/ 
│ │── 📂 img/           # Stores images related to training/validation
│ │── 📂 log/           # Logs generated during the training process
│
│── 📂 Ver01_25_02_23/  # Version 1 of the models (February 25, 2023)
│ │── 📂 img/           # Stores images related to training/validation
│ │── 📂 trained/       # Stores fully trained models for later use
│ │── 📂 training_log/  # Logs generated during the training process
│ │── 📜 README.md      # Documentation for the Ver02_25_02_23 directory
│
│── 📂 Ver02_25_02_25/  # Version 2 of the models (February 25, 2025)
│ │── 📂 img/           # Stores images related to training/validation
│ │── 📂 other/         # Relevant files
│ │── 📂 trained/       # Stores fully trained models for later use
│ │── 📂 training_log/  # Logs generated during the training process
│ │── 📜 README.md      # Documentation for the Ver02_25_02_25 directory
│
│── 📜 README.md        # Documentation for the models directory
```

---

## Purpose

This directory helps organize and version control your trained models, making it easier to:

*   Track different versions of your models as you experiment and improve them.
*   Reproduce past results by accessing the corresponding trained models, logs, and visualizations.
*   Understand the evolution of your models over time.
*   Share your trained models and their associated data with others.

---

## Key Components

*   **Report Subdirectories (`report_info_ass1/`, `report_info_ass2/`:** Store visual artifacts and logs related to reporting, validation, or interpretability for each assignment.
*   **`img/` (Inside Reports and Model Versions):** Contains visualizations such as loss/accuracy curves, heatmaps, or attention maps.
*   **`log/` (Inside Reports):** Includes any runtime or inference logs useful for interpreting results or reproducing report data.
*   **Versioned Subdirectories (e.g., `Ver01_25_02_23`, `Ver02_25_02_25`, `Ver03_25_04_13`):** Each contains one iteration of trained models and relevant artifacts.
*   **`trained/` (Within Versioned Subdirectories):** Serialized model files.
*   **`training_log/` (Within Versioned Subdirectories):** Captures hyperparameters, training metrics, and any error output.
*   **`other/` (Optional in some versions):** Supporting files like preprocessing scripts, dataset splits, or evaluation notebooks.
*   **`README.md` (Within Each Directory):** Explains that version’s purpose, structure, and usage instructions.


---

## Versioning Strategy

The directory structure uses a simple versioning scheme based on dates.  Each time you significantly retrain or modify your models, create a new versioned subdirectory (e.g., `Ver03_YY_MM_DD`) to store the updated models and data.  This ensures that you can always access and reproduce past results. The `other/` directory should be used to store all supporting files such as preprocessing scripts, train/test splits, and anything else that is not the training logs, image results, or the serialized trained model.

---

## How to Use

1.  **Browse the Versioned Subdirectories:**  Explore the different versioned subdirectories to find the specific model version you are interested in.
2.  **Consult the `README.md`:**  Read the `README.md` file within the versioned subdirectory to understand the contents of that directory and how to use the trained models.
3.  **Load the Trained Models:**  Load the serialized model files from the `trained/` directory using the appropriate framework-specific methods (e.g., `joblib.load()` for scikit-learn, `tf.keras.models.load_model()` for TensorFlow).
4.  **Refer to the Training Logs and Images:**  Use the training logs in `training_log/` and the images in `img/` to understand the model's training process and performance characteristics.
5.  **Utilize the `other/` directory:** For versions that contain an `other/` directory, consult the relevant files to understand more about that specific model version.

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