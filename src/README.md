# Introduction

The `src/` directory contains all the source code required for data processing, feature engineering, model training, and visualization. Each subdirectory is organized to facilitate a structured and maintainable workflow, promoting collaboration and ease of understanding.

---

# Directory Structure

```
📂 src/                             # Main source code directory
│── 📂 data/                        # Scripts 
│ │── 📜 init.py                    # Initializes the data module
│ │── 📜 make_dataset.py            # Script to create and load the dataset
│ │── 📜 preprocess.py              # Script for data preprocessing steps
│ │── 📒 process_v0.ipynb           # Initial data processing notebook
│ │── 📒 process_v1.ipynb           # Refined data processing notebook
│ │
│── 📂 features/                    # Feature engineering and selection scripts
│ │── 📜 init.py                    # Initializes the features module
│ │── 📜 build_features_utils.py    # Utility functions for feature engineering
│ │── 📒 example.ipynb              # Example notebook for feature engineering
│ │
│── 📂 models/                      # Model training, evaluation, and inference scripts
│ │── 📜 init.py                    # Initializes the models module
│ │── 📜 models_utils.py            # Utility functions for model training and evaluation
│ │── 📒 train_test_model_v0.ipynb  # Initial model training and testing notebook
│ │── 📒 train_test_model_v1.ipynb  # Refined model training and testing notebook
│ │
│── 📂 visualization/               # Scripts for generating plots and visual analysis
│ │── 📜 init.py                    # Initializes the visualization module
│ │── 📒 visualize_v0.ipynb         # Initial data visualization notebook
│ │── 📒 visualize_v1.ipynb         # Refined data visualization notebook
│ │── 📜 visualize.py               # Script to generate common visualizations
│ │
│── 📜 README.md                    # Documentation for the src/ directory
```

**Explanation**:

- **`data/`**: Contains scripts to load datasets, clean data, and prepare it for modeling.
    - `__init__.py`:  An empty file that indicates that the `data` directory should be treated as a Python package.
    - `make_dataset.py`: This script is responsible for creating and loading the dataset. It may include functions to download the data from a source or read it from a local file.
    - `preprocess.py`: This script contains functions for data preprocessing steps, such as cleaning, transforming, and preparing the data for modeling.
    - `process_v0.ipynb` and `process_v1.ipynb`: Jupyter Notebooks that likely contain exploratory data analysis (EDA) and initial data processing steps. The 'v0' and 'v1' likely indicate different versions or iterations of the processing workflow.
- **`features/`**: Includes feature engineering scripts, such as feature selection and transformation.
    - `__init__.py`:  An empty file that indicates that the `features` directory should be treated as a Python package.
    - `build_features_utils.py`: This script contains utility functions to help build new features from the existing data.
    - `example.ipynb`: A Jupyter Notebook that provides an example or demonstration of how to use the feature engineering utilities.
- **`models/`**: Houses scripts for training, testing, and deploying machine learning models.
    - `__init__.py`:  An empty file that indicates that the `models` directory should be treated as a Python package.
    - `models_utils.py`: This script contains utility functions for model training, evaluation, and saving/loading models.
    - `train_test_model_v0.ipynb` and `train_test_model_v1.ipynb`: Jupyter Notebooks that implement the model training and testing pipeline.  The different versions likely represent improvements or variations in the modeling approach.
- **`visualization/`**: Provides tools for visualizing data distributions, model performance, and results.
    - `__init__.py`:  An empty file that indicates that the `visualization` directory should be treated as a Python package.
    - `visualize_v0.ipynb` and `visualize_v1.ipynb`: Jupyter Notebooks used for creating visualizations of the data and model results. The versions indicate iterations or different approaches to visualization.
    - `visualize.py`: A Python script containing functions to generate common visualizations, which can be reused throughout the project.
- **`README.md`**: Documents the purpose and structure of this directory.

This structured organization ensures that the codebase is modular, scalable, and easy to maintain.

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