<div align="center">
  <h2>VIETNAM NATIONAL UNIVERSITY, HO CHI MINH CITY</h2>
  <h3>UNIVERSITY OF TECHNOLOGY</h3>
  <h3>FACULTY OF COMPUTER SCIENCE AND ENGINEERING</h3>
  
  <br />
  
  <img src="https://hcmut.edu.vn/img/nhanDienThuongHieu/01_logobachkhoasang.png" alt="logo" style="width: 350px; height: auto;">
  
  <br />
  <br />
</div>

<h2 align="center">💡 Course: Machine Learning - 242 💡</h2>
<h2 align="center">Assignment 2</h2>
<h3 align="center">💡 Class: CC01 - Group LHPD2  💡</h2>

---


## 👥 **Team Members, Team Repository Link and Team Report**

Our repository is at this link: [https://github.com/pdz1804/ML_LHPD2](https://github.com/pdz1804/ML_LHPD2).

This README that we submitted to you is stored in the repository at [ML_LHPD2/notebooks/assignment2/ML_LHPD2_Ass2_README.md](https://github.com/pdz1804/ML_LHPD2/blob/main/notebooks/assignment2/ML_LHPD2_Ass2_README.md).

Also, our team's report which contains detailed information (data analysis, data preprocessing, normalization, models training process and evaluation of those trained models) for this Assignment 1 is stored in the repository at [ML_LHPD2/reports/final_project/](https://github.com/pdz1804/ML_LHPD2/tree/main/reports/final_project/).


| Name | Student ID |
|-------------------------|------------|
| **Nguyen Quang Phu** | 2252621 |
| **Pham Huynh Bao Dai** | 2252139 |
| **Nguyen Thanh Dat** | 2252145 |
| **Nguyen Tien Hung** | 2252280 |
| **Nguyen Thien Loc** | 2252460 |

---

## 📝 **I. Group Project Requirements**

### **1️⃣ Project Organization**
✅ **Group Formation**:
- The group consists of **5 members**.
- **Nguyen Quang Phu** is the **repository owner**.

✅ **Collaboration Workflow**:
- The **main repository** is managed by the designated team member.
- Other members **fork** the repository and contribute via **pull requests**.
- **Roles and responsibilities** are clearly defined for each team member.

---

### **2️⃣ GitHub Repository Structure**
We are implementing **ALL machine learning models from the syllabus** using:
- **ONE unified dataset** with one use-case before the midterm (Chapters 2-6) (which we do in this Assignment - Assignment 1).
- **THAT dataset** for that same use-case for the second stage after midterm (Chapters 6-10) (which we will do in our Assignment 2).

✅ **Repository Setup**:
- The **main repository** is created by the team lead.
- Team members **fork** and contribute via **pull requests**.
- **Branching strategy** is used for different models and features.
- **Comprehensive documentation** is maintained.

📌 **Suggested Repository Structure**:

```
📂 Project-Root 
│── 📂 data/            # Raw & Processed Datasets 
│── 📂 notebooks/       # Jupyter Notebooks for EDA & Preprocessing 
│── 📂 models/          # ML Model Implementations 
│── 📂 other/           # Some other related things here
│── 📂 reports/         # Contain figures and Reports
│── 📂 src/             # Contain src code of how to process data, build features and train the models
│── 📂 tests/           # Evaluation Results & Comparisons 
│── 📜 README.md        # Project Documentation 
│── 📜 requirements.txt # Dependencies
```

---

### **3️⃣ Key Requirements**
✅ Well-documented problem statements & model variations  
✅ Consistent structure for model implementations  
✅ Comprehensive testing of all components  
✅ Comparative analysis of models  
✅ Code reviews and pull requests  
✅ Version control best practices  

---

### **4️⃣ Each Team Member Should:**
- **Actively contribute to the repository** 🚀
- **Document their work thoroughly** 📝
- **Review and improve others' code** 👀
- **Participate in group discussions** 💬
- **Help maintain code quality** ✅

---

## 📊 **II. Completed Tasks**
### **1️⃣ Data Collection & Preprocessing**
We have **collected and preprocessed** the dataset, and it has been uploaded to **Kaggle**.

🔗 **Kaggle Dataset**: [Tweets Clean PosNeg v1](https://www.kaggle.com/datasets/zphudzz/tweets-clean-posneg-v1)  

📌 **Dataset Preview:**
<!-- ![Dataset Preview](https://drive.google.com/uc?id=1wpm_40bfYTrLWnJkSuE19E-VuZhuxste) -->

![test](https://external-content.duckduckgo.com/iu/?u=http%3A%2F%2Fdrive.google.com/uc?id=1wpm_40bfYTrLWnJkSuE19E-VuZhuxste)

✅ **Tasks Completed:**
- Collected **raw tweets dataset** 📝
- Applied **data cleaning & preprocessing** 🧹
- Uploaded the final dataset to **Kaggle** 📤

### **2️⃣ Models that we have done in Assignment 1**
- **Train and Evaluate Models** 🚀
  - Decision Tree 
  - Random Forest 
  - XGBoost
  - Logistic Regression 
  - Naive Bayes
  - GA, HMM, Bayesian Network
  - MLPClassifier and simple Perceptron (ANN)
  - CNN / LSTM for deep learning 

- **Compare Model Performance** 🏆
- **Create Visualizations for Analysis** 📊
- **Write Final Report & Documentation** 📝

### **3️⃣ Models that we have done in Assignment 2**
- **Train and Evaluate Models** 🚀
  - Graphical Models (Bayesian Networks, HMM)
  - Support Vector Machines (kernel and regularization)
  - PCA for Dimension Reduction 
    - Feature Selection Strategies 
    - Topic Modeling, Variance, Chi-squared
  - LDA as linear classifier 
  - Ensemble Models / Methods
    - Random Forest 
    - XGBoost
    - Stacking: Stack multiple Logistic Regressions; Stack multiple "good" models from Assignment 1
    - Voting: Use multiple Logistic Regressions and Use multiple "good" models from Assignment 1 for evaluating
  - Discriminative Models:
    - Logistic Regression 
    - CRF: specifically we use BiLSTM-CRF to make use of the superior performance of BiLSTM model.

- **Compare Model Performance** 🏆
- **Create Visualizations for Analysis** 📊
- **Write Final Report & Documentation** 📝

---

## 💡 **IV. Contributions and Task Distribution**

### Task Distribution for Assignment 1

| **Team Member**  | **Task** |
|------------------|----------|
| **Nguyen Quang Phu** | Team leader; Repository management; Participate and Ensure everything stays on schedule and verify all work done by other members. 📂 |
| **Pham Huynh Bao Dai** | Data preparation; Data preprocessing; feature engineering; Visualization; Document Data Collecting; Preprocessing and Merging. ⚙️|
| **Nguyen Thanh Dat** | Consistent model training and evaluating; Model implementation (Decision Tree, Random Forest, XGBoost, Perceptron - ANN, MLP); Document Model Implementation. 🌳 |
| **Nguyen Tien Hung** | Consistent model training and evaluating; Model implementation (GA, HMM Bayesian Network, Logistic Regression, LSTM); Document Model Implementation. 📈 |
| **Nguyen Thien Loc** | Model evaluation, hyperparameter tuning, Model Comparison, Document Performance analysis. 🔍 |

### Task Distribution for Assignment 2

| **Team Member** | **Task** |
|------------------|----------|
| **Nguyen Quang Phu** | Team leader; Manage repository and schedule; Stacking & Voting with Logistic Regressions; Final integration & verification. 📂|
| **Pham Huynh Bao Dai** | Implement PCA, SVM; Visualizations; Document his Models' workflow ⚙️|
| **Nguyen Thanh Dat** | Implement LDA, Random Forest, XGBoost; Document Ensemble Methods and Classifiers. 🌳|
| **Nguyen Tien Hung** | Build Bayesian Networks, HMMs, Logistic Regression; Stacking with good models from Assignment 1; Document Graphical Models. 📈|
| **Nguyen Thien Loc** | Implement BiLSTM-CRF; Evaluate models; Voting with good models; Performance Analysis and Comparison. 🔍|

---

## 📌 **V. How to Run This Project**

### 1️⃣ **Clone the Repository**
```sh
git clone https://github.com/pdz1804/ML_LHPD2
cd ML_LHPD2
```

### 2️⃣ Setting Up the Environment

If you want to train the model in local (on your computer, in VSC...), to install the necessary dependencies, refer to the environment file (```environment.yml```) provided in the repository. Ensure you have Conda installed to create and activate the required environment.

```bash
conda env create -f environment.yml
conda activate ml_env  # Replace ml_env with your environment name
```

### 3️⃣ **Explore the Project**

- First, you should revisit the Project structure above to see what is the use of those folders and subfolders in this project.
- In details:
    - To know how we collect all the data, visit the ```data``` folder.
    - To see the final preprocessing dataset, visiting this link: [Tweets Clean PosNeg v1](https://www.kaggle.com/datasets/zphudzz/tweets-clean-posneg-v1)  
    - To know how we preprocessing, make all the datasets consistent and how do we merge them together, visit the location ```src/data/process.ipynb``` and its utility file ```src/data/preprocess.py``` 
    - To know how and what we use to make the features for all the texts/documents in the dataset, visit the location ```src/features/build_features_utils.py``` 
    - To know how we perform hyperparameter tuning, do k-fold cross-validation and train our models, visit ```src/models/models_utils.py``` and ```src/models/train_model.ipynb``` 
    - We have also created some visualization functions inside the location ```src/visualization/```. Although now we have not used it anywhere in code because all the visualization or plotting now we just use the **in-line-function-plotting**, but we intend to use it later in Assignment 2.
    - The ```environment.yml``` contains the information about the libraries or dependencies that we use when performing and executing our project.


---

## 📄 **VI. License**
This project is licensed under the **MIT License**.

---

## 📧 **VII. Contact**
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

