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

## 👥 **Team Members**

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
- **ONE unified dataset** before the midterm (Chapters 2-6) (which we have done in Assignment 1).
- **ANOTHER dataset** for the second stage after midterm (Chapters 6-10) (which we do in this Assignment 2).

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


### **2️⃣ Models that we have done in Assignment 2**
- **Train and Evaluate Models** 🚀
  - Some models
  - Some models
  - Some models
  - Some models
  - Some models

- **Compare Model Performance** 🏆
- **Create Visualizations for Analysis** 📊
- **Write Final Report & Documentation** 📝

---

## 💡 **III. Contributions and Task Distribution**

| **Team Member**  | **Task** |
|------------------|----------|
| **Nguyen Quang Phu** | Team leader; Repository management; Participate and Ensure everything stays on schedule and verify all work done by other members. 📂 |
| **Pham Huynh Bao Dai** | Data preparation; Data preprocessing; feature engineering; Document Data Collecting; Preprocessing and Merging. ⚙️|
| **Nguyen Thanh Dat** | Consistent model training and evaluating; Model implementation (Some models); Document Model Implementation. 🌳 |
| **Nguyen Tien Hung** | Consistent model training and evaluating; Model implementation (Some models); Document Model Implementation. 📈 |
| **Nguyen Thien Loc** | Model evaluation, hyperparameter tuning, Model Comparison, Document Performance analysis. 🔍 |

---

## 📌 **IV. How to Run This Project**

### 1️⃣ **Clone the Repository**
```sh
git clone https://github.com/pdz1804/ML_LHPD2
cd ML_LHPD2
```

### 2️⃣ **Explore the Project**

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

## 📄 **V. License**
This project is licensed under the **MIT License**.

---

## 📧 **VI. Contact**
For any questions or contributions, please contact:

📩 Email: phu.nguyenquang2004@hcmut.edu.vn

🔗 GitHub: https://github.com/pdz1804/

--- 

<h2 align="center">✨💟 Contributors 💟✨</h2>

<p align="center">
  <strong>We fairly contribute to this repository with dedication and teamwork!</strong> 💖
</p>

<div align="center">
  <a href="https://github.com/pdz1804"><img src="https://avatars.githubusercontent.com/u/123137268?v=4" title="pdz1804" width="50" height="50"></a>
  <a href="https://github.com/MarkX04"><img src="https://avatars.githubusercontent.com/u/105540317?v=4" title="MarkX04" width="50" height="50"></a>
  <a href="https://github.com/DatNguyen1402"><img src="https://avatars.githubusercontent.com/u/137872945?v=4" title="DatNguyen1402" width="50" height="50"></a>
  <a href="https://github.com/hungyle123"><img src="https://avatars.githubusercontent.com/u/138371452?v=4" title="hungyle123" width="50" height="50"></a>
  <a href="https://github.com/nguyen1oc"><img src="https://avatars.githubusercontent.com/u/131537455?v=4" title="nguyen1oc" width="50" height="50"></a>
</div>

---