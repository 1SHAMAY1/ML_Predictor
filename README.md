# 🏥 Medical Insurance Cost Predictor (AI/ML)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end Machine Learning application that predicts annual medical insurance premiums using a **Random Forest Regressor** trained on real-world historical data.

---

## 🚀 Live Demo
**[Launch the Interactive Web App!](https://mlpredictor-1shamay1.streamlit.app)**  
---

## 📸 Project Preview
<img width="1896" height="1003" alt="image" src="https://github.com/user-attachments/assets/3e4b82b2-3cac-4cfb-bf2f-a4cca5d87b43" />


---

## 🧠 How it Works
This project uses **Data Science** fundamentals to extract insights from the "Medical Cost Personal Datasets."

1.  **Exploratory Data Analysis (EDA)**: Analyzed the impact of features like smoking, BMI, and age on insurance charges.
2.  **Preprocessing**: Categorical variables (Gender, Smoker, Region) were encoded using `LabelEncoder`.
3.  **Model Training**: Used a **Random Forest Regressor** with Hyperparameter Tuning (`GridSearchCV`) to achieve an **R2 Score of ~87%**.
4.  **Deployment**: Built a responsive UI using **Streamlit** for real-time predictions.

## 🛠️ Tech Stack
- **Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-Learn, Joblib
- **UI Framework**: Streamlit
- **Model**: Random Forest Regressor

## 🏃 How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/1SHAMAy1/ML_Predictor.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the training script (to generate the model):
   ```bash
   python train_model.py
   ```
4. Start the app:
   ```bash
   streamlit run app.py
   ```

### ⚡ Quick Start (Windows)
For a one-click setup and execution, simply run the included batch script:
```bash
run_app.bat
```
This will automatically install dependencies, train the model, and launch the web app.

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
