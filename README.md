# 🧠 Depression Detection Using Sentiment Analysis 5/1/2025

This project applies sentiment analysis and machine learning to detect depressive symptoms in social media text. Our goal is to identify at-risk individuals by analyzing patterns in their language, and support timely mental health interventions.

## 🎯 Objectives

- Identify linguistic features and sentiment indicators linked to depression.
- Develop and evaluate machine learning models for depression detection.
- Deploy a user-friendly interface for real-time prediction.

## 🛠️ Tools & Technologies

- **Python**
- **Google Colab**
- **Gradio** (for deployment)
- **Kaggle** (for code sharing)
- **Machine Learning Model**: **XGBoost**, **Bernoulli Naive Bayes**, **Decision Tree**, **Logistic Regression**

## 🧪 Modeling Techniques

We evaluated four classification models:

| Model                   | Accuracy | Strengths                                     |
|------------------------|----------|-----------------------------------------------|
| Bernoulli Naive Bayes  | 64%      | Simple and fast, but limited with complex data |
| Decision Tree          | 62%      | Interpretable but prone to overfitting        |
| Logistic Regression    | 77%      | Good for linear relationships                 |
| **XGBoost**            | **81%**  | Best overall; captures complex patterns       |

📌 **XGBoost was selected as the final model** due to its high accuracy and robustness.

## 📊 Data Insights

Common words associated with depressive sentiment:
`feel`, `want`, `life`, `know`, `people`, `even`, `depression`, `better`, `time`, `family`, `friend`

These words guided our sentiment and feature engineering processes.

## 🚀 Deployment Quick Start (Gradio on Google Colab)

You can launch the Gradio app **without re-training the model** by following these steps:

### 🛠️ Step-by-Step Instructions
1. Open the Jupyter notebook: `depression_detection.ipynb` in **Google Colab**
2. Download and upload the following files into the Colab session:
   - ✅ `xgb_depression_model.joblib` — pre-trained XGBoost model
   - ✅ `tfidf_vectorizer.pkl` — TF-IDF vectorizer
3. Skip the training cells and scroll to the **Gradio deployment section**
4. Run the Gradio cell to launch the web interface
5. Enter any text and receive a prediction: `Depressed` or `Not Depressed`

## 🧪 Testing the Model with Sample Inputs

The repository includes a test file: `Testing Text.txt` for quick experimentation.

### 📄 About the File:
- Contains both **depressed** and **non-depressed** example sentences
- Each line = one input text for prediction

### 🔍 How to Use It:
1. Open the Gradio app in Colab
2. Open `Testing Text.txt`
3. Copy a line and paste it into the input box
4. Click **Submit** and view the prediction

This helps evaluate how the model performs on real-like social media text inputs.

## 🔁 Reproducibility

- All code is available on [Kaggle](https://www.kaggle.com/code/szegeelim/depression-detection-using-sentiment-analysis)
- Managed dependencies using `requirements.txt`
- Notebook-based development in Google Colab uploaded here as WQD7001_GA2.ipynb

## 🤝 Stakeholder Impact

- **Healthcare Providers**: Early identification of mental health risks.
- **Social Media Platforms**: Enhanced user monitoring & support.
- **Mental Health Organizations**: Better targeted resources and outreach.

## 🔮 Future Work

- Integrate LSTM/BERT for deep learning-based sentiment analysis.
- Apply **Random Search** for hyperparameter tuning.
- Expand dataset and incorporate multilingual sentiment analysis.
- Ensure ethical use and privacy compliance in real-world deployment.

---

### 👥 Team

This project was completed as part of the WQD7001 *Principles of Data Science* course by **Group 2** in 5th January 2025.

---

### 📄 License

This project is for academic purposes only.

