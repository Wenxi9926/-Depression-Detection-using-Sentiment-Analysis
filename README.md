# Depression-Detection-using-Sentiment-Analysis
# 🧠 Depression Detection Using Sentiment Analysis

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
- **Scikit-learn**, **XGBoost**, **NLTK**

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

## 🚀 Deployment

We used **Gradio** to build an interactive interface for users to input text and receive real-time depression predictions.

🌐 **Live Demo**: [Click to launch Gradio App](https://ebba8a928c165679f6.gradio.live)

- Input: Any social media-like text
- Output: "Depressed" or "Not Depressed"

## 🔁 Reproducibility

- All code is available on [Kaggle](https://www.kaggle.com/code/szegeelim/depression-detection-using-sentiment-analysis)
- Managed dependencies using `requirements.txt`
- Notebook-based development in Google Colab

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

This project was completed as part of the WQD7001 *Principles of Data Science* course by **Group 11**.

---

### 📄 License

This project is for academic purposes only.

