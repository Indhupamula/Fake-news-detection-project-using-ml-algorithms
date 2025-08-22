# 📰 Fake News Detection Using Machine Learning

## 🔍 Overview
The spread of fake news on the internet is a big problem today. This project aims to create a machine-learning model that can tell whether a news article is fake or real. We use different techniques to analyze and understand the text in these articles. Our dataset includes articles that are already labeled as 'fake' or 'real', which helps us train and test our model.

## 📝 Objective

The main goal is to accurately classify news articles as either 'fake' or 'real' based on their content. By leveraging the power of machine learning and text analysis, we aim to create a reliable tool that can help mitigate the spread of misinformation.

## 📄 Dataset

The dataset used in this project is sourced from the [Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets). It includes a balanced set of labeled news articles, which consist of the following features:

- **Title**: The headline of the news article.
- **Text**: The full-text content of the news article.
- **Label**: 1 for fake news and 0 for real news.

## 📋 Methodology

### i. Data Preprocessing:

- **Exploratory Data Analysis (EDA)**: Understanding the data distribution, identifying patterns, and visualizing trends.
- **Text Cleaning**: Removing noise, stop words, and irrelevant characters from the text data.
- **Feature Engineering**: Converting text into numerical representations using techniques like TF-IDF, Bag of Words, and word embeddings.

### ii. Model Selection:

- **Algorithm Comparison**: Evaluating the performance of various models such as Logistic Regression, Random Forest, Decision Tree, Gradient Boost, and Support Vector Machines (SVM).
- **Hyperparameter Tuning**: Optimizing model parameters through techniques like Grid Search and Cross-Validation to enhance model performance.

### iii. Model Evaluation:

- **Evaluation Metrics**: Using metrics such as accuracy, precision, recall, F1-score, and ROC AUC to assess model performance.
- **Confusion Matrix**: Analyzing the confusion matrix to understand the model's ability to distinguish between fake and real news.

### iv. Deployment:

- **Prediction Script**: Creating a Python script to predict whether a new article is fake or real.
- **Web Interface (Optional)**: Developing a simple web application using Streamlit for news classification.

## ✅ Requirements

- **Python 3.x**
- **Libraries**: pandas, numpy, scikit-learn, nltk, matplotlib, seaborn, Streamlit (optional), etc.

## 🌐 The Web Interface
![WhatsApp Image 2024-06-25 at 22 58 42_6c529be6](https://github.com/jicsjitu/Fake_News_Using_ML/assets/162569175/1dc301d6-337a-4b87-a984-21bdc0fa9402)

## 🧪 Testing Model
i. **For True News** 

![WhatsApp Image 2024-06-25 at 23 02 17_7eb13c2b](https://github.com/jicsjitu/Fake_News_Using_ML/assets/162569175/50abbd22-91d8-46fa-bba6-d56bc1ccdcc6)

ii. **For Fake News**

![WhatsApp Image 2024-06-25 at 23 03 56_1155faec](https://github.com/jicsjitu/Fake_News_Using_ML/assets/162569175/77174024-0819-4409-b2dc-291899009645)

## 🖥️ Usage

i. **Clone the repository**:
   ```bash
   git clone https://github.com/jicsjitu/Fake_News_Using_ML.git
   cd Fake_News_Detection_Using_ML
   ```

ii. **Install the required libraries**:
   ```bash
   pip install -r requirements.txt
   ```

## 📁 Files in the Repository

Certainly! Here’s a simple description of each file and folder in the repository:

### i. README.md

This is the file you’re currently reading. It provides an overview of the project, how to set it up, use it, and other essential information about the fake news detection project.

### ii. Visualizations Folder

This folder contains various charts, graphs, and plots that help visualize data insights and the performance of our model. You’ll find:
- Histograms and scatter plots showing the distribution of data.
- Word clouds illustrating the most common words in fake and real news articles.
- ROC curves and confusion matrices that display how well our model is performing.

### iii. app.py

This Python script is the main application file for predicting whether news articles are fake or real. It uses the trained machine learning model to classify new articles based on their text content. 

### iv. Fake.csv

This file contains the data of fake news articles used for training and testing our model. Each row represents an article labeled as fake, along with its title and content.

### v. True.csv

This file holds the data of real news articles. Similar to `Fake.csv`, it includes articles that are labeled as real, along with their title and content.

### vi. fake_news_detection_code.ipynb

This is a Jupyter notebook that includes the complete code for our project. It covers:
- Data preprocessing steps like cleaning and preparing the text.
- Feature extraction methods to convert text into a format suitable for machine learning.
- Training and evaluating different machine learning models to find the best one for detecting fake news.

### vii. manual_testing.csv

This file is used for manually testing our model. It contains a small set of news articles that you can use to see how well the model predicts fake or real news. It’s helpful for checking the model’s performance on new, unseen data.

### viii. Fake_News_Detection_Using_ML.pptx

This is a PowerPoint presentation that explains the project. It includes slides on the problem of fake news, how we approached the solution, the data and methods we used, the results we achieved, and the impact of our work. It's perfect for sharing our findings with others.

### ix. requirements.txt

This file lists all the Python libraries and packages needed to run the project. By installing these dependencies, you can ensure your environment is set up correctly to execute the code and scripts in this project. 

## 📈 Results

Our best-performing model achieved the following metrics:
- **Accuracy**: 99%
- **Precision**: 99%
- **Recall**: 99%
- **F1 Score**: 99%

## 👤 Credits

- **Dataset**: [Dataset](https://www.kaggle.com/datasets/emineyetm/fake-news-detection-datasets)

## 💬 Feedback and Questions

If you have any feedback or questions about the project, please feel free to ask. We appreciate your input and are here to help. You can reach out by opening an issue on GitHub or by emailing me at jitukumar9387@gmail.com.

Thank you for exploring the Fake News Detection Project! We hope you find it useful and informative.

Happy coding!
