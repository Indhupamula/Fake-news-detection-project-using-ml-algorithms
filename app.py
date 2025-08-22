import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Streamlit page configuration
st.set_page_config(page_title="Fake News Detector", layout="wide")

# Download required NLTK resources
nltk.download('stopwords')

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

# -----------------------------
# Preprocessing
# -----------------------------
def stem_content(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if word not in stop_words]
    return ' '.join(stemmed_content)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_and_process_data(sample_size=1000):
    dataframe_fake = pd.read_csv("Fake.csv")
    dataframe_true = pd.read_csv("True.csv")
    dataframe_fake['label'] = 1
    dataframe_true['label'] = 0
    news_dataframe = pd.concat(
        [dataframe_fake.sample(sample_size), dataframe_true.sample(sample_size)],
        axis=0
    ).reset_index(drop=True)
    return news_dataframe

# -----------------------------
# Vectorize Data
# -----------------------------
@st.cache_data
def vectorize_data(news_dataframe):
    news_dataframe['content'] = news_dataframe['text'].apply(stem_content)
    vectorizer = TfidfVectorizer(max_features=500)
    X = vectorizer.fit_transform(news_dataframe['content'])
    y = news_dataframe['label'].values
    return X, y, vectorizer

# -----------------------------
# Train Model
# -----------------------------
@st.cache_resource
def train_model(_X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        _X, y, test_size=0.2, stratify=y, random_state=2
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy

# -----------------------------
# Main Execution
# -----------------------------
news_dataframe = load_and_process_data()
X, y, vectorizer = vectorize_data(news_dataframe)
model, accuracy = train_model(X, y)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title('📰 Fake News Detector')
st.write("""
    Welcome to the Fake News Detector! This application uses Natural Language Processing (NLP) techniques to classify news articles as real or fake. 
    Simply enter the news text in the box below or upload a text file and click on 'Predict' to see the result.
""")

st.sidebar.title("🛠 App Options")
if st.sidebar.button("Reload Data"):
    st.cache_data.clear()
    st.experimental_rerun()

st.sidebar.markdown("### Model Performance")
st.sidebar.write(f"Accuracy: {accuracy:.2%}")

# -----------------------------
# Input Section
# -----------------------------
input_text = ""
uploaded_file = None
with st.form(key='news_form'):
    input_text = st.text_area('Enter News Article', height=250)
    uploaded_file = st.file_uploader("Or upload a text file", type=["txt"])
    if uploaded_file is not None:
        input_text = uploaded_file.read().decode("utf-8")
        st.text_area('File Content', input_text, height=250)
    submit_button = st.form_submit_button(label='Predict')
    clear_button = st.form_submit_button(label='Clear Input')

if clear_button:
    input_text = ""
    uploaded_file = None

if submit_button:
    if input_text:
        with st.spinner('Analyzing the article...'):
            # Apply same preprocessing as training
            processed_text = stem_content(input_text)
            input_data = vectorizer.transform([processed_text])
            pred = model.predict(input_data)[0]
            confidence = model.predict_proba(input_data)[0][pred]
            result = '🚨 The News is Fake' if pred == 1 else '✅ The News is Real'
            st.success(result)
            st.info(f"Confidence: {confidence:.2%}")

            st.write("Key Features:")
            feature_names = vectorizer.get_feature_names_out()
            coefficients = model.coef_[0]
            sorted_indices = coefficients.argsort()
            if pred == 1:
                top_features = [feature_names[i] for i in sorted_indices[:10]]
            else:
                top_features = [feature_names[i] for i in sorted_indices[-10:]]
            st.write(top_features)
    else:
        st.error("Please enter some text to analyze.")



