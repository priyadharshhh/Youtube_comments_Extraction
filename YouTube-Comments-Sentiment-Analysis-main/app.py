import os
import time
import re
import nltk
import requests
import numpy as np
import torch
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
from wordcloud import WordCloud, STOPWORDS
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from webdriver_manager.chrome import ChromeDriverManager

# Download necessary NLTK data
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialize Flask
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Initialize NLP tools
wnl = WordNetLemmatizer()
sia = SentimentIntensityAnalyzer()
stop_words = set(stopwords.words('english'))

# Load mBERT tokenizer and model
mbert_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
mbert_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")

# Setup Chrome options for Selenium WebDriver
options = Options()
options.add_argument("--headless")  # Run without opening a browser window
options.add_argument("--disable-gpu")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")

# YouTube API Key (Replace with your actual API key)
YOUTUBE_API_KEY = ""

def extract_video_id(url):
    """Extracts the YouTube video ID from various possible URL formats."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    return match.group(1) if match else None

def fetch_comments_api(video_id):
    """Fetches YouTube comments using the YouTube Data API."""
    comments = []
    try:
        url = f"https://www.googleapis.com/youtube/v3/commentThreads?key={YOUTUBE_API_KEY}&textFormat=plainText&part=snippet&videoId={video_id}&maxResults=100"
        response = requests.get(url)
        data = response.json()

        if 'items' in data:
            comments = [item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in data['items']]
        
        return comments
    except Exception as e:
        print("API Error:", str(e))
        return []

def fetch_comments_selenium(url):
    """Scrapes YouTube comments using Selenium (Fallback)."""
    comments = []
    try:
        with webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options) as driver:
            wait = WebDriverWait(driver, 15)
            driver.get(url)

            # Scroll to load comments
            for _ in range(5):  # Scroll multiple times to load more comments
                wait.until(EC.visibility_of_element_located((By.TAG_NAME, "body"))).send_keys(Keys.END)
                time.sleep(2)

            # Extract comments
            elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, "#content")))
            comments = [el.text for el in elements if el.text.strip()]

        print(f"Retrieved {len(comments)} comments via Selenium.")
        return comments

    except Exception as e:
        print("Selenium Error:", str(e))
        return []

def clean_comments(comments):
    """Cleans and preprocesses comments."""
    cleaned_comments = []
    for comment in comments:
        comment = comment.lower()
        comment = re.sub(r"http\S+|www\S+|https\S+", '', comment, flags=re.MULTILINE)
        comment = re.sub(r'[^a-zA-Z\s]', '', comment)  # Remove punctuation and special characters
        words = [wnl.lemmatize(word) for word in comment.split() if word not in stop_words and len(word) > 2]
        cleaned_comments.append(' '.join(words))

    return cleaned_comments

def create_wordcloud(comments):
    """Generates and saves a word cloud from the comments."""
    if not comments:
        print("No comments available for word cloud.")
        return

    text = ' '.join(comments).strip()

    if not text:
        print("Word cloud cannot be generated. No meaningful words found.")
        return

    wc = WordCloud(width=1400, height=800, stopwords=set(STOPWORDS), background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.savefig('static/images/wc.png')
    plt.close()

def analyze_sentiment(comment):
    """Performs sentiment analysis using mBERT and VADER."""
    
    # **Step 1: Use VADER for English**
    vader_score = sia.polarity_scores(comment)['compound']
    if vader_score >= 0.05:
        vader_sentiment = 'Positive'
    elif vader_score <= -0.05:
        vader_sentiment = 'Negative'
    else:
        vader_sentiment = 'Neutral'

    # **Step 2: Use mBERT for Multilingual Sentiment Analysis**
    tokens = mbert_tokenizer(comment, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        output = mbert_model(**tokens)
    
    probs = softmax(output.logits, dim=-1)
    label = torch.argmax(probs).item()

    # Convert label (0-4) to sentiment
    mbert_sentiment = ['Very Negative', 'Negative', 'Neutral', 'Positive', 'Very Positive'][label]

    # **Step 3: Combine Both Results**
    if vader_sentiment == 'Neutral' and mbert_sentiment in ['Very Positive', 'Very Negative']:
        return mbert_sentiment
    elif mbert_sentiment == 'Neutral':
        return vader_sentiment
    else:
        return mbert_sentiment if probs[label].item() > 0.6 else vader_sentiment

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/results', methods=['GET'])
def result():
    url = request.args.get('url')
    video_id = extract_video_id(url)

    if not video_id:
        return "Invalid YouTube URL. Please provide a valid video link."

    # Try API first, fallback to Selenium
    comments = fetch_comments_api(video_id)
    if not comments:
        print("API failed or limit exceeded. Switching to Selenium...")
        comments = fetch_comments_selenium(url)

    if not comments:
        return "No comments found for this video. Please try another video."

    clean_comments_list = clean_comments(comments)
    create_wordcloud(clean_comments_list)

    sentiments = [analyze_sentiment(comment) for comment in clean_comments_list]
    np, nn, nne = sentiments.count('Positive') + sentiments.count('Very Positive'), sentiments.count('Negative') + sentiments.count('Very Negative'), sentiments.count('Neutral')

    results = [
        {'comment': org, 'clean_comment': clean, 'sentiment': sent}
        for org, clean, sent in zip(comments, clean_comments_list, sentiments)
    ]

    return render_template('result.html', n=len(clean_comments_list), np=np, nn=nn, nne=nne, dic=results)

@app.route('/wc')
def wc():
    return render_template('wc.html')

if __name__ == '__main__':
    app.run(debug=True)
