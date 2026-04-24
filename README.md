# Sentiment-Based-Travel-Recommendation-System-Sri-Lanka-
A machine learning-based travel recommendation system that uses NLP and sentiment analysis on tourist reviews to generate personalized destination suggestions in Sri Lanka.

# Sentiment-Based Travel Recommendation System (Sri Lanka)

## 📌 Overview
The tourism industry is a major contributor to Sri Lanka’s economy, generating foreign exchange, employment, and regional development. However, personalized travel recommendations are often based on unstructured tourist opinions, making them difficult to analyze and use effectively.

This project presents a **Sentiment-Based Travel Recommendation System** that leverages machine learning and natural language processing (NLP) to analyze tourist reviews and generate personalized travel suggestions.

---

## 🎯 Objectives
- Analyze tourist reviews using sentiment analysis  
- Classify reviews into **positive, neutral, and negative sentiments**  
- Develop a **personalized travel recommendation system**  
- Enhance decision-making using data-driven insights  

---

## ⚙️ Technologies Used
- **Programming Language:** Python  
- **Machine Learning:** Logistic Regression  
- **NLP Techniques:** TF-IDF Vectorization  
- **Libraries:** Pandas, NumPy, Scikit-learn  
- **Web Application:** SL Smart Tour  

---

## 🧠 Methodology

### 1. Data Collection
- Collected historical tourist reviews from various sources  

### 2. Data Preprocessing
- Text cleaning (removing stopwords, punctuation, etc.)  
- Tokenization and normalization  

### 3. Feature Extraction
- Applied **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text into numerical features  

### 4. Model Development
- Trained a **Logistic Regression model** to classify sentiments:
  - Positive  
  - Neutral  
  - Negative  

### 5. Recommendation System
- Developed a **hybrid scoring model** combining:
  - Sentiment predictions  
  - Text similarity  
- Ranked destinations based on user preferences and sentiment scores  

---

## 🌐 Application
The system is implemented as a web application called **SL Smart Tour**, which:
- Accepts user input/preferences  
- Analyzes relevant tourist reviews  
- Generates **personalized travel recommendations**  
- Ranks destinations based on sentiment and similarity  

---

## 📊 Results
- Successfully classified tourist sentiments with good accuracy  
- Generated personalized recommendations based on user input  
- Improved destination ranking using sentiment-driven insights  

---

## ⚠️ Limitations
- Lower performance in predicting **minority sentiment classes** (e.g., neutral)  
- Limited incorporation of additional user preferences (budget, weather, etc.)  
- Dependent on quality and quantity of review data  

---

## 🚀 Future Improvements
- Integrate deep learning models (e.g., LSTM, BERT) for better sentiment analysis  
- Include user-specific preferences (budget, travel type, seasonality)  
- Expand dataset with real-time reviews  
- Improve recommendation accuracy using advanced hybrid models  
