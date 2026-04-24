import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(page_title="SL Smart Tour", layout="wide")

# ----------------------------
# APP THEME (CSS + Background only on Home)
# ----------------------------
st.markdown("""
<style>
h1, h2, h3, h4, h5, h6 { color: #1b3c59; font-weight: 700; }
.stButton>button {
    background: linear-gradient(90deg, #ffb347, #ffcc33);
    color: white;
    border-radius: 12px;
    padding: 0.5em 1.2em;
    font-size: 1rem;
    font-weight: bold;
    border: none;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #ffcc33, #ffb347);
}
.card {
    background-color: #ffffffcc;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 3px 3px 15px rgba(0,0,0,0.15);
}
.map-frame { border-radius: 15px; box-shadow: 0 0 20px rgba(0,0,0,0.2); }
.home-overlay {
    background-color: rgba(255,255,255,0.75);
    padding: 25px;
    border-radius: 15px;
    box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
}
.css-1d391kg label, .css-1q8dd3e, .css-1inwz65 { font-size: 20px !important; }
.css-1v3fvcr, .css-1p5j3m4 { font-size: 20px !important; }
.block-container { padding-top: 0rem !important; }
header, .stApp header, [data-testid="stHeader"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# LOAD DATA
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv('F:/KDU/6th Semester/Dissertation/ML Model/Reviews.csv', encoding='latin1')
    df = df[['Location_Name','Location_Type','Location','Text','Rating']].dropna()

    # ---------------------------------------------
    # ML SENTIMENT CLASSIFIER
    # ---------------------------------------------
    df['Sentiment'] = df['Rating'].apply(lambda r: 'Positive' if r >= 4 else 'Negative' if r <= 2 else 'Neutral')
    X_train, X_test, y_train, y_test = train_test_split(df['Text'], df['Sentiment'], 
                                                        test_size=0.2, random_state=42)
    tfidf_clf = TfidfVectorizer(stop_words='english', max_df=0.8)
    X_train_tfidf = tfidf_clf.fit_transform(X_train)
    X_test_tfidf = tfidf_clf.transform(X_test)
    clf = LogisticRegression(max_iter=300)
    clf.fit(X_train_tfidf, y_train)
    df['Predicted_Sentiment'] = clf.predict(tfidf_clf.transform(df['Text']))
    sentiment_map = {"Negative": -1, "Neutral": 0, "Positive": 1}
    df['Sentiment_Score'] = df['Predicted_Sentiment'].map(sentiment_map)
    
    location_reviews = df.groupby(['Location_Name','Location_Type','Location'])['Text'].apply(lambda x: " ".join(x)).reset_index()
    sentiment_avg = df.groupby('Location_Name')['Sentiment_Score'].mean().reset_index()
    location_reviews = pd.merge(location_reviews, sentiment_avg, on='Location_Name')
    return location_reviews

location_reviews = load_data()

# ----------------------------
# SESSION STATE INIT
# ----------------------------
for key in ["page","show_results","selected_location","recommendations"]:
    if key not in st.session_state:
        st.session_state[key] = None
st.session_state.page = st.session_state.page or "home"
st.session_state.show_results = st.session_state.show_results or False

# ----------------------------
# NAVIGATION
# ----------------------------
pages = ["🏠 Home","🧭 Get Recommendations","🗺️ Map View"]
if st.session_state.page=="home":
    selected_page = st.sidebar.radio("Navigation", pages, index=0)
elif st.session_state.page=="recommend":
    selected_page = st.sidebar.radio("Navigation", pages, index=1)
else:
    selected_page = st.sidebar.radio("Navigation", pages, index=2)
st.session_state.page = {"🏠 Home":"home","🧭 Get Recommendations":"recommend","🗺️ Map View":"map"}[selected_page]

# ----------------------------
# LOCATION TYPE ICONS
# ----------------------------
location_type_icons = {
    "Beach":"🏖️","Mountain":"⛰️","Culture":"🏛️","Adventure":"🚵",
    "Historical":"🏰","Nature":"🌳","Wildlife":"🦁","Relaxation":"🛶","Luxury":"🏨"
}

# ----------------------------
# HOME PAGE
# ----------------------------
if st.session_state.page=="home":
    st.markdown("""
    <style>
    .stApp { background: url('https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1470&q=80') no-repeat center center fixed; background-size: cover; }
    </style>
    """, unsafe_allow_html=True)
    st.markdown("<div class='home-overlay'>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center;'>🗺️ SL Smart Tour – Personalized Travel Recommender</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
        <h3>✨ Personalized Sri Lanka Travel Planner </h3>
        <p>Discover the best travel destinations in Sri Lanka tailored just for you! This app recommends places based on your interests and preferences.</p>
    </div>
    """, unsafe_allow_html=True)
    col_center = st.columns(3)[1]
    with col_center:
        if st.button("🚀 Discover Now"):
            st.session_state.page="recommend"
            st.rerun()

# ----------------------------
# RECOMMENDATION PAGE
# ----------------------------
elif st.session_state.page=="recommend":
    st.subheader("🧳 Personalize Your Journey")
    col1,col2 = st.columns(2)
    with col1:
        travel_type = st.selectbox("👥 Travel Group",["Solo","Couple","Family","Friends","Business"])
        budget = st.select_slider("💰 Budget Range",["Low","Medium","High"])
    with col2:
        country = st.text_input("🌐 Home Country (optional)")
        travel_purpose = st.multiselect("🎯 Travel Interest",["Adventure","Relaxation","Culture","Nature","Luxury","Historical","Wildlife"])
    
    st.divider()
    st.subheader("🎯 Destination Preference")
    location_types = sorted(location_reviews['Location_Type'].dropna().unique().tolist())
    selected_types = st.multiselect("Choose preferred location types:", location_types, default=[])
    
    keywords = st.text_input("Optional keywords (e.g., surfing, hiking, temple):","")
    
    if st.button("🔍 Find Recommended Locations"):
        if not selected_types:
            st.warning("Please select at least one destination type.")
        else:
            all_recs = []
            for loc_type in selected_types:
                filtered = location_reviews[location_reviews['Location_Type']==loc_type].reset_index(drop=True)
                if filtered.empty: continue
                tfidf_rec = TfidfVectorizer(stop_words='english', max_df=0.8, min_df=1)
                tfidf_matrix = tfidf_rec.fit_transform(filtered['Text'])
                text_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
                idx=0
                text_scores = text_sim[idx]
                sent_scores = 1 - np.abs(filtered['Sentiment_Score'] - filtered['Sentiment_Score'].iloc[idx])
                final_score = 0.7*text_scores + 0.3*sent_scores
                top_idx = final_score.argsort()[::-1][1:6]
                recs = filtered['Location_Name'].iloc[top_idx].tolist()
                all_recs.extend(recs)
            st.session_state.recommendations = location_reviews[location_reviews['Location_Name'].isin(all_recs)]
            st.session_state.show_results=True
    
    if st.session_state.show_results and st.session_state.recommendations is not None:
        st.subheader("🌟 Top Recommendations")
        for i, (_, row) in enumerate(st.session_state.recommendations.iterrows(),1):
            icon = location_type_icons.get(row['Location_Type'],"📍")
            st.markdown(f"""
            <div class='card'>
                <h4>#{i} {icon} <b>{row['Location_Name']}</b></h4>
                <p>📍 Location: <b>{row['Location']}</b></p>
                <p>🏷️ Type: <b>{row['Location_Type']}</b></p>
            </div>
            """, unsafe_allow_html=True)
            if st.button(f"📍 View {row['Location_Name']} on Map", key=f"map_{i}"):
                st.session_state.selected_location=row['Location_Name']
                st.session_state.page="map"
                st.rerun()

# ----------------------------
# MAP PAGE
# ----------------------------
elif st.session_state.page=="map":
    st.markdown("<h2>🗺️ Explore on Google Maps</h2>", unsafe_allow_html=True)
    selected_place = st.session_state.selected_location or st.selectbox("🌍 Choose a location:", location_reviews['Location_Name'].unique())
    st.info(f"Showing map for: **{selected_place}**")
    google_map_url = f"https://www.google.com/maps?q={selected_place.replace(' ','+')}&output=embed"
    st.components.v1.html(f"<iframe src='{google_map_url}' width='100%' height='500' class='map-frame' allowfullscreen loading='lazy'></iframe>", height=520)
    if st.button("🔙 Back to Recommendations"):
        st.session_state.page="recommend"
        st.rerun()
