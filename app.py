import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
import re
from collections import Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

@st.cache_resource
def download_nltk_deps():
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass
download_nltk_deps()

# Page config
st.set_page_config(page_title="GATEWAYS-2025 Dashboard", layout="wide")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    h1 {
        color: #1e3d59;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("fest_dataset.csv")
    
    # State coordinates
    state_coords = {
        'Kerala': [76.2711, 10.8505],  # [Longitude, Latitude] for Shapely Points
        'Tamil Nadu': [78.6569, 11.1271],
        'Uttar Pradesh': [80.9462, 26.8467],
        'Karnataka': [75.7139, 15.3173],
        'Telangana': [79.0193, 18.1124],
        'Delhi': [77.1025, 28.7041],
        'Gujarat': [71.1924, 22.2587],
        'Rajasthan': [74.2179, 27.0238],
        'Maharashtra': [75.7139, 19.7515]
    }
    
    df['Longitude'] = df['State'].map(lambda x: state_coords.get(x, [None, None])[0])
    df['Latitude'] = df['State'].map(lambda x: state_coords.get(x, [None, None])[1])
    
    return df

try:
    df = load_data()
except Exception as e:
    st.error(f"Error loading dataset: {e}")
    st.stop()

# Header
st.title("GATEWAYS-2025 National Level Fest Dashboard")
st.markdown("**Interactive analytics for participation and feedback insights.**")

# Sidebar - Layouts & Input Widgets
st.sidebar.header("Analytical Filters")
selected_event = st.sidebar.multiselect(
    "Select Event", 
    options=df['Event Name'].unique(), 
    default=df['Event Name'].unique()
)

filtered_df = df[df['Event Name'].isin(selected_event)]

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Filtered Participants", len(filtered_df), f"{len(filtered_df) - len(df)} from total")
with col2:
    st.metric("Events Selected", filtered_df['Event Name'].nunique(), f"{filtered_df['Event Name'].nunique() - df['Event Name'].nunique()} from total")
with col3:
    st.metric("Colleges Participated", filtered_df['College'].nunique())
with col4:
    avg_rating = filtered_df['Rating'].mean()
    total_avg = df['Rating'].mean()
    delta_avg = avg_rating - total_avg if pd.notna(avg_rating) and pd.notna(total_avg) else 0.0
    st.metric("Average Rating", f"{avg_rating:.2f} / 5" if pd.notna(avg_rating) else "0 / 5", f"{delta_avg:.2f} vs overall")

st.markdown("---")

# 1. Analysis of Participation Trends
st.header("1. Participation Trends & Geography")
col_trend1, col_trend2 = st.columns(2)

with col_trend1:
    st.subheader("Event-wise Participation")
    event_counts = filtered_df['Event Name'].value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(event_counts.index, event_counts.values, color='skyblue')
    ax.set_ylabel("Participants", fontsize=10)
    ax.set_title("Participation per Event", fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

with col_trend2:
    st.subheader("College-wise Participation (Top 10)")
    college_counts = filtered_df['College'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.pie(college_counts.values, labels=college_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
    ax.set_title("Top 10 Colleges", fontsize=12)
    ax.axis('equal') 
    st.pyplot(fig)

# Map Plot using GeoPandas
st.subheader("State-wise Participants in INDIA Map")
st.markdown("Visualizing Geodata using GeoPandas & Matplotlib")
state_counts = filtered_df.dropna(subset=['Longitude', 'Latitude'])
if not state_counts.empty:
    state_agg = state_counts.groupby(['State', 'Longitude', 'Latitude']).size().reset_index(name='Participants')
    
    # Create GeoDataFrame
    geometry = [Point(xy) for xy in zip(state_agg['Longitude'], state_agg['Latitude'])]
    gdf = gpd.GeoDataFrame(state_agg, geometry=geometry)
    
    try:
        url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"
        india = gpd.read_file(url)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        india.plot(ax=ax, color='lightgrey', edgecolor='black')
        
        # Plot bubble map depending on participants
        gdf.plot(ax=ax, color='red', markersize=gdf['Participants']*20, alpha=0.6, label='Participants')
        
        for idx, row in gdf.iterrows():
            ax.annotate(text=row['State'], xy=(row['Longitude'], row['Latitude']), 
                        xytext=(3,3), textcoords="offset points", fontsize=8)
            
        ax.set_title("India Map - Statewise Participations")
        st.pyplot(fig)
    except Exception as e:
        st.warning(f"Could not load India Map: {e}. Rendering simple scatter plot instead.")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(state_agg['Longitude'], state_agg['Latitude'], s=state_agg['Participants']*20, c='red', alpha=0.6)
        for idx, row in gdf.iterrows():
            ax.annotate(text=row['State'], xy=(row['Longitude'], row['Latitude']), fontsize=8)
        st.pyplot(fig)
else:
    st.info("No map data available for the selected filters.")

st.markdown("---")

# 2. Participant Text Feedback and Ratings
st.header("2. Participant Text Feedback & Ratings")
col_fb1, col_fb2 = st.columns(2)

with col_fb1:
    st.subheader("Ratings Distribution")
    rating_counts = filtered_df['Rating'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(rating_counts.index.astype(str), rating_counts.values, color='coral')
    ax.set_xlabel("Rating", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title("Distribution of Ratings", fontsize=12)
    
    # Add count labels on top
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),  # 2 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    st.pyplot(fig)

with col_fb2:
    st.subheader("Text Processing: Common Words")
    all_feedback = " ".join(filtered_df['Feedback on Fest'].dropna())
    
    try:
        stop_words = set(stopwords.words('english'))
        # Add custom stopwords matching the context
        stop_words.update(['experience', 'session', 'event', 'needs', 'slight', 'very', 'good', 'well'])
        words = word_tokenize(all_feedback.lower())
        filtered_words = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 3]
    except Exception:
        # Fallback if NLTK tokenizer/stopwords failed loading
        words = re.findall(r'\b[a-z]+\b', all_feedback.lower())
        stop_words = {'and', 'the', 'is', 'in', 'it', 'to', 'of', 'for', 'on', 'with', 'very', 'good', 'well', 'needs', 'slight', 'experience', 'session', 'event'}
        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
        
    if filtered_words:
        word_counts = Counter(filtered_words).most_common(10)
        word_df = pd.DataFrame(word_counts, columns=['Word', 'Frequency'])
        
        # Horizontal Bar Chart
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(word_df['Word'], word_df['Frequency'], color='teal')
        ax.set_xlabel("Frequency")
        ax.set_title("Top Words in Feedback", fontsize=12)
        ax.invert_yaxis()
        st.pyplot(fig)
    else:
        st.info("Not enough textual data for frequency analysis.")

st.markdown("---")
# Automated Text Classification & Sentiment
st.header("3. Automated Text Classification (NLTK Sentiment)")
try:
    sia = SentimentIntensityAnalyzer()
    
    def classify_sentiment(text):
        if not isinstance(text, str):
            return 'Neutral'
        score = sia.polarity_scores(text)['compound']
        if score >= 0.05: return 'Positive'
        elif score <= -0.05: return 'Negative'
        else: return 'Neutral'
        
    filtered_df['Sentiment'] = filtered_df['Feedback on Fest'].apply(classify_sentiment)
    
    sent_col1, sent_col2 = st.columns(2)
    with sent_col1:
        st.subheader("Feedback Sentiment Distribution")
        sentiment_counts = filtered_df['Sentiment'].value_counts()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#90ee90', '#d3d3d3', '#ffcccb'])
        ax.axis('equal') 
        st.pyplot(fig)
        
    with sent_col2:
        st.subheader("Sentiment vs Subjective Rating")
        st.markdown('''
        * **Text Classification Model:** NLTK VADER Analyzer
        * Used to group feedback strings directly into Positive, Neutral, or Negative without relying on user ratings.
        ''')
        st.dataframe(filtered_df[['Feedback on Fest', 'Sentiment', 'Rating']].head(10), use_container_width=True, hide_index=True)
        
except Exception as e:
    st.info("Sentiment Analyzer not fully loaded yet or missing data.")

# Display raw feedback based on rating (Text Similarity/Classification subset filter)
st.subheader("Information Retrieval: Filter Feedback by Rating")
filter_rating = st.slider("Select Rating range", min_value=1, max_value=5, value=(1, 5))
filtered_feedback = filtered_df[(filtered_df['Rating'] >= filter_rating[0]) & (filtered_df['Rating'] <= filter_rating[1])]

st.dataframe(filtered_feedback[['Student Name', 'College', 'State', 'Event Name', 'Feedback on Fest', 'Rating']], use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>GATEWAYS-2025 Dashboard</p>", unsafe_allow_html=True)
