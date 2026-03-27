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

st.set_page_config(page_title="GATEWAYS-2025 Dashboard", layout="wide")

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

@st.cache_data
def load_data():
    df = pd.read_csv("fest_dataset.csv")
    
    state_coords = {
        'Kerala': [76.2711, 10.8505], 
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

st.title("GATEWAYS-2025 National Level Fest Dashboard")
st.markdown("**Interactive analytics for participation and feedback insights.**")

st.sidebar.header("Analytical Filters")
selected_event = st.sidebar.multiselect("Select Event", options=df['Event Name'].unique(), default=df['Event Name'].unique(), help="Filter the dashboard by specific events.")
selected_college = st.sidebar.multiselect("Select College", options=df['College'].unique(), default=df['College'].unique(), help="Filter to see metrics for specific participating colleges.")
selected_state = st.sidebar.multiselect("Select State", options=df['State'].unique(), default=df['State'].unique(), help="Filter geographically by state.")

st.sidebar.markdown("---")
st.sidebar.info("Use the filters above to dynamically update all charts and metrics!")

filtered_df = df[
    (df['Event Name'].isin(selected_event)) & 
    (df['College'].isin(selected_college)) & 
    (df['State'].isin(selected_state))
]

col1, col2, col3, col4, col5 = st.columns(5)
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
with col5:
    revenue = filtered_df['Amount Paid'].sum()
    total_rev = df['Amount Paid'].sum()
    st.metric("Total Revenue", f"₹{revenue:,}", f"₹{revenue - total_rev:,} vs overall")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Overview & Geography", "Financial Analysis", "Text Analysis"])

with tab1:
    st.header("1. Participation Trends & Geography")
    
    col_trend1, col_trend2 = st.columns(2)
    with col_trend1:
        st.subheader("Event-wise Participation")
        event_counts = filtered_df['Event Name'].value_counts()
        if not event_counts.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(event_counts.index, event_counts.values, color='skyblue')
            ax.set_ylabel("Participants", fontsize=10)
            ax.set_title("Participation per Event", fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No data available.")
    
    with col_trend2:
        st.subheader("College-wise Participation (Top 10)")
        college_counts = filtered_df['College'].value_counts().head(10)
        if not college_counts.empty:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(college_counts.values, labels=college_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
            ax.set_title("Top 10 Colleges", fontsize=12)
            ax.axis('equal') 
            st.pyplot(fig)
        else:
            st.info("No data available.")

    st.subheader("State-wise Participants in INDIA Map")
    st.markdown("Visualizing Geodata using GeoPandas & Matplotlib")
    state_counts = filtered_df.dropna(subset=['Longitude', 'Latitude'])
    if not state_counts.empty:
        state_agg = state_counts.groupby(['State', 'Longitude', 'Latitude']).size().reset_index(name='Participants')
        
        geometry = [Point(xy) for xy in zip(state_agg['Longitude'], state_agg['Latitude'])]
        gdf = gpd.GeoDataFrame(state_agg, geometry=geometry)
        
        try:
            url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"
            india = gpd.read_file(url)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            india.plot(ax=ax, color='lightgrey', edgecolor='black')
            
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

with tab2:
    st.header("2. Revenue & Financial Analytics")
    col_fin1, col_fin2 = st.columns(2)
    with col_fin1:
        st.subheader("Revenue by Event")
        if not filtered_df.empty:
            rev_event = filtered_df.groupby('Event Name')['Amount Paid'].sum().sort_values(ascending=False)
            fig_rev, ax_rev = plt.subplots(figsize=(6, 4))
            ax_rev.bar(rev_event.index, rev_event.values, color='gold', edgecolor='black')
            ax_rev.set_ylabel("Revenue (₹)")
            ax_rev.set_title("Total Revenue Collected per Event")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig_rev)
        else:
            st.info("No data available.")
            
    with col_fin2:
        st.subheader("Revenue by College (Top 10)")
        if not filtered_df.empty:
            rev_college = filtered_df.groupby('College')['Amount Paid'].sum().sort_values(ascending=False).head(10)
            fig_rev_col, ax_rev_col = plt.subplots(figsize=(6, 4))
            ax_rev_col.barh(rev_college.index, rev_college.values, color='mediumseagreen')
            ax_rev_col.set_xlabel("Revenue (₹)")
            ax_rev_col.set_title("Top 10 Colleges by Revenue")
            ax_rev_col.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig_rev_col)
        else:
            st.info("No data available.")
            
    st.subheader("Raw Financial Data Filtered")
    with st.expander("View Raw Financial Data"):
        st.dataframe(filtered_df[['Student Name', 'College', 'Event Name', 'Amount Paid']], use_container_width=True, hide_index=True)
        csv_fin = filtered_df[['Student Name', 'College', 'Event Name', 'Amount Paid']].to_csv(index=False).encode('utf-8')
        st.download_button("Download Financial Data (CSV)", data=csv_fin, file_name="financial_data.csv", mime="text/csv")

with tab3:
    st.header("3. Participant Text Feedback & Ratings")
    col_fb1, col_fb2 = st.columns(2)
    
    with col_fb1:
        st.subheader("Ratings Distribution")
        rating_counts = filtered_df['Rating'].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(rating_counts.index.astype(str), rating_counts.values, color='coral')
        ax.set_xlabel("Rating", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.set_title("Distribution of Ratings", fontsize=12)
        
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
            stop_words.update(['experience', 'session', 'event', 'needs', 'slight', 'very', 'good', 'well'])
            words = word_tokenize(all_feedback.lower())
            filtered_words = [w for w in words if w.isalpha() and w not in stop_words and len(w) > 3]
        except Exception:
            words = re.findall(r'\b[a-z]+\b', all_feedback.lower())
            stop_words = {'and', 'the', 'is', 'in', 'it', 'to', 'of', 'for', 'on', 'with', 'very', 'good', 'well', 'needs', 'slight', 'experience', 'session', 'event'}
            filtered_words = [w for w in words if w not in stop_words and len(w) > 3]
            
        if filtered_words:
            word_counts = Counter(filtered_words).most_common(10)
            word_df = pd.DataFrame(word_counts, columns=['Word', 'Frequency'])
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(word_df['Word'], word_df['Frequency'], color='teal')
            ax.set_xlabel("Frequency")
            ax.set_title("Top Words in Feedback", fontsize=12)
            ax.invert_yaxis()
            st.pyplot(fig)
        else:
            st.info("Not enough textual data for frequency analysis.")
            
    st.markdown("---")
    st.header("4. Automated Text Classification (NLTK Sentiment)")
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
            if not sentiment_counts.empty:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#90ee90', '#d3d3d3', '#ffcccb'])
                ax.axis('equal') 
                st.pyplot(fig)
            else:
                st.info("No sentiment data.")
            
        with sent_col2:
            st.subheader("Sentiment vs Subjective Rating")
            st.markdown('''
            * **Text Classification Model:** NLTK VADER Analyzer
            * Used to group feedback strings directly into Positive, Neutral, or Negative without relying on user ratings.
            ''')
            st.dataframe(filtered_df[['Feedback on Fest', 'Sentiment', 'Rating']].head(10), use_container_width=True, hide_index=True)
            
    except Exception as e:
        st.info("Sentiment Analyzer not fully loaded yet or missing data.")

    st.subheader("Information Retrieval: Filter Feedback by Rating")
    filter_rating = st.slider("Select Rating range", min_value=1, max_value=5, value=(1, 5))
    filtered_feedback = filtered_df[(filtered_df['Rating'] >= filter_rating[0]) & (filtered_df['Rating'] <= filter_rating[1])]
    
    with st.expander("View Filtered Feedback Data"):
        st.dataframe(filtered_feedback[['Student Name', 'College', 'State', 'Event Name', 'Feedback on Fest', 'Rating']], use_container_width=True, hide_index=True)
        csv_fb = filtered_feedback[['Student Name', 'College', 'State', 'Event Name', 'Feedback on Fest', 'Rating']].to_csv(index=False).encode('utf-8')
        st.download_button("Download Feedback Data (CSV)", data=csv_fb, file_name="feedback_data.csv", mime="text/csv")

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>GATEWAYS-2025 Dashboard</p>", unsafe_allow_html=True)
