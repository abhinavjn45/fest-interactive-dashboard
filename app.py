import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import geopandas as gpd
from shapely.geometry import Point
import re
from collections import Counter
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import ssl

# Fix SSL context for NLTK just in case
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except:
    pass

@st.cache_resource
def get_nltk():
    try:
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
    except:
        pass
get_nltk()

st.set_page_config(page_title="GATEWAYS-2025 Dashboard", layout="wide")

st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    h1 { color: #1e3d59; font-family: sans-serif; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("fest_dataset.csv")
    
    # Coordinates for mapping
    coords = {
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
    
    df['Longitude'] = df['State'].map(lambda x: coords.get(x, [None, None])[0])
    df['Latitude'] = df['State'].map(lambda x: coords.get(x, [None, None])[1])
    return df

try:
    data = load_data()
except Exception as e:
    st.error("Error loading data!")
    st.stop()

st.title("GATEWAYS-2025 National Level Fest Dashboard")
st.markdown("**Interactive analytics for participation and feedback insights.**")

# sidebar
st.sidebar.header("Filter Data")
sel_event = st.sidebar.multiselect("Select Event", options=data['Event Name'].unique(), default=data['Event Name'].unique())
sel_college = st.sidebar.multiselect("Select College", options=data['College'].unique(), default=data['College'].unique())
sel_state = st.sidebar.multiselect("Select State", options=data['State'].unique(), default=data['State'].unique())

st.sidebar.markdown("---")

df = data[
    (data['Event Name'].isin(sel_event)) & 
    (data['College'].isin(sel_college)) & 
    (data['State'].isin(sel_state))
].copy()

# top metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Participants", len(df), f"{len(df) - len(data)} total")
c2.metric("Events", df['Event Name'].nunique(), f"{df['Event Name'].nunique() - data['Event Name'].nunique()} total")
c3.metric("Colleges", df['College'].nunique())
avg_r = df['Rating'].mean()
c4.metric("Avg Rating", f"{avg_r:.2f} / 5", f"{avg_r - data['Rating'].mean():.2f} overall")
rev = df['Amount Paid'].sum()
c5.metric("Total Revenue", f"₹{rev:,}", f"₹{rev - data['Amount Paid'].sum():,} overall")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Overview & Geography", "Financial Analysis", "Text Analysis"])

with tab1:
    st.header("Participation & Geography")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Event-wise Participation (3D Bar)")
        evt_counts = df['Event Name'].value_counts()
        if len(evt_counts) > 0:
            fig = plt.figure(figsize=(6, 4))
            ax = fig.add_subplot(111, projection='3d')
            x = np.arange(len(evt_counts))
            y = np.zeros(len(evt_counts))
            z = np.zeros(len(evt_counts))
            dx = np.ones(len(evt_counts)) * 0.5
            dy = np.ones(len(evt_counts)) * 0.5
            dz = evt_counts.values
            
            ax.bar3d(x, y, z, dx, dy, dz, color='skyblue', shade=True)
            ax.set_xticks(x + 0.25)
            ax.set_xticklabels(evt_counts.index, rotation=45, ha='right')
            ax.set_yticks([])
            ax.set_zlabel("Participants")
            ax.set_title("Participation per Event")
            plt.tight_layout()
            st.pyplot(fig)
    
    with col2:
        st.subheader("Top 10 Colleges")
        col_counts = df['College'].value_counts().head(10)
        if len(col_counts) > 0:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(col_counts.values, labels=col_counts.index, autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1.colors)
            ax.set_title("Top 10 Colleges")
            ax.axis('equal') 
            st.pyplot(fig)

    st.markdown("---")
    st.subheader("3D Regional Participation by Rating")
    if not df.empty:
        agg_3d = df.groupby(['State', 'Rating']).size().reset_index(name='Count')
        if not agg_3d.empty:
            states = agg_3d['State'].unique()
            state_dict = {s: i for i, s in enumerate(states)}
            
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            x = agg_3d['State'].map(state_dict).values
            y = agg_3d['Rating'].values
            z = np.zeros(len(x))
            dx = np.ones(len(x)) * 0.4
            dy = np.ones(len(y)) * 0.4
            dz = agg_3d['Count'].values
            
            ax.bar3d(x, y, z, dx, dy, dz, color='skyblue', shade=True)
            ax.set_xticks(range(len(states)))
            ax.set_xticklabels(states, rotation=45, ha='right')
            ax.set_ylabel("Rating (1-5)")
            ax.set_zlabel("Participant Count")
            st.pyplot(fig)

    st.markdown("---")
    st.subheader("State-wise Participants in INDIA Map")
    state_counts = df.dropna(subset=['Longitude', 'Latitude'])
    if not state_counts.empty:
        agg = state_counts.groupby(['State', 'Longitude', 'Latitude']).size().reset_index(name='Participants')
        
        geometry = [Point(xy) for xy in zip(agg['Longitude'], agg['Latitude'])]
        gdf = gpd.GeoDataFrame(agg, geometry=geometry)
        
        try:
            url = "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson"
            india = gpd.read_file(url)
            
            fig, ax = plt.subplots(figsize=(10, 8))
            india.plot(ax=ax, color='lightgrey', edgecolor='black')
            gdf.plot(ax=ax, color='red', markersize=gdf['Participants']*20, alpha=0.6)
            
            for idx, row in gdf.iterrows():
                ax.annotate(text=row['State'], xy=(row['Longitude'], row['Latitude']), 
                            xytext=(3,3), textcoords="offset points", fontsize=8)
                
            ax.set_title("India Map - Statewise Participations")
            st.pyplot(fig)
        except Exception as e:
            st.warning("Could not load map. Showing scatter plot instead.")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(agg['Longitude'], agg['Latitude'], s=agg['Participants']*20, c='red', alpha=0.6)
            for idx, row in gdf.iterrows():
                ax.annotate(text=row['State'], xy=(row['Longitude'], row['Latitude']), fontsize=8)
            st.pyplot(fig)

with tab2:
    st.header("Revenue Insights")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Revenue by Event (3D Bar)")
        if not df.empty:
            rev = df.groupby('Event Name')['Amount Paid'].sum().sort_values(ascending=False)
            fig_rev = plt.figure(figsize=(6, 4))
            ax_rev = fig_rev.add_subplot(111, projection='3d')
            x = np.arange(len(rev))
            y = np.zeros(len(rev))
            z = np.zeros(len(rev))
            dx = np.ones(len(rev)) * 0.5
            dy = np.ones(len(rev)) * 0.5
            dz = rev.values
            
            ax_rev.bar3d(x, y, z, dx, dy, dz, color='gold', shade=True)
            ax_rev.set_xticks(x + 0.25)
            ax_rev.set_xticklabels(rev.index, rotation=45, ha='right')
            ax_rev.set_yticks([])
            ax_rev.set_zlabel("Revenue (₹)")
            ax_rev.set_title("Revenue Collected")
            plt.tight_layout()
            st.pyplot(fig_rev)
            
    with c2:
        st.subheader("Top 10 Colleges by Revenue")
        if not df.empty:
            rev_c = df.groupby('College')['Amount Paid'].sum().sort_values(ascending=False).head(10)
            fig_rev_c, ax_rev_c = plt.subplots(figsize=(6, 4))
            ax_rev_c.barh(rev_c.index, rev_c.values, color='mediumseagreen')
            ax_rev_c.set_xlabel("Revenue (₹)")
            ax_rev_c.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig_rev_c)
            
    st.subheader("Financial Data")
    with st.expander("View Raw Financial Data"):
        st.dataframe(df[['Student Name', 'College', 'Event Name', 'Amount Paid']], use_container_width=True, hide_index=True)
        csv_f = df[['Student Name', 'College', 'Event Name', 'Amount Paid']].to_csv(index=False).encode('utf-8')
        st.download_button("Download CSV", data=csv_f, file_name="fin_data.csv", mime="text/csv")

with tab3:
    st.header("Participant Feedback & Text Analytics")
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Ratings Distribution (3D)")
        r_counts = df['Rating'].value_counts().sort_index()
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111, projection='3d')
        x_vals = r_counts.index.values
        x = np.arange(len(x_vals))
        y = np.zeros(len(x_vals))
        z = np.zeros(len(x_vals))
        dx = np.ones(len(x_vals)) * 0.5
        dy = np.ones(len(x_vals)) * 0.5
        dz = r_counts.values
        
        ax.bar3d(x, y, z, dx, dy, dz, color='coral', shade=True)
        ax.set_xticks(x + 0.25)
        ax.set_xticklabels(x_vals)
        ax.set_yticks([])
        ax.set_zlabel("Count")
        st.pyplot(fig)
        
    with c2:
        st.subheader("Common Words Used")
        all_text = " ".join(df['Feedback on Fest'].dropna())
        
        try:
            sw = set(stopwords.words('english'))
            sw.update(['experience', 'session', 'event', 'needs', 'slight', 'very', 'good', 'well'])
            wlist = word_tokenize(all_text.lower())
            filt_words = [w for w in wlist if w.isalpha() and w not in sw and len(w) > 3]
        except:
            wlist = re.findall(r'\b[a-z]+\b', all_text.lower())
            sw = {'and', 'the', 'is', 'in', 'it', 'to', 'of', 'for', 'on', 'with', 'very', 'good', 'well', 'needs', 'slight', 'experience', 'session', 'event'}
            filt_words = [w for w in wlist if w not in sw and len(w) > 3]
            
        if filt_words:
            wc = Counter(filt_words).most_common(10)
            wdf = pd.DataFrame(wc, columns=['Word', 'Frequency'])
            
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.barh(wdf['Word'], wdf['Frequency'], color='teal')
            ax.invert_yaxis()
            st.pyplot(fig)
            
    st.markdown("---")
    st.subheader("Automated Sentiment Classification")
    try:
        sia = SentimentIntensityAnalyzer()
        
        def get_sentiment(text):
            if not isinstance(text, str): return 'Neutral'
            s = sia.polarity_scores(text)['compound']
            if s >= 0.05: return 'Positive'
            elif s <= -0.05: return 'Negative'
            else: return 'Neutral'
            
        df['Sentiment'] = df['Feedback on Fest'].apply(get_sentiment)
        
        sc1, sc2 = st.columns(2)
        with sc1:
            sent_counts = df['Sentiment'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.pie(sent_counts.values, labels=sent_counts.index, autopct='%1.1f%%', startangle=90, colors=['#90ee90', '#d3d3d3', '#ffcccb'])
            ax.axis('equal') 
            st.pyplot(fig)
            
        with sc2:
            st.markdown("Sentiment Data vs Rating")
            st.dataframe(df[['Feedback on Fest', 'Sentiment', 'Rating']].head(10), use_container_width=True, hide_index=True)
            
    except:
        st.info("Sentiment logic failed to load.")

    st.markdown("---")
    st.subheader("Text Similarity Search")
    
    q = st.text_input("Search Feedback:")
    if q:
        try:
            qw = set(word_tokenize(q.lower()))
        except:
            qw = set(re.findall(r'\b[a-z]+\b', q.lower()))
            
        def sim(text):
            if not isinstance(text, str): return 0.0
            try: tw = set(word_tokenize(text.lower()))
            except: tw = set(re.findall(r'\b[a-z]+\b', text.lower()))
            if not tw or not qw: return 0.0
            return len(qw.intersection(tw)) / len(qw.union(tw))
            
        df['Sim'] = df['Feedback on Fest'].apply(sim)
        res = df[df['Sim'] > 0].sort_values(by='Sim', ascending=False).head(5)
        
        if not res.empty:
            st.success("Found matches:")
            st.dataframe(res[['Student Name', 'College', 'Feedback on Fest']], use_container_width=True, hide_index=True)
        else:
            st.warning("No matches found.")

    st.subheader("Filter by Rating")
    r_filter = st.slider("Rating range", 1, 5, (1, 5))
    fb = df[(df['Rating'] >= r_filter[0]) & (df['Rating'] <= r_filter[1])]
    
    with st.expander("View Filtered Feedback"):
        st.dataframe(fb[['Student Name', 'College', 'Event Name', 'Feedback on Fest', 'Rating']], use_container_width=True, hide_index=True)
        csv_ff = fb.to_csv(index=False).encode('utf-8')
        st.download_button("Download Feedback Data", data=csv_ff, file_name="fb_data.csv", mime="text/csv")

st.markdown("---")
st.markdown("<p style='text-align: center; color: grey;'>GATEWAYS-2025 Dashboard<br>Abhinav Jain | Register Number: 2547203 | Class: 3 MCA B</p>", unsafe_allow_html=True)
