# Gateways 2025 - Interactive Dashboard

this is the dashboard for gateways fest 2025. it shows all the participation details, revenue stuff and feedback analysis from the dataset provided. 

built using streamlit and python for the ETE exam submission.

### features included
- interactive filters for event, college, and state
- 3d bar charts for participation and revenue (using matplotlib)
- india map plotted with geopandas (with fallback if geojson url fails)
- sentiment analysis of feedback (nltk vader)
- search function for feedback strings using jaccard similarity

### how to run locally

1. make sure you have python installed.
2. install the requirements:
`pip install -r requirements.txt`

3. run the app:
`streamlit run app.py`

### libraries used
streamlit, pandas, numpy, matplotlib, geopandas, shapely, nltk

---
Developed by Abhinav Jain
Registration Number: 2547203
Class: 3 MCA B
