import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load data and compute similarity matrix
@st.cache_data
def load_data_and_similarity():
    df = pd.read_csv("data/fashion_products.csv")
    
    # Create a feature text combination for the similarity engine
    # Adjust columns here if your text-matching uses something else
    feature_cols = ['Brand', 'Category', 'Color']
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].fillna('')
            
    # Combine features into a single string per row
    df['combined_features'] = df['Category'] + " " + df['Brand'] + " " + df['Color']
    
    # Compute the Cosine Similarity Matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['combined_features'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    return df, cosine_sim

try:
    df, cosine_sim = load_data_and_similarity()
except FileNotFoundError:
    st.error("Could not find 'fashion_products.csv' inside the 'data' folder.")
    st.stop()

# 2. Your Exact Recommendation Logic
def recommend(product_name):
    idx = df[df['Product Name'] == product_name].index[0]

    # Similarity scores
    similarity_scores = list(enumerate(cosine_sim[idx]))

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )
    similarity_scores = similarity_scores[1:11]

    product_indices = [i[0] for i in similarity_scores]

    similar_products = df.iloc[product_indices].copy()
    similar_products['Rating'] = (
        similar_products['Rating']
        .round()
        .astype(int)
    )

    similar_products = similar_products.sort_values(
        by='Rating',
        ascending=False
    )

    return similar_products[
        [
            'Product Name',
            'Brand',
            'Category',
            'Color',
            'Size',
            'Rating',
            'Price'
        ]
    ].head(5)

# 3. Streamlit UI
st.set_page_config(page_title="Fashion Recs", layout="centered")
st.title(" Fashion Product Recommendation Engine")

product_list = sorted(df['Product Name'].unique())
product = st.selectbox("Choose a Product:", product_list)

if st.button("Generate Recommendations", type="primary"):
    with st.spinner("Calculating similarity matrix..."):
        recs = recommend(product)
        
        st.subheader(" Top 5 Highest-Rated Recommendations for You:")
        st.dataframe(recs.reset_index(drop=True), use_container_width=True)
