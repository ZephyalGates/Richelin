import streamlit as st
import pandas as pd
from PIL import Image
from streamlit_searchbox import st_searchbox
from io import BytesIO
import threading
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os
import ast
import jieba
import Levenshtein
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import streamlit_nested_layout
import re
from streamlit_extras.stylable_container import stylable_container
from huggingface_hub import hf_hub_download, list_repo_files
import requests
import base64

# Main page
st.set_page_config(layout="wide", page_title="Richelin")

# Function to convert image to base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Load images as base64
header_image_base64 = get_base64_image("header_image.png")
search_bg_base64 = get_base64_image("search_bar_background.png")

header = st.container()
header.write(f"""<div class='header-container'><img src='data:image/png;base64,{header_image_base64}' alt='Header Image' style='width:350px;height:auto;'/></div>""", unsafe_allow_html=True)

# Custom CSS for the transparent header
st.markdown(
    """
<style>
    /* Non-sticky transparent header */
    .header-container {
        display: flex;
        align-items: center;
        background-color: transparent;
        padding: 1rem 0;
    }
    .header-container img {
        margin-right: 10px;
    }
</style>
    """,
    unsafe_allow_html=True
)

# HuggingFace configuration
HF_REPO_ID = "Johnson9/richelin"
HF_REPO_TYPE = "dataset"
HF_TOKEN = st.secrets["HUGGINGFACE_TOKEN"]

# Cache functions
@st.cache_data
def load_main_page_data():
    """Load the main page dataframe from local file (GitHub)"""
    try:
        return pd.read_csv('main_page_df.csv')
    except Exception as e:
        st.error(f"Error loading main data: {e}")
        return pd.DataFrame()

@st.cache_data
def get_available_restaurants():
    """Get list of available restaurants from HuggingFace"""
    try:
        files = list_repo_files(repo_id=HF_REPO_ID, repo_type=HF_REPO_TYPE, token=HF_TOKEN)
        restaurant_files = [f for f in files if f.startswith("data/") and f.endswith(".csv")]
        restaurants = [f.replace("data/", "").replace(".csv", "") for f in restaurant_files]
        return set(restaurants)
    except Exception as e:
        st.error(f"Error getting restaurant list: {e}")
        return set()

@st.cache_data
def load_restaurant_data(restaurant_name):
    """Load specific restaurant data from HuggingFace only"""
    try:
        with st.spinner(f"Loading {restaurant_name} data..."):
            file_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=f"data/{restaurant_name}.csv",
                repo_type=HF_REPO_TYPE,
                token=HF_TOKEN
            )
            return pd.read_csv(file_path)
    except Exception as e:
        st.error(f"Error loading restaurant data: {e}")
        return None

# Load data
data = load_main_page_data()
available_restaurants = get_available_restaurants()

# # Sidebar info
# if not data.empty:
#     if len(available_restaurants) > 0:
#         st.sidebar.success(f"✓ Main data loaded\n✓ Connected to HuggingFace\n📊 {len(available_restaurants)} restaurants available")
#     else:
#         st.sidebar.warning("⚠️ Could not connect to HuggingFace dataset")

item_quality_columns = [
    'positive_Item Quality', 'neutral_Item Quality', 'negative_Item Quality'
]
other_sentiment_columns = [
    'positive_Menu Variety', 'neutral_Menu Variety', 'negative_Menu Variety',
    'positive_Service', 'neutral_Service', 'negative_Service',
    'positive_Ambience', 'neutral_Ambience', 'negative_Ambience',
    'positive_Value', 'neutral_Value', 'negative_Value',
    'positive_Dietary Accommodations', 'neutral_Dietary Accommodations', 'negative_Dietary Accommodations',
    'positive_Location and Accessibility', 'neutral_Location and Accessibility', 'negative_Location and Accessibility',
    'positive_Cleanliness and Hygiene', 'neutral_Cleanliness and Hygiene', 'negative_Cleanliness and Hygiene',
    'positive_Portion Size', 'neutral_Portion Size', 'negative_Portion Size',
    'positive_Wait Time', 'neutral_Wait Time', 'negative_Wait Time',
    'positive_Experience', 'neutral_Experience', 'negative_Experience',
    'positive_Special Features', 'neutral_Special Features', 'negative_Special Features'
]



st.markdown("<div class='search-section'>", unsafe_allow_html=True)



# Implementing the autocomplete search bar using streamlit-searchbox
def search_restaurants(query):
    results = data[data['name'].str.contains(query, case=False, na=False)]
    return results['name'].tolist()

with stylable_container(key="test",
                        css_styles=f"""
                        {{
        background-image: url('data:image/png;base64,{search_bg_base64}');
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        padding-top: 150px; /* Adjust padding as needed */
        padding-right: 100px; /* Adjust padding as needed */
        padding-bottom: 0px; /* Adjust padding as needed */
        padding-left: 50px; /* Adjust padding as needed */
        }}
    """):
    cc1, cc2 = st.columns([5, 8])
    with cc1:
        sb_text = "Discover restaurant on OpenRice with Data Analysis"
        st.markdown(
            f"""
            <style>
            .custom-header {{
                color: white !important; /* Force white text color */
                padding: 10px;
                border-radius: 5px;
                padding-bottom: 50px;
            }}
            /* Override any theme colors */
            [data-testid="stMarkdown"] .custom-header {{
                color: white !important;
            }}
            </style>
            <h1 class="custom-header">{sb_text}</h1>
            """,
            unsafe_allow_html=True
        )
        with stylable_container(key="test2",
                                css_styles="""
                                {
            width: 100%;
            min-height: 400px;
            }
        """):
            selected_restaurant = st_searchbox(
                search_function=search_restaurants,
                placeholder="Search for restaurant",
                key="restaurant_searchbox"

        )

def is_standard_date_format(date_str):
    try:
        pd.to_datetime(date_str, format='%Y-%m-%d', errors='raise')
        return True
    except:
        return False



if selected_restaurant:
    st.session_state.selected_restaurant = selected_restaurant
    # st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

if 'selected_restaurant' in st.session_state:
    
    restaurant_name = st.session_state.selected_restaurant
    restaurant = data[data['name'] == restaurant_name].iloc[0]
    
    # Check availability
    if restaurant_name not in available_restaurants:
        st.error(f"Restaurant '{restaurant_name}' data not found in HuggingFace dataset")
        st.info("This restaurant's data may not have been uploaded yet.")
        st.stop()
    
    restaurant_df = load_restaurant_data(restaurant_name)
    if restaurant_df is None:
        st.error(f"Failed to load data for '{restaurant_name}'")
        st.stop()
    
    # Filter out non-standard date formats
    standard_dates = restaurant_df[restaurant_df['review_posted_date'].apply(is_standard_date_format)]

    # Convert the filtered dates to pd.Timestamp
    standard_dates['review_posted_date'] = pd.to_datetime(standard_dates['review_posted_date'], format='%Y-%m-%d', errors='coerce')

    # Get the latest date
    latest_date = standard_dates['review_posted_date'].max()
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    mcol1, mcol2 = st.columns(2)
    
    
    
    with mcol1:
        st.header(restaurant['name'])
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader(f"Location: {restaurant['location']}")
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(f"Price Range: {restaurant['price_range'][1:]}")
        st.markdown("<br>", unsafe_allow_html=True)
        cleaned_ct = re.sub(r'[\[\]\'"]', '', restaurant['cuisine_types'])
        st.subheader(f"Cuisine Type: {cleaned_ct}")
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(f"Last update: {latest_date}")
        
    
    with mcol2:
        st.map(pd.DataFrame([[restaurant['latitude'], restaurant['longitude']]], columns=['latitude', 'longitude']))
    
    st.subheader("Overall Ratings")
    st.markdown("<br>", unsafe_allow_html=True)
    rcol1, rcol2, rcol3, rcol4, rcol5 = st.columns(5)
    rcol1.metric("Taste", f"{restaurant_df['味道'].mean():.3g}", delta=None)
    rcol1.write(f"By {restaurant_df['味道'].count()} reviewers")
    rcol2.metric("Ambience", f"{restaurant_df['環境'].mean():.3g}", delta=None)
    rcol2.write(f"By {restaurant_df['環境'].count()} reviewers")
    rcol3.metric("Service", f"{restaurant_df['服務'].mean():.3g}", delta=None)
    rcol3.write(f"By {restaurant_df['服務'].count()} reviewers")
    rcol4.metric("Hygiene", f"{restaurant_df['衛生'].mean():.3g}", delta=None)
    rcol4.write(f"By {restaurant_df['衛生'].count()} reviewers")
    rcol5.metric("Value", f"{restaurant_df['抵食'].mean():.3g}", delta=None)
    rcol5.write(f"By {restaurant_df['抵食'].count()} reviewers")
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Openrice Ratings
    for col in item_quality_columns + other_sentiment_columns:
        restaurant_df[f'{col}_adjectives'] = restaurant_df[f'{col}_adjectives'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        restaurant_df[f'{col}_parsed'] = restaurant_df[f'{col}_parsed'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
    st.header("Review Highlights")
    def generate_wordcloud(frequencies, title):
        wordcloud = WordCloud(font_path='Arial_Unicode_MS.ttf', width=800, height=400, background_color='white').generate_from_frequencies(frequencies)
        plt.figure(figsize=(10, 5))
        plt.title(title)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        plt.close()

    def generate_wordcloud_2(frequencies, title):
        wordcloud = WordCloud(font_path='Arial_Unicode_MS.ttf', width=800, height=400, background_color='white').generate(frequencies)
        plt.figure(figsize=(10, 5))
        plt.title(title)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        plt.close()

    for sentiment in ['positive', 'neutral', 'negative']:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader(f'Word Clouds for {sentiment.capitalize()} Sentiment')
        col_count = 0
        row_container = st.container()
    
        for col in item_quality_columns + other_sentiment_columns:
            if sentiment in col:
                col_frequencies = restaurant_df[f'{col}_adjectives'].explode().value_counts()
                if len(col_frequencies) > 0:
                    if col_count % 3 == 0:
                        cols = row_container.columns(3)
                    with cols[col_count % 3]:
                        generate_wordcloud(col_frequencies, col.split('_')[1] + f" from {restaurant_df[col].count()} reviews")
                    col_count += 1
                else:
                    if col_count % 3 == 0:
                        cols = row_container.columns(3)
                    with cols[col_count % 3]:
                        st.write(f"No Data for {col.split('_')[1]}")
                    col_count += 1
            
            if col_count >= 12:
                break
        
    def flatten_and_join(adjectives_list):
        flat_list = [item for sublist in adjectives_list for item in sublist]
        return ' '.join(flat_list)




    restaurant_df['recommended_dishes'] = restaurant_df['recommended_dishes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    def tokenize(text):
        return ' '.join(jieba.cut(text))

    # Step 1: Extract all unique keys from the dictionaries
    all_keys = []
    for column in ['recommended_dishes', ]:
        for klist in restaurant_df[column]:
            if klist:
                if len(klist) > 0:
                    for key in klist:
                        all_keys.append(key)

    for column in ['positive_Item Quality_parsed', 'neutral_Item Quality_parsed', 'negative_Item Quality_parsed']:
        for kdict in restaurant_df[column]:
            if isinstance(kdict, list):
                # print(kdict)
                for ld in kdict:
                    if isinstance(ld, dict):
                        for key in list(ld.keys()):
                            all_keys.append(key)

    all_keys = set(all_keys)
    # Tokenize the keys
    tokenized_keys = [tokenize(key) for key in all_keys]

    # Step 2: Compute text similarity between keys to group them
    vectorizer = TfidfVectorizer().fit_transform(tokenized_keys)
    similarity_matrix = cosine_similarity(vectorizer)

    # Step 3: Create a mapping of keys based on high similarity
    threshold = 0.4  # Adjust the threshold as needed
    key_map = defaultdict(set)

    keys_list = list(all_keys)
    for i, key in enumerate(keys_list):
        for j, similar_key in enumerate(keys_list):
            if i != j and similarity_matrix[i, j] > threshold:
                key_map[key].add(similar_key)
                key_map[similar_key].add(key)

    # Normalize the mapping to group similar keys together
    grouped_keys = []
    visited = set()

    for key in keys_list:
        if key not in visited:
            similar_group = set(key_map[key])
            similar_group.add(key)  # Add the key itself to the group
            for similar_key in similar_group:
                visited.add(similar_key)
            grouped_keys.append(similar_group)
            
    def merge_groups(grouped_keys, merge_threshold=0.5):
        merged_groups = []
        while grouped_keys:
            group = grouped_keys.pop(0)
            merge_candidates = []
            for other_group in grouped_keys:
                match_count = sum(1 for elem in group if any(Levenshtein.ratio(elem, other_elem) > merge_threshold for other_elem in other_group))
                if match_count / len(group) > merge_threshold:
                    merge_candidates.append(other_group)
            
            for candidate in merge_candidates:
                grouped_keys.remove(candidate)
                group.update(candidate)
            
            merged_groups.append(group)
        return merged_groups

    merged_keys = merge_groups(grouped_keys, merge_threshold=0.4)

    def filter_groups(merged_keys, vectorizer, max_iterations=1):
        for _ in range(max_iterations):
            filtered_groups = []
            for group in merged_keys:
                if len(group) <= 5:
                    continue
                group_list = list(group)
                group_tokenized_keys = [tokenize(key) for key in group_list]
                group_tfidf_matrix = TfidfVectorizer().fit_transform(group_tokenized_keys)
                group_sim_matrix = cosine_similarity(group_tfidf_matrix)
                group_mean_sim = group_sim_matrix.mean()
                
                filtered_group = set()
                for i, key in enumerate(group_list):
                    if group_sim_matrix[i].mean() >= group_mean_sim:
                        filtered_group.add(key)
                
                if filtered_group:
                    filtered_groups.append(filtered_group)
            
            # Check for convergence: if filtered groups are the same as previous iteration
            if filtered_groups == merged_keys:
                break
            
            # Update merged_keys with filtered groups for next iteration
            merged_keys = filtered_groups
        
        return merged_keys

    filtered_groups = filter_groups(merged_keys, vectorizer)

    filtered_groups = merge_groups(filtered_groups, merge_threshold=0.7)       

    restaurant_df['image_data'] = restaurant_df['image_data'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Function to match keys in merged_keys to the caption
    def match_keys_in_caption(caption, merged_keys):
        matched_keys = set()
        for group in merged_keys:
            for key in group:
                if key in caption:
                    matched_keys.update(group)
                    break
        return matched_keys

    def find_group(key, merged_keys):
        for group in merged_keys:
            if key in group:
                return group
        return None



    results = []

    for group in filtered_groups:
        if len(group) <= 3:
            continue
        # print(f"group: {group}")
        group_dict = {
            'group': ', '.join(group),
            'appears_in_recommended': any(key in restaurant_df['recommended_dishes'].explode().values for key in group),
            'frequency': 0,
            'images': [],
            'positive_adjectives': [],
            'neutral_adjectives': [],
            'negative_adjectives': []
        }
        
        for idx, row in restaurant_df.iterrows():
            group_appears = False
            for img_data in row['image_data']:
                caption = img_data['caption']
                matched_food_items = match_keys_in_caption(caption, filtered_groups)
                if any(item in group for item in matched_food_items):
                    group_dict['images'].append(img_data['link'])
                    group_appears = True

            if group_appears:
                group_dict['frequency'] += 1

            for quality in ['positive_Item Quality_parsed', 'neutral_Item Quality_parsed', 'negative_Item Quality_parsed']:
                if row[quality]:  # Ensure the column is not empty
                    quality_dict_list = row[quality]  # Accessing the dictionary in the list
                    if isinstance(quality_dict_list, list):
                        for quality_dict in quality_dict_list:
                            if isinstance(quality_dict, dict):
                                # print(quality_dict)
                                for key in quality_dict.keys():
                                    if len(quality_dict[key]) <= 0:
                                        continue
                                    key_group = find_group(key, filtered_groups)
                                    # print(key_group)
                                    if key_group:
                                        if key_group == group:
                                            sentiment = quality.split('_')[0]
                                    #         print(quality_dict[key])
                                            # adjective_entry = {'adjective': quality_dict[key], 'row_index': idx}
                                            if sentiment == 'positive':
                                                group_dict['positive_adjectives'].append(quality_dict[key])
                                            elif sentiment == 'neutral':
                                                group_dict['neutral_adjectives'].append(quality_dict[key])
                                            elif sentiment == 'negative':
                                                group_dict['negative_adjectives'].append(quality_dict[key])

        results.append(group_dict)

    # Convert results to a DataFrame
    result_df = pd.DataFrame(results)
    result_df['positive_adjectives_flat'] = result_df['positive_adjectives'].apply(flatten_and_join)
    result_df['neutral_adjectives_flat'] = result_df['neutral_adjectives'].apply(flatten_and_join)
    result_df['negative_adjectives_flat'] = result_df['negative_adjectives'].apply(flatten_and_join)
    st.header("Food Items Highlight")
    for index, row in result_df.iterrows():
        text1 = row['positive_adjectives_flat']
        text2 = row['neutral_adjectives_flat']
        text3 = row['negative_adjectives_flat']
        if len(text1) > 0 or len(text2) > 0 or len(text3) > 0:
            with st.expander(row['group'].split(',')[0]):
                # Display images in rows of 5 columns
                images = row['images'][:20]
                num_images = len(images)
                cols = st.columns(5)
                for i in range(num_images):
                    with cols[i % 5]:
                        st.image(images[i])
                wccols = st.columns(3)
                # Display word clouds
                with wccols[0]:
                    if len(text1) > 0:
                        generate_wordcloud_2(text1, "Positive Adjectives")
                    else:
                        st.write("No data for Positive Adjectives")
                with wccols[1]:
                    if len(text2) > 0:
                        generate_wordcloud_2(text2, "Neutral Adjectives")
                    else:
                        st.write("No data for Neutral Adjectives")
                with wccols[2]:
                    if len(text3) > 0:
                        generate_wordcloud_2(text3, "Negative Adjectives")
                    else:
                        st.write("No data for Negative Adjectives")




    # Placeholder for further sections (Review Highlights, Signature Dishes, Polarized Reviews)
    
    # st.write("Wordclouds here")

    # st.subheader("Signature Dishes")
    # st.write("Dish details here")

    # st.subheader("Polarized Reviews")
    # st.write("Review details here")