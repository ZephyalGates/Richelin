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
import re
from streamlit_extras.stylable_container import stylable_container
from huggingface_hub import hf_hub_download, list_repo_files
import requests
import base64
from st_screen_stats import ScreenData

# Main page
st.set_page_config(layout="wide", page_title="Richelin")

# Initialize screen data
screen_data = ScreenData()
screen_stats = screen_data.st_screen_data()
screen_width = screen_stats.get("innerWidth", 0) if screen_stats else 0

# Calculate scale factor
DESIGN_WIDTH = 1080  # Base design width
MIN_WIDTH = 320  # Minimum supported width
scale_factor = 1.0

if screen_width > 0:
    # Calculate scale factor with minimum threshold
    scale_factor = max(screen_width / DESIGN_WIDTH, MIN_WIDTH / DESIGN_WIDTH)
    # Cap maximum scale at 1.0 to prevent upscaling on large screens
    scale_factor = min(scale_factor, 1.0)

# Apply responsive scaling CSS with CSS Custom Properties (FIXED VERSION)
st.markdown(f"""
<style>
    /* Define CSS custom properties for scaling */
    :root {{
        --scale-factor: {scale_factor};
        --base-font-size: 2rem;
        --header-font-size: 3.5rem;
        --subheader-font-size: 3rem;
        --small-header-font-size: 2.75rem;
    }}
    
    /* Override Streamlit's mobile breakpoints */
    @media (max-width: 9999px) {{
        /* Force all columns to maintain their layout */
        [data-testid="column"] {{
            flex: 1 1 0% !important;
            width: auto !important;
            min-width: 0 !important;
        }}
        
        /* Prevent column stacking */
        [data-testid="column"]:nth-child(n) {{
            flex: 1 1 0% !important;
            width: auto !important;
        }}
        
        /* Override Streamlit's responsive grid */
        .row-widget.stHorizontalBlock {{
            flex-wrap: nowrap !important;
            gap: 0.5rem !important;
        }}
        
        /* Ensure columns stay horizontal */
        .stColumns {{
            flex-direction: row !important;
        }}
    }}
    
    /* Apply uniform scaling to the entire app */
    .stApp > div > div {{
        transform: scale(var(--scale-factor));
        transform-origin: top left;
        width: calc(100% / var(--scale-factor));
        overflow-x: hidden;
    }}
    
    /* Scale specific elements */
    .stApp {{
        overflow-x: hidden;
        max-width: 100vw;
    }}
    
    /* FIXED: Use CSS custom properties instead of !important for font sizes */
    * {{
        font-size: calc(var(--base-font-size) * var(--scale-factor));
    }}
    
    /* Now these headers can be properly overridden */
    h1 {{
        font-size: calc(var(--header-font-size) * var(--scale-factor));
    }}
    
    h2 {{
        font-size: calc(var(--subheader-font-size) * var(--scale-factor));
    }}
    
    h3 {{
        font-size: calc(var(--small-header-font-size) * var(--scale-factor));
    }}
    
    /* Scale metrics */
    [data-testid="metric-container"] {{
        transform: scale(var(--scale-factor));
        transform-origin: top left;
    }}
    
    /* Scale map container */
    .stDeckGlJsonChart {{
        transform: scale(var(--scale-factor));
        transform-origin: top left;
        width: calc(100% / var(--scale-factor)) !important;
    }}
    
    /* Scale matplotlib figures */
    .stImage, .stPlotlyChart {{
        transform: scale(var(--scale-factor));
        transform-origin: top left;
    }}
    
    /* Prevent horizontal scrolling */
    html, body {{
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }}
    
    /* Scale search container background */
    .stContainer > div {{
        background-size: cover !important;
    }}
    
    /* Ensure nested layouts maintain structure */
    [data-testid="stHorizontalBlock"] {{
        flex-wrap: nowrap !important;
        min-width: 0 !important;
    }}
    
    /* Force columns to stay in row */
    @media screen and (max-width: 640px) {{
        [data-testid="column"] {{
            flex: 1 1 0% !important;
            max-width: none !important;
        }}
        
        .row-widget.stHorizontalBlock {{
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
        }}
    }}
    
    /* Additional overrides for very small screens */
    @media screen and (max-width: 480px) {{
        /* Maintain multi-column layouts */
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
            flex: 1 !important;
            min-width: 0 !important;
            max-width: none !important;
        }}
        
        /* Scale padding and margins */
        .stApp > div > div {{
            padding: calc(1rem * var(--scale-factor)) !important;
        }}
        
        [data-testid="stVerticalBlock"] > div {{
            gap: calc(1rem * var(--scale-factor)) !important;
        }}
    }}
    
    /* Debug info styling */
    .debug-info {{
        position: fixed;
        top: 10px;
        right: 10px;
        background: rgba(0,0,0,0.8);
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-size: 12px !important;
        z-index: 9999;
    }}
    
    /* Fix for search container - prevent expansion */
    .search-container {{
        height: 600px !important;
        overflow: visible !important;
        position: relative;
    }}
    
    /* Ensure dropdown appears outside container */
    .stSearchbox {{
        position: relative;
        z-index: 1000;
    }}
    
    /* Make dropdown position absolute to prevent container expansion */
    .stSearchbox > div > div:last-child {{
        position: absolute !important;
        top: 100% !important;
        left: 0 !important;
        right: 0 !important;
        z-index: 1001 !important;
    }}
    
    /* FIXED: Custom header styling that now works properly */
    .custom-header {{
        color: white !important;
        font-size: calc(3rem * var(--scale-factor)) !important;
        font-weight: 700 !important;
        line-height: 1.2 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5) !important;
        margin-bottom: 1rem !important;
    }}
    
    /* Responsive custom header for smaller screens */
    @media screen and (max-width: 768px) {{
        .custom-header {{
            font-size: calc(2rem * var(--scale-factor)) !important;
        }}
    }}
    
    @media screen and (max-width: 480px) {{
        .custom-header {{
            font-size: calc(1.5rem * var(--scale-factor)) !important;
        }}
    }}
    
    /* Enhanced search section styling */
    .search-section {{
        position: relative;
        z-index: 1;
    }}
    
    /* Make sure markdown containers respect custom styling */
    [data-testid="stMarkdownContainer"] .custom-header {{
        color: white !important;
        font-size: calc(3rem * var(--scale-factor)) !important;
    }}
</style>
""", unsafe_allow_html=True)

# # Debug information (remove in production)
# if st.sidebar.checkbox("Show Debug Info", value=False):
#     st.markdown(f"""
#     <div class="debug-info">
#         Screen Width: {screen_width}px<br>
#         Scale Factor: {scale_factor:.2f}<br>
#         Effective Width: {screen_width/scale_factor:.0f}px
#     </div>
#     """, unsafe_allow_html=True)

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

# Wrap the search container in a fixed-height div
with stylable_container(key="test",
                        css_styles=f"""
                        {{
        background-image: url('data:image/png;base64,{search_bg_base64}');
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
        padding-top: 150px;
        padding-right: 100px;
        padding-bottom: 0px;
        padding-left: 50px;
        height: 600px !important;
        min-height: 600px !important;
        max-height: 600px !important;
        overflow: visible !important;
        position: relative;
        }}
    """):
    # Use regular columns
    cc1, cc2 = st.columns([5, 8])
    with cc1:
        sb_text = "Discover restaurant on OpenRice with Data Analysis"
        # FIXED: Now the custom header font size will work properly
        st.markdown(
            f"""
            <div class="custom-header">{sb_text}</div>
            """,
            unsafe_allow_html=True
        )
        # Create a container div that won't expand
        st.markdown("""
        <style>
            /* Ensure the searchbox container doesn't affect parent height */
            .search-wrapper {
                position: relative;
                height: 60px;
                overflow: visible !important;
            }
            
            /* Make dropdown overlay instead of pushing content */
            .stSearchbox {
                position: relative;
            }
            
            /* Style dropdown to appear outside container flow */
            div[data-baseweb="popover"] {
                position: fixed !important;
                z-index: 10000 !important;
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Wrap searchbox in a container
        search_container = st.container()
        with search_container:
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
    standard_dates['review_posted_date'] = pd.to_datetime(standard_dates['review_posted_date'], format='%Y-%m-%d', errors='coerce')
    latest_date = standard_dates['review_posted_date'].max()
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Restaurant info columns
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
    
    # Force 5-column layout for ratings
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

    # Process data for word clouds
    for col in item_quality_columns + other_sentiment_columns:
        restaurant_df[f'{col}_adjectives'] = restaurant_df[f'{col}_adjectives'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        restaurant_df[f'{col}_parsed'] = restaurant_df[f'{col}_parsed'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
    st.header("Review Highlights")
    
    def generate_wordcloud(frequencies, title):
        # Adjust figure size based on scale factor
        fig_width = max(10 * scale_factor, 5)
        fig_height = max(5 * scale_factor, 2.5)
        
        wordcloud = WordCloud(
            font_path='Arial_Unicode_MS.ttf', 
            width=int(800 * scale_factor), 
            height=int(400 * scale_factor), 
            background_color='white'
        ).generate_from_frequencies(frequencies)
        
        plt.figure(figsize=(fig_width, fig_height))
        plt.title(title, fontsize=50 * scale_factor)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        plt.close()

    def generate_wordcloud_2(frequencies, title):
        fig_width = max(10 * scale_factor, 5)
        fig_height = max(5 * scale_factor, 2.5)
        
        wordcloud = WordCloud(
            font_path='Arial_Unicode_MS.ttf', 
            width=int(800 * scale_factor), 
            height=int(400 * scale_factor), 
            background_color='white'
        ).generate(frequencies)
        
        plt.figure(figsize=(fig_width, fig_height))
        plt.title(title, fontsize=50 * scale_factor)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
        plt.close()

    # Word clouds with forced 3-column layout
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

    # Process restaurant data
    restaurant_df['recommended_dishes'] = restaurant_df['recommended_dishes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    def tokenize(text):
        return ' '.join(jieba.cut(text))

    # Extract all unique keys from the dictionaries
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
                for ld in kdict:
                    if isinstance(ld, dict):
                        for key in list(ld.keys()):
                            all_keys.append(key)

    all_keys = set(all_keys)
    tokenized_keys = [tokenize(key) for key in all_keys]

    # Compute text similarity between keys to group them
    vectorizer = TfidfVectorizer().fit_transform(tokenized_keys)
    similarity_matrix = cosine_similarity(vectorizer)

    # Create a mapping of keys based on high similarity
    threshold = 0.4
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
            similar_group.add(key)
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
            
            if filtered_groups == merged_keys:
                break
            
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
                if row[quality]:
                    quality_dict_list = row[quality]
                    if isinstance(quality_dict_list, list):
                        for quality_dict in quality_dict_list:
                            if isinstance(quality_dict, dict):
                                for key in quality_dict.keys():
                                    if len(quality_dict[key]) <= 0:
                                        continue
                                    key_group = find_group(key, filtered_groups)
                                    if key_group:
                                        if key_group == group:
                                            sentiment = quality.split('_')[0]
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
                # Display images in rows of 5 columns with forced layout
                images = row['images'][:20]
                num_images = len(images)
                cols = st.columns(5)
                for i in range(num_images):
                    with cols[i % 5]:
                        st.image(images[i])
                
                # Display word clouds with forced 3-column layout
                wccols = st.columns(3)
                
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