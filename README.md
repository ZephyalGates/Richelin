# Richelin - Restaurant Review Analysis App

A Streamlit web application that provides data-driven insights into OpenRice restaurant reviews using sentiment analysis and natural language processing.

## Features

- 🔍 **Restaurant Search**: Search from thousands of restaurants in the database
- 📊 **Sentiment Analysis**: View positive, neutral, and negative sentiments across multiple categories
- ☁️ **Word Clouds**: Visual representation of frequently mentioned terms in reviews
- 📈 **Rating Analytics**: Comprehensive rating breakdowns for taste, ambience, service, hygiene, and value
- 📱 **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- 🗄️ **HuggingFace Integration**: Restaurant data stored and retrieved from HuggingFace datasets

## Setup

### Prerequisites

- Python 3.8 or higher
- Streamlit account (for deployment)
- HuggingFace account and API token

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/richelin-app.git
cd richelin-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up Streamlit secrets:
   - Create `.streamlit/secrets.toml` in your local directory
   - Add your HuggingFace token:
   ```toml
   HUGGINGFACE_TOKEN = "your-huggingface-token-here"
   ```

### Running Locally

```bash
streamlit run app.py
```

## Deployment

### Deploy to Streamlit Cloud

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Set up secrets in Streamlit Cloud:
   - Go to App Settings > Secrets
   - Add your HuggingFace token

## Data Source

Restaurant review data is stored in HuggingFace dataset: `Johnson9/richelin`

## Technologies Used

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **WordCloud**: Generating word cloud visualizations
- **HuggingFace Hub**: Data storage and retrieval
- **Scikit-learn**: TF-IDF vectorization and similarity calculations
- **Matplotlib**: Data visualization
- **Jieba**: Chinese text segmentation

## License

This project is licensed under the MIT License.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.