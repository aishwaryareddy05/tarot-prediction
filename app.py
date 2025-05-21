import pandas as pd
import random
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import streamlit as st

import nltk

# Attempt to download the necessary NLTK resources
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')


# NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Tarot cards
tarot_cards = [
    "Random", "The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor",
    "The Hierophant", "The Lovers", "The Chariot", "Strength", "The Hermit",
    "Wheel of Fortune", "Justice", "The Hanged Man", "Death", "Temperance",
    "The Devil", "The Tower", "The Star", "The Moon", "The Sun",
    "Judgment", "The World"
]

# Preprocessing function for text data
def preprocess_query_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters and numbers
    tokens = text.split()  # Tokenize the text
    stop_words = set(stopwords.words('english'))  # Remove stopwords
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()  # Lemmatize the tokens
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(tokens)  # Return preprocessed text as a single string

# Named preprocessing function for FunctionTransformer
def preprocess_texts(texts):
    return [preprocess_query_text(t) for t in texts]

# Define the pipeline
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer

# Create a pipeline with preprocessing, TF-IDF, and SVC
pipeline = Pipeline([
    ('preprocess', FunctionTransformer(lambda x: preprocess_texts(x), validate=False)),
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('svm', SVC(probability=True))
])

# Load the model
@st.cache_resource
def load_model():
    return joblib.load('tarot_model.pkl')  # Make sure this path matches the location

model = load_model()

# Function to generate tarot response
def generate_tarot_response(query, selected_card="Random"):
    # Predict the category using the loaded model
    category = model.predict([query])[0]
    
    # Get the probability distribution
    category_probs = model.predict_proba([query])[0]
    category_confidence = max(category_probs)

    # Select random card if "Random" is chosen
    card = random.choice(tarot_cards[1:]) if selected_card == "Random" else selected_card

    # Response templates and content for categories
    templates = {
        'Love': [
            "The {card} reveals that {sentiment} in your love life. This suggests that {prediction}. {advice}.",
            "Your love situation is reflected in the {card}. It indicates {prediction}. Consider {advice}.",
            "Drawing the {card} for your question about love shows that {prediction}. {advice}.",
            "The energy of the {card} surrounds your romantic situation. {prediction}. {advice}."
        ],
        'Career': [
            "In your career path, the {card} appears, suggesting {prediction}. {advice}.",
            "The {card} indicates that professionally, {prediction}. {advice}.",
            "Your work situation is influenced by the {card}. This means {prediction}. {advice}.",
            "Drawing the {card} for your career question reveals that {prediction}. Now is the time to {advice}."
        ],
        'Finance': [
            "Regarding your finances, the {card} suggests that {prediction}. {advice}.",
            "The {card} appears in your financial reading, indicating {prediction}. Consider {advice}.",
            "Your financial situation is reflected by the {card}, showing that {prediction}. {advice}.",
            "The energy of the {card} influences your financial path. {prediction}. {advice}."
        ],
        'Health': [
            "For your health concerns, the {card} indicates that {prediction}. {advice}.",
            "The {card} appears in your health reading, suggesting {prediction}. {advice}.",
            "Your wellbeing is influenced by the {card}, which reveals {prediction}. {advice}.",
            "Drawing the {card} for your health question shows that {prediction}. {advice}."
        ],
        'Personal Growth': [
            "In your personal journey, the {card} emerges, indicating that {prediction}. {advice}.",
            "The {card} reflects your inner development, suggesting {prediction}. {advice}.",
            "Your spiritual path is illuminated by the {card}, showing that {prediction}. {advice}.",
            "The energy of the {card} guides your personal growth. {prediction}. {advice}."
        ]
    }

    # Content for each category
    category_content = {
        'Love': {
            'sentiments': [
                "there's a powerful transformation coming",
                "patience is needed",
                "new beginnings are on the horizon",
                "deeper connections are forming",
                "clarity is emerging from confusion"
            ],
            'predictions': [
                "a new relationship may enter your life soon",
                "existing bonds will strengthen through honest communication",
                "you may need to release past hurts to move forward",
                "unexpected romantic opportunities will appear",
                "you're entering a period of emotional healing and growth"
            ],
            'advice': [
                "open your heart to new possibilities",
                "communicate your needs more clearly",
                "trust your intuition about this connection",
                "give yourself time to heal before moving forward",
                "focus on self-love as the foundation for relationship success"
            ]
        },
        'Career': {
            'sentiments': [
                "significant changes are approaching",
                "your efforts are being recognized",
                "a period of assessment is necessary",
                "new opportunities are emerging",
                "patience with the current situation is important"
            ],
            'predictions': [
                "a leadership opportunity will soon present itself",
                "your creative approach will lead to recognition",
                "collaboration with others will bring unexpected success",
                "a challenging project will ultimately benefit your growth",
                "a period of evaluation will lead to better career alignment"
            ],
            'advice': [
                "pursue additional training or education",
                "network with those in positions you aspire to",
                "maintain balance between ambition and wellbeing",
                "trust your unique skills and perspective",
                "be open to paths you hadn't previously considered"
            ]
        },
        'Finance': {
            'sentiments': [
                "caution with resources is advised",
                "positive developments are approaching",
                "a reassessment of priorities is needed",
                "balance is returning to your situation",
                "patience with current limitations is important"
            ],
            'predictions': [
                "a new source of income may soon emerge",
                "careful planning now will lead to future stability",
                "unexpected expenses may require attention",
                "your financial discipline will be rewarded",
                "a cycle of scarcity is coming to an end"
            ],
            'advice': [
                "create a detailed budget for the coming months",
                "seek advice from a financial professional",
                "balance saving with allowing yourself small pleasures",
                "be cautious with investments at this time",
                "focus on building multiple streams of income"
            ]
        },
        'Health': {
            'sentiments': [
                "balance is key to your wellbeing",
                "attention to subtle signs is important",
                "a holistic approach is needed",
                "mental and physical connection is highlighted",
                "a period of restoration is beginning"
            ],
            'predictions': [
                "addressing stress will improve overall health",
                "a new approach to wellness will bring positive results",
                "listening to your body's signals will prevent future issues",
                "small, consistent changes will have significant impact",
                "connecting mind and body practices will accelerate healing"
            ],
            'advice': [
                "incorporate more mindfulness into your daily routine",
                "seek balance between rest and activity",
                "consider getting a second opinion about ongoing concerns",
                "prioritize quality sleep and nutrition",
                "find physical activities that bring you joy rather than obligation"
            ]
        },
        'Personal Growth': {
            'sentiments': [
                "inner wisdom is awakening",
                "a period of transformation has begun",
                "self-reflection will lead to important insights",
                "releasing old patterns is now possible",
                "alignment with your true purpose is emerging"
            ],
            'predictions': [
                "a deeper understanding of yourself is developing",
                "challenges ahead will ultimately strengthen your resolve",
                "spiritual insights will offer new perspectives",
                "old limitations are dissolving as you embrace authenticity",
                "connections with like-minded individuals will accelerate your growth"
            ],
            'advice': [
                "maintain a regular journaling practice",
                "explore meditation or contemplative practices",
                "seek learning opportunities that expand your worldview",
                "create boundaries that honor your authentic self",
                "practice gratitude to enhance awareness of life's gifts"
            ]
        }
    }

    # Select random components for response
    template = random.choice(templates[category])
    sentiment = random.choice(category_content[category]['sentiments'])
    prediction = random.choice(category_content[category]['predictions'])
    advice = random.choice(category_content[category]['advice'])

    # Generate response
    response = template.format(card=card, sentiment=sentiment, prediction=prediction, advice=advice)

    return {
        'query': query,
        'category': category,
        'card': card,
        'confidence': f"{category_confidence:.1%}",
        'response': response
    }


# Streamlit UI
def main():
     
    # Custom CSS for styling
    st.markdown("""
        <style>
            .stTextInput input {
                font-size: 18px;
                padding: 10px;
            }
            .stSelectbox select {
                font-size: 18px;
                padding: 8px;
            }
            .stButton button {
                font-size: 18px;
                padding: 10px 20px;
                background-color: #6a0dad;
                color: white;
                border: none;
                border-radius: 5px;
            }
            .stButton button:hover {
                background-color: black;
            }
            .result-box {
                padding: 20px;
                border-radius: 10px;
                background-color:  #5a0c9d;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                margin-top: 20px;
            }
            .card-highlight {
                color: #6a0dad;
                font-weight: bold;
            }
            .category-highlight {
                color: #d63384;
                font-weight: bold;
            }
            .confidence-highlight {
                color: #0d6efd;
                font-weight: bold;
            }
        </style>
    """, unsafe_allow_html=True)

    # App header
    st.title("🔮 Tarot Reading App")
    st.markdown("""
        Ask your question and receive guidance from the tarot cards. 
        You can let the cards choose a random card for you or select one specifically.
    """)
    
    # User input
    col1, col2 = st.columns([3, 1])
    with col1:
        user_query = st.text_input("What would you like guidance on?", 
                                  placeholder="e.g. Will I find love soon? Should I change careers?")
    with col2:
        selected_card = st.selectbox("Choose a card", tarot_cards, index=0)
    
    # Generate reading button
    if st.button("Get My Reading", use_container_width=True):
        if user_query.strip() == "":
            st.warning("Please enter your question first.")
        else:
            with st.spinner("Consulting the cards..."):
                result = generate_tarot_response(user_query, selected_card)
                
                # Display results
                st.markdown(f"""
                    <div class="result-box">
                        <h3>Your Reading</h3>
                        <p><strong>Your Question:</strong> {result['query']}</p>
                        <p><strong>Selected Card:</strong> <span class="card-highlight">{result['card']}</span></p>
                        <p><strong>Category:</strong> <span class="category-highlight">{result['category']}</span> 
                        (confidence: <span class="confidence-highlight">{result['confidence']}</span>)</p>
                        <hr>
                        <p><em>{result['response']}</em></p>
                    </div>
                """, unsafe_allow_html=True)

    # Information section
    st.markdown("---")
    st.subheader("About This App")
    st.markdown("""
        This app uses a machine learning model trained on tarot readings to provide personalized guidance. 
        The model analyzes your question to determine the most relevant category (Love, Career, Finance, Health, or Personal Growth) 
        and generates an appropriate response based on traditional tarot interpretations.
        
        - **Random Card**: Let the universe choose the most appropriate card for your question
        - **Specific Card**: Select a particular card you'd like insight from
    """)
    # Your UI setup and the rest of the application remains the same...

if __name__ == "__main__":
    main()
