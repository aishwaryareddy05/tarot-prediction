import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import joblib
import nltk
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('tarot_dataset.csv')
print(f"Dataset loaded with {len(df)} entries.")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['processed_query'], 
    df['category'], 
    test_size=0.2, 
    random_state=42
)

print(f"Training data size: {len(X_train)}")
print(f"Testing data size: {len(X_test)}")

# Create TF-IDF + SVM pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
    ('svm', SVC(kernel='linear', probability=True))
])

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("\nModel Evaluation:")
print(classification_report(y_test, y_pred))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Save the model
joblib.dump(model, 'tarot_prediction_model.pkl')
print("Model saved as 'tarot_prediction_model.pkl'")

# Function to preprocess a new query
def preprocess_query(query):
    # Reuse preprocessing function from data creation
    # Convert to lowercase
    text = query.lower()
    
    # Remove special characters and numbers
    import re
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Create a function that generates a tarot response
def generate_tarot_response(query):
    """
    Generate a tarot card reading response for a given query.
    """
    # Preprocess the query
    processed_query = preprocess_query(query)
    
    # Predict the category
    category = model.predict([processed_query])[0]
    
    # Get probability distribution
    category_probs = model.predict_proba([processed_query])[0]
    category_confidence = max(category_probs)
    
    print(f"Query: '{query}'")
    print(f"Predicted category: {category} (confidence: {category_confidence:.2f})")
    
    # Load tarot cards and templates (abbreviated version from the dataset creation)
    tarot_cards = [
        "The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor",
        "The Hierophant", "The Lovers", "The Chariot", "Strength", "The Hermit",
        "Wheel of Fortune", "Justice", "The Hanged Man", "Death", "Temperance",
        "The Devil", "The Tower", "The Star", "The Moon", "The Sun",
        "Judgment", "The World"
    ]
    
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
    card = random.choice(tarot_cards)
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
        'response': response
    }

# Test the model with unseen queries
test_queries = [
    "Will I get promoted at work?",
    "Is my relationship going to improve?",
    "How can I save more money?",
    "Will my health issue resolve soon?",
    "Should I pursue this new spiritual practice?",
    "Will I meet someone special this year?",
    "Should I invest in the stock market?",
    "Will I be successful in my new business venture?",
    "How can I improve my mental health?",
    "Am I on the right path in life?"
]

print("\n=== Model Testing with Unseen Queries ===")
test_results = []
for query in test_queries:
    result = generate_tarot_response(query)
    test_results.append(result)
    print(f"\nQuery: {result['query']}")
    print(f"Predicted Category: {result['category']}")
    print(f"Selected Card: {result['card']}")
    print(f"Generated Response: {result['response']}")
    print("-" * 80)

# Create a simple evaluation report
evaluation_df = pd.DataFrame(test_results)

# Save test results
evaluation_df.to_csv('tarot_model_evaluation.csv', index=False)
print("\nTest results saved to 'tarot_model_evaluation.csv'")