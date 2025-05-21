import pandas as pd
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Define categories and response templates
categories = ['Love', 'Career', 'Finance', 'Health', 'Personal Growth']

# Dictionary of templates for each category
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

# Tarot cards
tarot_cards = [
    "The Fool", "The Magician", "The High Priestess", "The Empress", "The Emperor",
    "The Hierophant", "The Lovers", "The Chariot", "Strength", "The Hermit",
    "Wheel of Fortune", "Justice", "The Hanged Man", "Death", "Temperance",
    "The Devil", "The Tower", "The Star", "The Moon", "The Sun",
    "Judgment", "The World", "Ace of Cups", "Two of Cups", "Three of Cups",
    "Four of Cups", "Five of Cups", "Six of Cups", "Seven of Cups", "Eight of Cups",
    "Nine of Cups", "Ten of Cups", "Page of Cups", "Knight of Cups", "Queen of Cups",
    "King of Cups", "Ace of Wands", "Two of Wands", "Three of Wands", "Four of Wands"
]

# Sentiments, predictions, and advice for each category
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

# Common user queries for each category
queries = {
    'Love': [
        "Will I find love soon?",
        "Is my current relationship going to last?",
        "When will I meet my soulmate?",
        "Should I give my ex another chance?",
        "Will getting back with my ex work out?",
        "How can I attract love into my life?",
        "Is my crush interested in me?",
        "Will I marry my current partner?",
        "How can I improve my relationship?",
        "Is there someone better for me out there?",
        "Will I find true love this year?",
        "Is my partner being honest with me?",
        "How can I heal from my breakup?",
        "Will my long-distance relationship work out?",
        "Should I confess my feelings to my friend?",
        "Is my marriage going to improve?",
        "Will I have a romantic connection with someone new soon?",
        "How can I rebuild trust in my relationship?",
        "Will my relationship survive this difficult period?",
        "Is my partner the right one for me?",
        "When will I know I've found the right person?",
        "How can I move on from unrequited love?",
        "Will my relationship evolve into something more serious?",
        "Should I stay in my current relationship or move on?",
        "How will my love life change in the coming months?"
    ],
    'Career': [
        "Will I get the job I applied for?",
        "Should I change careers?",
        "Will I receive the promotion I'm hoping for?",
        "Is starting my own business a good idea?",
        "Should I accept the new job offer?",
        "Will my current job situation improve?",
        "What career path should I pursue?",
        "Will my business be successful?",
        "Should I ask for a raise now?",
        "Will I find job satisfaction soon?",
        "Is my current workplace right for me?",
        "How can I advance in my career?",
        "Will I find a new job soon?",
        "Should I pursue further education for my career?",
        "Will my project be successful?",
        "Is my business partner trustworthy?",
        "How can I improve my work-life balance?",
        "Will I find fulfillment in my current career path?",
        "Should I relocate for my career?",
        "Will my interview go well tomorrow?",
        "How can I stand out in my industry?",
        "Is this the right time to launch my new venture?",
        "Will my colleagues support my new initiative?",
        "Should I take a risk with this career opportunity?",
        "Will my creative project receive recognition?"
    ],
    'Finance': [
        "Will my financial situation improve soon?",
        "Should I make this investment?",
        "Will I receive the loan I applied for?",
        "How can I increase my income?",
        "Is this a good time to buy property?",
        "Will I be able to pay off my debts?",
        "Should I save or invest my money right now?",
        "Will I face unexpected expenses soon?",
        "Is this financial partnership trustworthy?",
        "Will I achieve financial stability this year?",
        "Should I change my approach to managing money?",
        "Will my financial risk pay off?",
        "How can I prepare for future financial challenges?",
        "Is my current financial advisor right for me?",
        "Should I make a major purchase now?",
        "Will I receive the financial windfall I'm hoping for?",
        "How can I improve my relationship with money?",
        "Should I be more conservative with my finances?",
        "Will my new budget plan work?",
        "Is this business opportunity financially sound?",
        "Will I achieve my savings goals?",
        "Should I purchase that expensive item I've been wanting?",
        "Will my financial sacrifices pay off in the long run?",
        "How can I create more abundance in my life?",
        "Should I trust my intuition about this financial decision?"
    ],
    'Health': [
        "Will my health improve soon?",
        "How can I increase my energy levels?",
        "Should I try this new treatment?",
        "Will my chronic condition get better?",
        "How can I improve my mental health?",
        "Will my surgery be successful?",
        "Should I change my diet?",
        "Will this new medication help me?",
        "How can I reduce my stress levels?",
        "Will my test results be good?",
        "Should I seek a second medical opinion?",
        "How can I improve my sleep quality?",
        "Will my healing process be quick?",
        "Should I start this new fitness routine?",
        "How can I maintain better wellness habits?",
        "Will my health challenges resolve this year?",
        "Should I be concerned about these symptoms?",
        "How can I best support my body during this time?",
        "Will this alternative therapy be beneficial for me?",
        "Should I make major changes to my lifestyle for health?",
        "How can I strengthen my immune system?",
        "Will my upcoming medical procedure go smoothly?",
        "Should I be more proactive about preventative care?",
        "How can I achieve better balance in my physical wellbeing?",
        "Will my new health regimen lead to improvements?"
    ],
    'Personal Growth': [
        "Am I on the right spiritual path?",
        "How can I discover my true purpose?",
        "Will I overcome my fears?",
        "Should I pursue this creative passion?",
        "How can I become more confident?",
        "Will I achieve my personal goals this year?",
        "Should I let go of this relationship or situation?",
        "How can I improve my self-discipline?",
        "Will I find inner peace soon?",
        "Should I forgive this person who hurt me?",
        "How can I connect more deeply with my intuition?",
        "Will I find clarity about my life direction?",
        "Should I pursue spiritual development more seriously?",
        "How can I overcome these limiting beliefs?",
        "Will I find a community where I truly belong?",
        "Should I trust my instincts about this important decision?",
        "How can I live more authentically?",
        "Will my meditation practice deepen?",
        "Should I explore this new philosophy or belief system?",
        "How can I heal from past trauma?",
        "Will I experience a spiritual awakening soon?",
        "Should I become more vulnerable in my relationships?",
        "How can I cultivate more gratitude in my life?",
        "Will my perspective shift help me progress?",
        "How can I embrace change more willingly?"
    ]
}

# Generate the dataset
data = []
for _ in range(200):
    # Select a random category
    category = random.choice(categories)
    
    # Select a random query from that category
    query = random.choice(queries[category])
    
    # Get random card
    card = random.choice(tarot_cards)
    
    # Get random content elements for the selected category
    sentiment = random.choice(category_content[category]['sentiments'])
    prediction = random.choice(category_content[category]['predictions'])
    advice = random.choice(category_content[category]['advice'])
    
    # Generate tarot response using a template
    template = random.choice(templates[category])
    response = template.format(card=card, sentiment=sentiment, prediction=prediction, advice=advice)
    
    data.append({
        'user_query': query,
        'category': category,
        'tarot_response': response
    })

# Create DataFrame
df = pd.DataFrame(data)

# Function to preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = text.split()
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

# Apply preprocessing to user queries
df['processed_query'] = df['user_query'].apply(preprocess_text)

# Save to CSV
df.to_csv('tarot_dataset.csv', index=False)

print(f"Dataset created with {len(df)} entries.")
print(df.head())

# Data distribution
print("\nCategory distribution:")
print(df['category'].value_counts())