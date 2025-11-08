# Task 3 - Named Entity Recognition & Sentiment Analysis using spaCy

import spacy
nlp = spacy.load("en_core_web_sm")

# Sample text (you can replace this with actual Amazon reviews)
reviews = [
    "I love my new Samsung Galaxy phone, it's fast and the camera is amazing!",
    "The battery life of this Apple iPhone is terrible.",
    "Sony headphones have incredible sound quality!",
    "I had issues with the Dell laptop overheating frequently."
]

for text in reviews:
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    # Simple rule-based sentiment
    positive_words = ['love', 'amazing', 'incredible', 'great', 'fast']
    negative_words = ['terrible', 'bad', 'poor', 'slow', 'issues']

    sentiment = "Positive" if any(word in text.lower() for word in positive_words) else \
                "Negative" if any(word in text.lower() for word in negative_words) else "Neutral"

    print(f"\nReview: {text}")
    print("Entities Detected:", entities)
    print("Sentiment:", sentiment)
