import sqlite3

# initialize the database

conn = sqlite3.connect("memories.db")
cursor = conn.cursor()
cursor.execute("""
                create table if not exists memories
                (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 user_input TEXT,                             
                 category TEXT DEFAULT 'Uncategorized',
                 timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)
              """)
conn.commit()

# user input will store what user said
# category will categories data into different category (e.g. work,personal etc)
# timestamp will record time at which it was added

# nltk

'''import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')'''

# implement text processing

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))  # will remove the words like i,the,is,a etc
lemmatizer = WordNetLemmatizer()              # will make every word to it's origin like running become run

def preprocess_text(text):
    tokens = word_tokenize(text.lower())      # eg. "I am going to the mall" = ["i","am","going","to","the","mall"] in lower-case
    filtered_tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)

print(preprocess_text("I have meeting tomorrow !!"))

# categorising user text using NLP like work,personal,reminder etc

'''from nltk import classify, NaiveBayesClassifier'''
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder

training_data = [
    # Work-Related
    ("Meeting with the manager tomorrow at 3 PM", "Work"),
    ("I have a project deadline on Friday.", "Work"),
    ("Submit the final report to the client", "Work"),
    ("Schedule a Zoom call with the team", "Work"),
    ("Prepare slides for the business presentation", "Work"),

    # Mood-Related
    ("I felt really happy today after the gym", "Mood"),
    ("I'm feeling super excited today!", "Mood"),
    ("I'm a little stressed about the upcoming exams.", "Mood"),
    ("Today was such a relaxing day at the beach.", "Mood"),
    ("I feel really grateful for everything today.", "Mood"),

    # Personal Tasks
    ("Buy groceries and milk", "Personal"),
    ("Pick up laundry from the dry cleaner", "Personal"),
    ("Book a doctor’s appointment for next week", "Personal"),
    ("Renew my gym membership", "Personal"),
    ("Order a birthday gift for mom", "Personal"),

    # Reminders
    ("Remember to call Alex on Monday", "Reminder"),
    ("Set an alarm for 6 AM tomorrow", "Reminder"),
    ("Pay the electricity bill before the due date", "Reminder"),
    ("Take vitamins after breakfast", "Reminder"),
    ("Remind me to water the plants in the evening", "Reminder"),

    # Ideas & Creativity
    ("A new app idea: AI-based journal", "Ideas"),
    ("Thinking about starting a travel vlog", "Ideas"),
    ("Maybe I should write a book about my experiences", "Ideas"),
    ("A concept for a smart home assistant device", "Ideas"),
    ("A cool business idea: Subscription-based meal plans", "Ideas"),

    # Health & Fitness
    ("Go for a morning jog at 6 AM", "Health"),
    ("Drink 2 liters of water every day", "Health"),
    ("Meditate for 10 minutes before bed", "Health"),
    ("Join a yoga class this weekend", "Health"),
    ("Try a new healthy recipe for dinner", "Health"),

    # Finance
    ("Transfer money to savings account", "Finance"),
    ("Invest in cryptocurrency this month", "Finance"),
    ("Track monthly expenses in a budget app", "Finance"),
    ("Plan a budget for the next vacation", "Finance"),
    ("Check credit card bill before the due date", "Finance")
]

texts, labels = zip(*training_data)          # will seperate data into 2 lists text and label

'''vectorizer = TfidfVectorizer()               # will convert text into numerical
X = vectorizer.fit_transform(texts)

# Function to Convert TF-IDF Vectors to Dictionary (for NLTK)
def to_feature_dict(vector):
    return {str(i): vector[i] for i in range(len(vector))}

# Prepare Data for Naïve Bayes Classifier
feature_sets = [(to_feature_dict(X.toarray()[i]), labels[i]) for i in range(len(labels))]

# Train Classifier
classifier = NaiveBayesClassifier.train(feature_sets) 

def predict_category(text):
    processed_text = preprocess_text(text)  # Preprocess input text
    X_new = vectorizer.transform([processed_text])  # Convert to TF-IDF vector
    feature_dict = to_feature_dict(X_new.toarray()[0])  # Convert to dictionary
    return classifier.classify(feature_dict)  # Classify using trained model'''

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)  # Convert category names to numbers

# Train a New Model Using Scikit-Learn
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(texts, y)  # Train using TF-IDF and Naïve Bayes

def predict_category(text):
    processed_text = preprocess_text(text)  # Preprocess input text
    predicted_label = model.predict([processed_text])[0]  # Predict category index
    return label_encoder.inverse_transform([predicted_label])[0]  # Convert index back to category name

# Test Predictions
print(predict_category("I have a project deadline on Friday."))  # Expected: Work
print(predict_category("I'm feeling super excited today!"))  # Expected: Mood

# save and recall memory

# save fun
def save_memory(user_input):
    category = predict_category(user_input)
    cursor.execute("INSERT INTO memories (user_input, category) VALUES (?, ?)", (user_input, category))
    conn.commit()
    return f"Got it! I've saved this under '{category}'."

# recall function
def recall_memory():
    cursor.execute("SELECT user_input, category, timestamp FROM memories ORDER BY timestamp DESC LIMIT 5")
    memories = cursor.fetchall()
    
    if not memories:
        return "No memories yet."
    
    return "\n".join([f"{m[2]} - {m[1]}: {m[0]}" for m in memories])

# create ChatBot interface
while True:
    user_input = input("You: ")
    
    if user_input.lower() in ["recall", "remember", "show my memories"]:
        print(recall_memory())
    
    elif user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    
    else:
        print(save_memory(user_input))






     
    



