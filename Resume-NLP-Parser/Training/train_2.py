import spacy
from spacy.training.example import Example
import csv
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

# Function to train the NER model with data from the CSV file and save it
def train_and_save_spacy_model(output_dir="TrainedModel/test2", iterations=20):
    nlp = spacy.blank("en")  # Create a blank 'en' model

    # Create a Named Entity Recognition (NER) pipeline
    ner = nlp.add_pipe("ner", name="ner", last=True)
    ner.add_label("SKILL")  # Add the label for skills recognition

    # Load skills data from CSV file and create training data
    TRAIN_DATA = []
    with open('data/newSkills.csv', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            skill_text = row[0].strip()  # Skill text from the CSV row
            if skill_text:  # Check for non-empty skill text
                doc = nlp.make_doc(skill_text)
                entities = [(0, len(skill_text), "SKILL")]
                TRAIN_DATA.append((doc, {"entities": entities}))

    # Begin training
    nlp.begin_training()

    # Iterate through training data
    for itn in range(iterations):
        random.shuffle(TRAIN_DATA)
        losses = {}
        # Create examples and update the model
        for text, annotations in TRAIN_DATA:
            example = Example.from_dict(text, annotations)
            nlp.update([example], drop=0.5, losses=losses)

        print("Iteration:", itn+1, "Loss:", losses)

    # Save the trained model to the specified output directory
    nlp.to_disk(output_dir)
    print("Trained model saved to:", output_dir)
    return nlp

# Evaluate the model with Random Forest Classifier
def evaluate_model_with_rf(nlp, csv_file='./data/newSkills.csv'):
    # Load skills data
    texts = []
    labels = []
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            skill_text = row[0].strip()
            if skill_text:
                texts.append(skill_text)
                labels.append(1)  # Assuming all skills in the CSV are true skills

    # Generate predictions using the SpaCy model
    predictions = []
    for text in texts:
        doc = nlp(text)
        skill_detected = any(ent.label_ == "SKILL" for ent in doc.ents)
        predictions.append(1 if skill_detected else 0)

    # Convert texts into features using CountVectorizer
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(texts).toarray()

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, predictions, test_size=0.2, random_state=42)

    # Train a Random Forest Classifier
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    # Make predictions
    y_pred = rf.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=1)
    recall = recall_score(y_test, y_pred, zero_division=1)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)

# Run the training function and evaluation
trained_model = train_and_save_spacy_model()
evaluate_model_with_rf(trained_model)
