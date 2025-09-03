# -*- coding: utf-8 -*-
"""
nlp_beschwerden_subjects_mit_themen.py

Dieses Skript liest den deutschsprachigen Teil des „Multilingual Customer Support Tickets“-Datensatzes (Kaggle),
verwendet ausschließlich die Kurzbetreff-Felder ('subject'), bereinigt sie mit NLP und
extrahiert die Top-Themen per BERTopic und LDA. 

In jeder generierten HTML-Datei sind:
1. Die Liste aller Topics mit den Top-Begriffen (Klartext oben)
2. Darunter die interaktive Visualisierung (BERTopic oder pyLDAvis)

Voraussetzung:
- Die CSV-Datei (multilingual_support_tickets.csv) liegt im gleichen Verzeichnis.
- Spalten: 'subject' (Betreff-Text), 'language' (Sprache).
"""

import os
import pandas as pd
import nltk
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt
import pyLDAvis.gensim_models
import pyLDAvis

# ────────────────────────────────────────────────────────────────────────────────
# 1. NLTK- und spaCy-Vorbereitung
# ────────────────────────────────────────────────────────────────────────────────
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('german'))

# spaCy-Modell für Deutsch laden (falls noch nicht installiert: python -m spacy download de_core_news_sm)
nlp = spacy.load("de_core_news_sm")

# ────────────────────────────────────────────────────────────────────────────────
# 2. DATEN EINLESEN & FILTERN
# ────────────────────────────────────────────────────────────────────────────────
csv_path = "multilingual_support_tickets.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Datensatz {csv_path} nicht gefunden. Bitte in dasselbe Verzeichnis legen.")

df = pd.read_csv(csv_path, dtype=str)

# Debug-Ausgabe: Spalten prüfen
print("Spalten in der CSV:", df.columns.tolist())

# Wir verwenden die Spalte 'subject' als Kurzbetreff
if 'subject' not in df.columns:
    raise KeyError("Spalte 'subject' nicht gefunden. Bitte passe den Spaltennamen an.")

# Nur deutschsprachige Tickets auswählen
df = df[df['language'] == 'de'].copy()

# Leere Einträge in 'subject' entfernen
df = df.dropna(subset=['subject'])

# Liste aller Betreff-Texte
subjects = df['subject'].tolist()
print(f"Anzahl deutschsprachiger Subjects: {len(subjects)}")

# ────────────────────────────────────────────────────────────────────────────────
# 3. VORVERARBEITUNG DER SUBJECT-TEXTE
# ────────────────────────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    """
    - Kleinschreibung
    - Tokenisierung mit spaCy
    - Entfernen von nicht-alphabetischen Tokens und Stopwords
    - Lemmatisierung
    """
    doc = nlp(text.lower())
    tokens = []
    for token in doc:
        if token.is_alpha and token.lemma_ not in stop_words:
            tokens.append(token.lemma_)
    return " ".join(tokens)

clean_subjects = [preprocess(s) for s in subjects]
print("Vorverarbeitung der Subjects abgeschlossen.")

# ────────────────────────────────────────────────────────────────────────────────
# 4. TF-IDF-Vektorisierung (Subjects)
# ────────────────────────────────────────────────────────────────────────────────
tfidf_vectorizer = TfidfVectorizer(
    max_df=0.95,      # Wörter in >95 % der Dokumente verwerfen
    min_df=2,         # Wörter in <2 Dokumenten verwerfen
    max_features=500  # Da Subjects kurz sind, reichen 500 Features
)
tfidf_matrix = tfidf_vectorizer.fit_transform(clean_subjects)
print("TF-IDF für Subjects abgeschlossen. Matrix-Shape:", tfidf_matrix.shape)

# ────────────────────────────────────────────────────────────────────────────────
# 5. Sentence Embeddings (Subjects)
# ────────────────────────────────────────────────────────────────────────────────
print("Subject-Embeddings generieren …")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = embedder.encode(clean_subjects, show_progress_bar=True)
print("Embeddings-Generierung abgeschlossen. Embedding-Shape:", embeddings.shape)

# ────────────────────────────────────────────────────────────────────────────────
# 6. BERTopic-Modellierung (Subjects)
# ────────────────────────────────────────────────────────────────────────────────
print("Starte BERTopic (Subjects) …")
topic_model = BERTopic(language="german", calculate_probabilities=True)
topics, probs = topic_model.fit_transform(clean_subjects, embeddings)
print("BERTopic (Subjects) abgeschlossen. Anzahl Themen:",
      topic_model.get_topic_info().shape[0] - 1)

# Themeninfo als DataFrame
topic_info = topic_model.get_topic_info()  # einschließlich Topic-ID, Count, Name
# Nur die tatsächlich modellierten Topics (Topic ID != –1)
modeled_topics = topic_info[topic_info.Topic != -1].copy()

# Top-Begriffe je Topic
topic_terms = {}
for topic_id in modeled_topics.Topic:
    terms = topic_model.get_topic(topic_id)  # Liste: [(Wort, Score), …]
    # Nur die Wörter extrahieren
    topic_terms[topic_id] = [t[0] for t in terms[:10]]

# ────────────────────────────────────────────────────────────────────────────────
# 7. BERTopic HTML zusammenstellen
# ────────────────────────────────────────────────────────────────────────────────
bertopic_html = []

# 7.1 Kopfzeile mit Topic-Liste (Klartext)
bertopic_html.append("<html><head><meta charset='utf-8'><title>BERTopic Subjects</title></head><body>")
bertopic_html.append("<h1>BERTopic – Extrahierte Themen (Subjects)</h1>")
bertopic_html.append("<p>Im Folgenden sind die modellierten Topics mit ihren Top-10-Begriffen:</p>")
bertopic_html.append("<ul>")
for topic_id, terms in topic_terms.items():
    b = ", ".join(terms)
    bertopic_html.append(f"<li><strong>Thema {topic_id}:</strong> {b}</li>")
bertopic_html.append("</ul><hr>")

# 7.2 Interaktive Visualisierung (gekapselt in einen Div)
bertopic_vis = topic_model.visualize_topics()
html_vis = bertopic_vis.to_html()  # Ganze Visualisierung als HTML-String
bertopic_html.append(html_vis)

# 7.3 Abschluss
bertopic_html.append("</body></html>")

output_vis_bert = "bertopic_subjects_mit_themen.html"
with open(output_vis_bert, "w", encoding="utf-8") as f:
    f.write("\n".join(bertopic_html))
print(f"BERTopic-Subjects HTML mit Topics gespeichert als {output_vis_bert}.")

# ────────────────────────────────────────────────────────────────────────────────
# 8. LDA-Modellierung (Subjects)
# ────────────────────────────────────────────────────────────────────────────────
print("Starte LDA (Subjects) …")
tokenized_subjects = [s.split() for s in clean_subjects]

dictionary = corpora.Dictionary(tokenized_subjects)
corpus = [dictionary.doc2bow(doc) for doc in tokenized_subjects]

lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)
print("LDA (Subjects) abgeschlossen.")

# LDA-Themen (Top-Wörter)  
lda_topics = lda_model.print_topics(num_words=10)

# ────────────────────────────────────────────────────────────────────────────────
# 9. LDA HTML zusammenstellen
# ────────────────────────────────────────────────────────────────────────────────
lda_html = []
lda_html.append("<html><head><meta charset='utf-8'><title>LDA Subjects</title></head><body>")
lda_html.append("<h1>LDA – Extrahierte Themen (Subjects)</h1>")
lda_html.append("<p>Im Folgenden sind die fünf modellierten LDA-Topics mit ihren Top-10-Wörtern:</p>")
lda_html.append("<ul>")
for idx, topic in lda_topics:
    lda_html.append(f"<li><strong>Thema {idx}:</strong> {topic}</li>")
lda_html.append("</ul><hr>")

# Interaktive LDA-Visualisierung einfügen
lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
html_lda_vis = pyLDAvis.prepared_data_to_html(lda_vis)  # HTML-String  
lda_html.append(html_lda_vis)

lda_html.append("</body></html>")

output_vis_lda = "lda_subjects_mit_themen.html"
with open(output_vis_lda, "w", encoding="utf-8") as f:
    f.write("\n".join(lda_html))
print(f"LDA-Subjects HTML mit Topics gespeichert als {output_vis_lda}.")

# ────────────────────────────────────────────────────────────────────────────────
# 10. FAZIT
# ────────────────────────────────────────────────────────────────────────────────
print("\n*** Themenextraktion (Subjects) abgeschlossen. ***")
print(f"> Siehe BERTopic-HTML: {output_vis_bert}")
print(f"> Siehe LDA-HTML: {output_vis_lda}")
