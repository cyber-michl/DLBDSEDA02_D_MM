# -*- coding: utf-8 -*-
"""
nlp_beschwerden_subjects_final.py

Ziel: Aus der Spalte “subject” deutschsprachiger Support-Tickets die am häufigsten diskutierten Themen
extrahieren und in zwei HTML-Berichten (BERTopic + LDA) mit verständlichen Themen-Namen anzeigen.

Voraussetzungen:
- Datei “multilingual_support_tickets.csv” im selben Verzeichnis.
- Spalten: 'subject' (Betreff-Text), 'language' (z. B. 'de').
- Conda-Umgebung “nlp-beschwerden” aktiv und alle Pakete installiert:
    pandas, nltk, spacy, scikit-learn, sentence-transformers, bertopic, gensim, pyLDAvis

Installation spaCy, Stopwords:
    python -m spacy download de_core_news_sm
    python -c "import nltk; nltk.download('stopwords')"
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
import pyLDAvis.gensim_models
import pyLDAvis

# ────────────────────────────────────────────────────────────────────────────────
# 1. NLTK‐ und spaCy‐Vorbereitung
# ────────────────────────────────────────────────────────────────────────────────
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('german'))

# spaCy‐Modell laden
nlp = spacy.load("de_core_news_sm")

# ────────────────────────────────────────────────────────────────────────────────
# 2. DATEN EINLESEN & FILTERN
# ────────────────────────────────────────────────────────────────────────────────
csv_path = "multilingual_support_tickets.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Datensatz {csv_path} nicht gefunden!")

df = pd.read_csv(csv_path, dtype=str)
print("Spalten in der CSV:", df.columns.tolist())

if 'subject' not in df.columns:
    raise KeyError("Spalte 'subject' nicht gefunden!")

# Nur deutschsprachige Tickets
df = df[df['language'] == 'de'].copy()
df = df.dropna(subset=['subject'])
subjects = df['subject'].tolist()
print(f"Anzahl deutschsprachiger Subjects: {len(subjects)}")

# ────────────────────────────────────────────────────────────────────────────────
# 3. VORVERARBEITUNG DER SUBJECT‐TEXTE
# ────────────────────────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stop_words]
    return " ".join(tokens)

clean_subjects = [preprocess(s) for s in subjects]
print("Vorverarbeitung abgeschlossen.")

# ────────────────────────────────────────────────────────────────────────────────
# 4. Sentence Embeddings (Subjects)
# ────────────────────────────────────────────────────────────────────────────────
print("Subject‐Embeddings generieren …")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = embedder.encode(clean_subjects, show_progress_bar=True)
print("Embeddings‐Generierung abgeschlossen.")

# ────────────────────────────────────────────────────────────────────────────────
# 5. BERTopic‐Modellierung (Subjects)
# ────────────────────────────────────────────────────────────────────────────────
print("Starte BERTopic (Subjects) …")
topic_model = BERTopic(language="german", calculate_probabilities=True)
topics, probs = topic_model.fit_transform(clean_subjects, embeddings)
print("BERTopic abgeschlossen.")

# 5.1: Automatisch generierte Labels (Top‐Begriffe) abrufen
topic_info = topic_model.get_topic_info()
auto_labels = {
    int(row.Topic): row.Name
    for _, row in topic_info[topic_info.Topic != -1].iterrows()
}

# 5.2: Menschliche, verständliche Themen-Namen definieren (Override)
# Passe diese Beschriftungen manuell an, basierend auf den Top‐Begriffen.
# Beispielzuordnung:
custom_labels = {
    0: "Passwort-/Kontozugangsprobleme",
    1: "Zahlungs-/Abrechnungsprobleme",
    2: "Support-Erreichbarkeit",
    3: "Anmelde-/Registrierungsfehler",
    4: "Vertrags-/Kündigungsangelegenheiten"
}

# Stelle sicher, dass keine ungültigen IDs verwendet werden
for tid in list(custom_labels.keys()):
    if tid not in auto_labels:
        custom_labels.pop(tid)

# Setze die custom_labels in BERTopic
topic_model.set_topic_labels(custom_labels)

# 5.3: Finale Liste der Themen (ID → Name)
final_bertopic_labels = {
    tid: custom_labels.get(tid, auto_labels[tid])
    for tid in auto_labels.keys()
}

# ────────────────────────────────────────────────────────────────────────────────
# 6. BERTopic‐HTML erstellen
# ────────────────────────────────────────────────────────────────────────────────
bertopic_html = []
bertopic_html.append("<html><head><meta charset='utf-8'><title>BERTopic Subjects</title></head><body>")
bertopic_html.append("<h1>BERTopic – Häufigste Themen (Subjects)</h1>")
bertopic_html.append("<p>Die folgenden Themen wurden aus den Betreff-Feldern extrahiert:</p>")
bertopic_html.append("<ul>")
for tid, name in final_bertopic_labels.items():
    bertopic_html.append(f"<li><strong>Thema {tid}:</strong> {name}</li>")
bertopic_html.append("</ul><hr>")

# Interaktive Visualisierung (nun mit benannten Topics)
vis_bert = topic_model.visualize_topics()
bertopic_html.append(vis_bert.to_html())

bertopic_html.append("</body></html>")
output_bert = "bertopic_subjects_final.html"
with open(output_bert, "w", encoding="utf-8") as f:
    f.write("\n".join(bertopic_html))
print(f"BERTopic‐HTML mit finalen Themen gespeichert als {output_bert}.")

# ────────────────────────────────────────────────────────────────────────────────
# 7. LDA‐Modellierung (Subjects)
# ────────────────────────────────────────────────────────────────────────────────
print("Starte LDA (Subjects) …")
tokenized = [s.split() for s in clean_subjects]
dictionary = corpora.Dictionary(tokenized)
corpus = [dictionary.doc2bow(doc) for doc in tokenized]

lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)
print("LDA abgeschlossen.")

lda_topics = lda_model.print_topics(num_words=10)
# Beispielausgabe: [(0, '0.025*"zahlung" + 0.020*"rückerstattung" + …'), ...]

# 7.1: Menschliche Themen-Namen auch hier manuell zuordnen
lda_custom_labels = {
    0: "Zahlungs-/Abrechnungsprobleme",
    1: "Passwort-/Kontozugangsprobleme",
    2: "Support-Erreichbarkeit",
    3: "Anmelde-/Registrierungsfehler",
    4: "Vertrags-/Kündigungsangelegenheiten"
}

# Stelle sicher, dass LDA tatsächlich genau 5 Topics (0–4) hat
lda_labels = {}
for idx, _ in lda_topics:
    if idx in lda_custom_labels:
        lda_labels[idx] = lda_custom_labels[idx]
    else:
        # Fallback: falls ID nicht in custom_labels, verwende die Roh-Ausgabe
        lda_labels[idx] = f"Topic {idx}"

# ────────────────────────────────────────────────────────────────────────────────
# 8. LDA‐HTML erstellen
# ────────────────────────────────────────────────────────────────────────────────
lda_html = []
lda_html.append("<html><head><meta charset='utf-8'><title>LDA Subjects</title></head><body>")
lda_html.append("<h1>LDA – Häufigste Themen (Subjects)</h1>")
lda_html.append("<p>Die folgenden Themen wurden aus den Betreff-Feldern extrahiert:</p>")
lda_html.append("<ul>")
for idx, name in lda_labels.items():
    # Zeige zusätzlich die Top-10-Wörter für Transparenz
    topic_str = next(t for t in lda_topics if t[0] == idx)[1]
    lda_html.append(f"<li><strong>Thema {idx}:</strong> {name} <br> &nbsp;&nbsp;&nbsp;&nbsp;<em>({topic_str})</em></li>")
lda_html.append("</ul><hr>")

# Interaktive LDA-Visualisierung einbetten
lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
lda_html.append(pyLDAvis.prepared_data_to_html(lda_vis))

lda_html.append("</body></html>")
output_lda = "lda_subjects_final.html"
with open(output_lda, "w", encoding="utf-8") as f:
    f.write("\n".join(lda_html))
print(f"LDA‐HTML mit finalen Themen gespeichert als {output_lda}.")

# ────────────────────────────────────────────────────────────────────────────────
# 9. Abschluss
# ────────────────────────────────────────────────────────────────────────────────
print("\n*** Extraktion der Kern-Themen abgeschlossen. ***")
print(f"> BERTopic‐HTML: {output_bert}")
print(f"> LDA‐HTML:      {output_lda}")
