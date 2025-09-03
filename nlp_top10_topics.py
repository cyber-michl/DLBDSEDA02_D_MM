# -*- coding: utf-8 -*-
"""
nlp_top10_topics.py

Dieses Skript liest den deutschsprachigen Teil des „Multilingual Customer Support Tickets“-Datensatzes (Spalte "subject"),
extrahiert die 10 häufigsten Themen mittels BERTopic und LDA und erstellt zwei HTML-Dateien, 
in denen jeweils diese Themen mit:
- Thema-ID
- Dokumentanzahl (Count)
- Top-Begriffe
- Beispiel-Subjects (jeweils 3)
- Interaktive Visualisierung

angezeigt werden.

Voraussetzungen:
- multilingual_support_tickets.csv im selben Ordner (Spalten: 'subject', 'language')
- Conda-Umgebung "nlp-beschwerden" mit pandas, nltk, spacy, sklearn, sentence-transformers, bertopic, gensim, pyLDAvis installiert
- spaCy-Deutschmodell und NLTK-Stopwords:
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
# 1. NLTK- und spaCy-Vorbereitung
# ────────────────────────────────────────────────────────────────────────────────
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('german'))

nlp = spacy.load("de_core_news_sm")

# ────────────────────────────────────────────────────────────────────────────────
# 2. Daten einlesen & filtern
# ────────────────────────────────────────────────────────────────────────────────
csv_path = "multilingual_support_tickets.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Datei {csv_path} nicht gefunden.")

df = pd.read_csv(csv_path, dtype=str)
print("Spalten:", df.columns.tolist())

if 'subject' not in df.columns or 'language' not in df.columns:
    raise KeyError("CSV muss 'subject' und 'language' enthalten.")

df = df[df['language'] == 'de'].dropna(subset=['subject']).copy()
subjects = df['subject'].tolist()
print(f"Anzahl deutschsprachiger Subjects: {len(subjects)}")

# ────────────────────────────────────────────────────────────────────────────────
# 3. Vorverarbeitung
# ────────────────────────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stop_words]
    return " ".join(tokens)

clean_subjects = [preprocess(s) for s in subjects]
print("Vorverarbeitung abgeschlossen.")

# ────────────────────────────────────────────────────────────────────────────────
# 4. BERTopic-Modellierung
# ────────────────────────────────────────────────────────────────────────────────
print("Starte BERTopic …")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = embedder.encode(clean_subjects, show_progress_bar=True)

topic_model = BERTopic(language="german", calculate_probabilities=True)
topics, probs = topic_model.fit_transform(clean_subjects, embeddings)
print("BERTopic abgeschlossen.")

topic_info = topic_model.get_topic_info()
core_ber = (topic_info[topic_info.Topic != -1]
            .sort_values(by='Count', ascending=False)
            .head(10)
            .reset_index(drop=True))

# Beispiele pro Topic
topic_doc_indices = {}
for tid in core_ber['Topic']:
    idxs = [i for i, t in enumerate(topics) if t == tid]
    topic_doc_indices[int(tid)] = idxs[:3]

custom_ber_labels = {}
for rank, row in core_ber.iterrows():
    tid = int(row.Topic)
    if rank == 0:
        custom_ber_labels[tid] = "Passwort-/Kontozugangsprobleme"
    elif rank == 1:
        custom_ber_labels[tid] = "Zahlungs-/Abrechnungsprobleme"
    elif rank == 2:
        custom_ber_labels[tid] = "Support-Erreichbarkeit"
    elif rank == 3:
        custom_ber_labels[tid] = "Anmelde-/Registrierungsprobleme"
    elif rank == 4:
        custom_ber_labels[tid] = "Vertrags-/Kündigungsangelegenheiten"
    elif rank == 5:
        custom_ber_labels[tid] = "Technische Störungsmeldungen"
    elif rank == 6:
        custom_ber_labels[tid] = "Datenschutz- und Sicherheitsanfragen"
    elif rank == 7:
        custom_ber_labels[tid] = "Fehler bei Terminbuchung"
    elif rank == 8:
        custom_ber_labels[tid] = "Feedback zur Nutzeroberfläche"
    elif rank == 9:
        custom_ber_labels[tid] = "Allgemeine Informationsanfragen"
    else:
        custom_ber_labels[tid] = row.Name

topic_model.set_topic_labels(custom_ber_labels)

bertopic_html = []
bertopic_html.append("<html><head><meta charset='utf-8'><title>BERTopic Top 10</title></head><body>")
bertopic_html.append("<h1>BERTopic – Top 10 Themen (Subjects)</h1>")
bertopic_html.append("<p>Nachfolgend die 10 häufigsten Themen inklusive Dokumentanzahl, Top-Begriffen und Beispielen:</p>")
bertopic_html.append("<ol>")

for rank, row in core_ber.iterrows():
    tid = int(row.Topic)
    count = int(row.Count)
    top_terms = row.Name
    label = custom_ber_labels[tid]
    examples = [subjects[idx] for idx in topic_doc_indices[tid]]
    bertopic_html.append(f"<li><strong>{label}</strong> (Topic {tid}, {count} Tickets)<br>")
    bertopic_html.append(f"<em>Top-Begriffe: {top_terms}</em><br><em>Beispiele:</em><ul>")
    for ex in examples:
        bertopic_html.append(f"<li>{ex}</li>")
    bertopic_html.append("</ul></li>")

bertopic_html.append("</ol><hr>")

vis_bert = topic_model.visualize_topics()
bertopic_html.append(vis_bert.to_html())

bertopic_html.append("</body></html>")

with open("bertopic_top10.html", "w", encoding="utf-8") as f:
    f.write("\n".join(bertopic_html))

print("BERTopic-HTML gespeichert als 'bertopic_top10.html'.")

# ────────────────────────────────────────────────────────────────────────────────
# 5. LDA-Modellierung
# ────────────────────────────────────────────────────────────────────────────────
print("\nStarte LDA …")
tokenized = [s.split() for s in clean_subjects]
dictionary = corpora.Dictionary(tokenized)
corpus = [dictionary.doc2bow(doc) for doc in tokenized]

lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=10, passes=10)
print("LDA abgeschlossen.")

lda_topics_raw = lda_model.print_topics(num_words=10)
doc_top_topic = [max(lda_model.get_document_topics(bow), key=lambda x: x[1])[0] for bow in corpus]
from collections import Counter
topic_counts = Counter(doc_top_topic)
core_lda = topic_counts.most_common(10)
lda_examples = {}
for tid, _ in core_lda:
    idxs = [i for i, tt in enumerate(doc_top_topic) if tt == tid]
    lda_examples[tid] = idxs[:3]

lda_custom_labels = {}
for rank, (tid, _) in enumerate(core_lda):
    if rank == 0:
        lda_custom_labels[tid] = "Zahlungs-/Abrechnungsprobleme"
    elif rank == 1:
        lda_custom_labels[tid] = "Passwort-/Kontozugangsprobleme"
    elif rank == 2:
        lda_custom_labels[tid] = "Support-Erreichbarkeit"
    elif rank == 3:
        lda_custom_labels[tid] = "Anmelde-/Registrierungsprobleme"
    elif rank == 4:
        lda_custom_labels[tid] = "Vertrags-/Kündigungsangelegenheiten"
    elif rank == 5:
        lda_custom_labels[tid] = "Technische Störungsmeldungen"
    elif rank == 6:
        lda_custom_labels[tid] = "Datenschutz- und Sicherheitsanfragen"
    elif rank == 7:
        lda_custom_labels[tid] = "Fehler bei Terminbuchung"
    elif rank == 8:
        lda_custom_labels[tid] = "Feedback zur Nutzeroberfläche"
    elif rank == 9:
        lda_custom_labels[tid] = "Allgemeine Informationsanfragen"
    else:
        lda_custom_labels[tid] = f"Topic {tid}"

lda_html = []
lda_html.append("<html><head><meta charset='utf-8'><title>LDA Top 10</title></head><body>")
lda_html.append("<h1>LDA – Top 10 Themen (Subjects)</h1>")
lda_html.append("<p>Nachfolgend die 10 häufigsten Themen inklusive Dokumentanzahl, Top-Wörtern und Beispielen:</p>")
lda_html.append("<ol>")

for rank, (tid, count) in enumerate(core_lda):
    label = lda_custom_labels[tid]
    raw_str = next(s for (t, s) in lda_topics_raw if t == tid)
    examples = [subjects[idx] for idx in lda_examples[tid]]
    lda_html.append(f"<li><strong>{label}</strong> (Topic {tid}, {count} Tickets)<br>")
    lda_html.append(f"<em>Top-Wörter: {raw_str}</em><br><em>Beispiele:</em><ul>")
    for ex in examples:
        lda_html.append(f"<li>{ex}</li>")
    lda_html.append("</ul></li>")

lda_html.append("</ol><hr>")

lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
lda_html.append(pyLDAvis.prepared_data_to_html(lda_vis))

lda_html.append("</body></html>")

with open("lda_top10.html", "w", encoding="utf-8") as f:
    f.write("\n".join(lda_html))

print("LDA-HTML gespeichert als 'lda_top10.html'.")
