```python
# -*- coding: utf-8 -*-
"""
nlp_top5_topics_automatic.py

Dieses Skript liest den deutschsprachigen Teil des „Multilingual Customer Support Tickets“-Datensatzes 
(Spalte "subject"), extrahiert die 5 häufigsten Themen mittels BERTopic und LDA und erstellt zwei 
HTML-Dateien, in denen jeweils diese Themen mit:
- Themenbereich-Nummer (beginnt bei 1)
- Dokumentanzahl (Count)
- Die automatisch generierten Top-5-Begriffe
- Drei Beispiel-Subjects
- Interaktive Visualisierung

angezeigt werden. Es wird keine manuelle Beschriftung verwendet, sondern direkt die Top-5-Begriffe 
als „Themenbereich“-Label.

Voraussetzungen:
- multilingual_support_tickets.csv im selben Ordner (Spalten: 'subject', 'language')
- Conda-Umgebung „nlp-beschwerden“ mit pandas, nltk, spacy, sklearn, sentence-transformers, bertopic, gensim, pyLDAvis installiert
- spaCy-Deutschmodell und NLTK-Stopwords:
    python -m spacy download de_core_news_sm
    python -c "import nltk; nltk.download('stopwords')"
"""

import os
import re
import pandas as pd
import nltk
import spacy
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import pyLDAvis.gensim_models
import pyLDAvis
from collections import Counter

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
    raise KeyError("CSV muss die Spalten 'subject' und 'language' enthalten.")

df = df[df['language'] == 'de'].dropna(subset=['subject']).copy()
subjects = df['subject'].tolist()
print(f"Anzahl deutschsprachiger Betreffe: {len(subjects)}")

# ────────────────────────────────────────────────────────────────────────────────
# 3. Vorverarbeitung
# ────────────────────────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and token.lemma_ not in stop_words]
    return " ".join(tokens)

clean_subjects = [preprocess(s) for s in subjects]
print("Vorverarbeitung abgeschlossen.")

# Hilfsfunktion: aus einer Komma-liste von Top-Begriffen die ersten fünf extrahieren
def get_top5_from_bert(name_str: str) -> str:
    terms = [t.strip() for t in name_str.split(",")]
    return ", ".join(terms[:5])

# Hilfsfunktion: aus LDA-String die ersten fünf Begriffe extrahieren
def get_top5_from_lda(topic_str: str) -> str:
    # find all Wörter in Anführungszeichen
    words = re.findall(r'\"([^\"]+)\"', topic_str)
    return ", ".join(words[:5])

# ────────────────────────────────────────────────────────────────────────────────
# 4. BERTopic-Modellierung (Top 5 Themen)
# ────────────────────────────────────────────────────────────────────────────────
print("Starte BERTopic …")
embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
embeddings = embedder.encode(clean_subjects, show_progress_bar=True)

topic_model = BERTopic(language="german", calculate_probabilities=True)
topics, probs = topic_model.fit_transform(clean_subjects, embeddings)
print("BERTopic abgeschlossen.")

topic_info = topic_model.get_topic_info()
# Fünf häufigste Topics (Topic != -1) nach Count
core_ber = (topic_info[topic_info.Topic != -1]
            .sort_values(by='Count', ascending=False)
            .head(5)
            .reset_index(drop=True))

# Beispiele pro Topic
topic_doc_indices = {}
for tid in core_ber['Topic']:
    idxs = [i for i, t in enumerate(topics) if t == tid]
    topic_doc_indices[int(tid)] = idxs[:3]

# Generiere HTML mit den Top 5 BERTopic-Themen
bertopic_html = []
bertopic_html.append("<html><head><meta charset='utf-8'><title>BERTopic Top 5 Themen</title></head><body>")
bertopic_html.append("<h1>BERTopic – Die 5 wichtigsten Themen (Betreffs)</h1>")
bertopic_html.append("<p>Im Folgenden sind die fünf häufigsten Themenbereiche anhand der automatisch ermittelten Top-5-Begriffe aufgelistet:</p>")
bertopic_html.append("<ol>")

for idx, row in core_ber.iterrows():
    nummer = idx + 1  # Nummerierung beginnt bei 1
    tid = int(row.Topic)
    count = int(row.Count)
    auto_name = get_top5_from_bert(row.Name)
    examples = [subjects[i] for i in topic_doc_indices[tid]]
    bertopic_html.append(f"<li><strong>Themenbereich {nummer}:</strong> {auto_name} "
                        f"(<em>{count} Tickets</em>)<br>")
    bertopic_html.append("<em>Beispiele:</em><ul>")
    for ex in examples:
        bertopic_html.append(f"<li>{ex}</li>")
    bertopic_html.append("</ul></li>")

bertopic_html.append("</ol><hr>")

vis_bert = topic_model.visualize_topics()
bertopic_html.append(vis_bert.to_html())

bertopic_html.append("</body></html>")

with open("bertopic_top5.html", "w", encoding="utf-8") as f:
    f.write("\n".join(bertopic_html))

print("BERTopic-HTML gespeichert als 'bertopic_top5.html'.")

# ────────────────────────────────────────────────────────────────────────────────
# 5. LDA-Modellierung (Top 5 Themen)
# ────────────────────────────────────────────────────────────────────────────────
print("\nStarte LDA …")
tokenized = [s.split() for s in clean_subjects]
dictionary = corpora.Dictionary(tokenized)
corpus = [dictionary.doc2bow(doc) for doc in tokenized]

lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10)
print("LDA abgeschlossen.")

lda_topics_raw = lda_model.print_topics(num_words=10)
doc_top_topic = [max(lda_model.get_document_topics(bow), key=lambda x: x[1])[0] for bow in corpus]
topic_counts = Counter(doc_top_topic)
core_lda = topic_counts.most_common(5)

# Beispiele pro LDA-Topic
lda_examples = {}
for tid, _ in core_lda:
    idxs = [i for i, tt in enumerate(doc_top_topic) if tt == tid]
    lda_examples[tid] = idxs[:3]

# Generiere HTML mit den Top 5 LDA-Themen
lda_html = []
lda_html.append("<html><head><meta charset='utf-8'><title>LDA Top 5 Themen</title></head><body>")
lda_html.append("<h1>LDA – Die 5 wichtigsten Themen (Betreffs)</h1>")
lda_html.append("<p>Im Folgenden sind die fünf häufigsten Themenbereiche anhand der automatisch ermittelten Top-5-Wörter aufgelistet:</p>")
lda_html.append("<ol>")

for idx, (tid, count) in enumerate(core_lda):
    nummer = idx + 1  # Nummerierung beginnt bei 1
    auto_name = get_top5_from_lda(next(s for (t, s) in lda_topics_raw if t == tid))
    examples = [subjects[i] for i in lda_examples[tid]]
    lda_html.append(f"<li><strong>Themenbereich {nummer}:</strong> {auto_name} "
                    f"(<em>{count} Tickets</em>)<br>")
    lda_html.append("<em>Beispiele:</em><ul>")
    for ex in examples:
        lda_html.append(f"<li>{ex}</li>")
    lda_html.append("</ul></li>")

lda_html.append("</ol><hr>")

lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
lda_html.append(pyLDAvis.prepared_data_to_html(lda_vis))

lda_html.append("</body></html>")

with open("lda_top5.html", "w", encoding="utf-8") as f:
    f.write("\n".join(lda_html))

print("LDA-HTML gespeichert als 'lda_top5.html'.")
```
