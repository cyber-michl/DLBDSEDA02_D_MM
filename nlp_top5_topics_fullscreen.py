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
core_ber = (topic_info[topic_info.Topic != -1]
            .sort_values(by='Count', ascending=False)
            .head(5)
            .reset_index(drop=True))

# Beispiele pro Topic
topic_doc_indices = {}
for tid in core_ber['Topic']:
    idxs = [i for i, t in enumerate(topics) if t == tid]
    topic_doc_indices[int(tid)] = idxs[:3]

# ────────────────────────────────────────────────────────────────────────────────
# 5. BERTopic-HTML (Full-Screen + Modell-Erklärung)
# ────────────────────────────────────────────────────────────────────────────────
bertopic_html = []
bertopic_html.append("<!DOCTYPE html>")
bertopic_html.append("<html lang='de'>")
bertopic_html.append("<head>")
bertopic_html.append("  <meta charset='utf-8'>")
bertopic_html.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
bertopic_html.append("  <title>BERTopic Top 5 Themen (Full-Screen)</title>")
# CSS für Full-Screen-Nutzung
bertopic_html.append("  <style>")
bertopic_html.append("    html, body { width: 100%; height: 100%; margin: 0; padding: 0; }")
bertopic_html.append("    body { display: flex; flex-direction: column; }")
bertopic_html.append("    #intro { padding: 20px; background-color: #f0f0f0; }")
bertopic_html.append("    #topics { padding: 20px; flex: 1; overflow: auto; }")
bertopic_html.append("  </style>")
bertopic_html.append("</head>")
bertopic_html.append("<body>")

# 5.1 Modell-Erklärung
bertopic_html.append("<div id='intro'>")
bertopic_html.append("  <h1>BERTopic – Die 5 wichtigsten Themen</h1>")
bertopic_html.append("  <p><strong>Kurze Funktionsweise:</strong></p>")
bertopic_html.append("  <ul>")
bertopic_html.append("    <li>1. Satz-Embeddings mit einem vortrainierten Modell erzeugen.</li>")
bertopic_html.append("    <li>2. Dimensionsreduktion per UMAP, anschließend dichtebasiertes Clustering (HDBSCAN).</li>")
bertopic_html.append("    <li>3. Für jedes Cluster automatisch relevante Top-Begriffe ermitteln.</li>")
bertopic_html.append("    <li>4. Die fünf Cluster mit den meisten Dokumenten als Top 5 Themen auswählen.</li>")
bertopic_html.append("  </ul>")
bertopic_html.append("</div>")

# 5.2 Liste der Top 5 Themen
bertopic_html.append("<div id='topics'>")
bertopic_html.append("  <ol>")
for idx, row in core_ber.iterrows():
    nummer = idx + 1  # Nummerierung beginnt bei 1
    tid = int(row.Topic)
    count = int(row.Count)
    auto_name = get_top5_from_bert(row.Name)
    examples = [subjects[i] for i in topic_doc_indices[tid]]
    bertopic_html.append("    <li>")
    bertopic_html.append(f"      <h2>Themenbereich {nummer}: {auto_name}</h2>")
    bertopic_html.append(f"      <p><em>Anzahl zugeordneter Betreffe: {count}</em></p>")
    bertopic_html.append("      <p>Beispiele:</p>")
    bertopic_html.append("      <ul>")
    for ex in examples:
        bertopic_html.append(f"        <li>{ex}</li>")
    bertopic_html.append("      </ul>")
    bertopic_html.append("    </li>")
bertopic_html.append("  </ol>")
bertopic_html.append("  <hr>")

# 5.3 Interaktive Visualisierung (mit vollem Viewport)
bertopic_html.append("  <div id='viz' style='flex: 1;'>")
vis_bert = topic_model.visualize_topics()
# Inlinestring einbetten
bertopic_html.append(vis_bert.to_html().replace("<html>", "").replace("</html>", "").replace("<body>", "").replace("</body>", ""))
bertopic_html.append("  </div>")

bertopic_html.append("</div>")  # Ende div#topics
bertopic_html.append("</body>")
bertopic_html.append("</html>")

with open("bertopic_top5_full.html", "w", encoding="utf-8") as f:
    f.write("\n".join(bertopic_html))

print("BERTopic-HTML gespeichert als 'bertopic_top5_full.html'.")

# ────────────────────────────────────────────────────────────────────────────────
# 6. LDA-Modellierung (Top 5 Themen)
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

lda_examples = {}
for tid, _ in core_lda:
    idxs = [i for i, tt in enumerate(doc_top_topic) if tt == tid]
    lda_examples[tid] = idxs[:3]

# ────────────────────────────────────────────────────────────────────────────────
# 7. LDA-HTML (Full-Screen + Modell-Erklärung)
# ────────────────────────────────────────────────────────────────────────────────
lda_html = []
lda_html.append("<!DOCTYPE html>")
lda_html.append("<html lang='de'>")
lda_html.append("<head>")
lda_html.append("  <meta charset='utf-8'>")
lda_html.append("  <meta name='viewport' content='width=device-width, initial-scale=1.0'>")
lda_html.append("  <title>LDA Top 5 Themen (Full-Screen)</title>")
# CSS für Full-Screen-Nutzung
lda_html.append("  <style>")
lda_html.append("    html, body { width: 100%; height: 100%; margin: 0; padding: 0; }")
lda_html.append("    body { display: flex; flex-direction: column; }")
lda_html.append("    #intro { padding: 20px; background-color: #f0f0f0; }")
lda_html.append("    #topics { padding: 20px; flex: 1; overflow: auto; }")
lda_html.append("  </style>")
lda_html.append("</head>")
lda_html.append("<body>")

# 7.1 Modell-Erklärung
lda_html.append("<div id='intro'>")
lda_html.append("  <h1>LDA – Die 5 wichtigsten Themen</h1>")
lda_html.append("  <p><strong>Kurze Funktionsweise:</strong></p>")
lda_html.append("  <ul>")
lda_html.append("    <li>1. Bag-of-Words-Korpus aus bereinigten Betreffs erstellen.</li>")
lda_html.append("    <li>2. LDA-Modell mit 5 Topics trainieren (Wortverteilungen pro Thema).</li>")
lda_html.append("    <li>3. Jedem Dokument (Betreff) das Thema mit höchster Wahrscheinlichkeit zuordnen.</li>")
lda_html.append("    <li>4. Die fünf Themen, die in den meisten Dokumenten dominieren, als Top 5 auswählen.</li>")
lda_html.append("  </ul>")
lda_html.append("</div>")

# 7.2 Liste der Top 5 Themen
lda_html.append("<div id='topics'>")
lda_html.append("  <ol>")
for idx, (tid, count) in enumerate(core_lda):
    nummer = idx + 1  # Nummerierung beginnt bei 1
    raw_str = next(s for (t, s) in lda_topics_raw if t == tid)
    auto_name = get_top5_from_lda(raw_str)
    examples = [subjects[i] for i in lda_examples[tid]]
    lda_html.append("    <li>")
    lda_html.append(f"      <h2>Themenbereich {nummer}: {auto_name}</h2>")
    lda_html.append(f"      <p><em>Anzahl zugeordneter Betreffe: {count}</em></p>")
    lda_html.append("      <p>Beispiele:</p>")
    lda_html.append("      <ul>")
    for ex in examples:
        lda_html.append(f"        <li>{ex}</li>")
    lda_html.append("      </ul>")
    lda_html.append("    </li>")
lda_html.append("  </ol><hr>")

# 7.3 Interaktive Visualisierung
lda_html.append("  <div id='viz' style='flex: 1;'>")
lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
lda_html.append(pyLDAvis.prepared_data_to_html(lda_vis))
lda_html.append("  </div>")

lda_html.append("</div>")
lda_html.append("</body>")
lda_html.append("</html>")

with open("lda_top5_full.html", "w", encoding="utf-8") as f:
    f.write("\n".join(lda_html))

print("LDA-HTML gespeichert als 'lda_top5_full.html'.")
