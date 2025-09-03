Alles klar 👍
Ich baue dir dein Skript **so um, dass es exakt der Aufgabenstellung entspricht**:

* **Extraktion der 5 häufigsten Themen** aus deutschsprachigen Texten.
* **Nur deutschsprachige Begriffe** in den Topics (englische werden mit gefiltert).
* **Zwei Verfahren**:

  1. **BERTopic** mit multilingualem Modell.
  2. **LDA** mit deutscher Stopwort-Filterung.
* **HTML-Output fullscreen**, mit kurzer Erklärung zum Modell.

Hier das angepasste Skript `nlp_top5_topics_fullscreen.py`:

```python
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# NLTK Ressourcen laden
nltk.download('stopwords')

# Stopwörter kombinieren (Deutsch + Englisch, damit englische Wörter gefiltert werden)
german_stopwords = set(stopwords.words('german'))
english_stopwords = set(stopwords.words('english'))
all_stopwords = german_stopwords.union(english_stopwords)

# CSV laden (z. B. Kaggle-Datensatz)
df = pd.read_csv("complaints.csv")  # <--- anpassen an deinen Dateinamen
print("Spalten:", df.columns)

# Nur deutsche Einträge
df = df[df["language"] == "de"]

# Texte bereinigen
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zäöüß ]', ' ', text)
    tokens = [w for w in text.split() if w not in all_stopwords and len(w) > 2]
    return " ".join(tokens)

df["clean_subject"] = df["subject"].astype(str).apply(clean_text)

print("Anzahl deutschsprachiger Betreffe:", len(df))

# ---------- BERTopic ----------
print("Starte BERTopic …")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
topic_model = BERTopic(embedding_model=embedding_model, language="german")

topics, _ = topic_model.fit_transform(df["clean_subject"])
freq = topic_model.get_topic_freq().sort_values("Count", ascending=False).head(5)

# HTML für BERTopic
bertopic_html = """
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>Top 5 Themen – BERTopic</title>
<style>
  body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
  .container { width: 100vw; height: 100vh; padding: 40px; box-sizing: border-box; }
  h1 { text-align: center; }
  .topic { margin: 20px 0; padding: 15px; border: 1px solid #ccc; border-radius: 8px; }
  .explain { background: #eef; padding: 10px; margin: 20px 0; border-radius: 8px; }
</style>
</head>
<body>
<div class="container">
  <h1>Top 5 Themen – BERTopic</h1>
  <div class="explain">
    <strong>Erklärung:</strong> BERTopic nutzt semantische Embeddings und Clustering, 
    um ähnliche Texte automatisch zu gruppieren. Jedes Thema wird durch die wichtigsten Begriffe beschrieben.
  </div>
"""

for i, row in freq.iterrows():
    topic_words = ", ".join([w for w, _ in topic_model.get_topic(row["Topic"])[:5]])
    bertopic_html += f"<div class='topic'><h2>Themenbereich {i+1}</h2><p><strong>Top-Begriffe:</strong> {topic_words}</p><p><strong>Anzahl Betreffe:</strong> {row['Count']}</p></div>"

bertopic_html += "</div></body></html>"

with open("bertopic_top5.html", "w", encoding="utf-8") as f:
    f.write(bertopic_html)

print("BERTopic abgeschlossen. HTML gespeichert als 'bertopic_top5.html'.")

# ---------- LDA ----------
print("Starte LDA …")
texts = [t.split() for t in df["clean_subject"]]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10, random_state=42)

# Bestes Thema je Dokument
doc_topics = [max(lda_model.get_document_topics(bow), key=lambda x: x[1])[0] for bow in corpus]
df["lda_topic"] = doc_topics

lda_freq = df["lda_topic"].value_counts().head(5)

# HTML für LDA
lda_html = """
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>Top 5 Themen – LDA</title>
<style>
  body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
  .container { width: 100vw; height: 100vh; padding: 40px; box-sizing: border-box; }
  h1 { text-align: center; }
  .topic { margin: 20px 0; padding: 15px; border: 1px solid #ccc; border-radius: 8px; }
  .explain { background: #efe; padding: 10px; margin: 20px 0; border-radius: 8px; }
</style>
</head>
<body>
<div class="container">
  <h1>Top 5 Themen – LDA</h1>
  <div class="explain">
    <strong>Erklärung:</strong> LDA (Latent Dirichlet Allocation) ist ein probabilistisches Modell, 
    das jedes Dokument als Mischung von Themen darstellt. Jedes Thema besteht aus den wichtigsten Wörtern mit hoher Wahrscheinlichkeit.
  </div>
"""

for idx, count in lda_freq.items():
    words = ", ".join([w for w, _ in lda_model.show_topic(idx, topn=5)])
    lda_html += f"<div class='topic'><h2>Themenbereich {idx+1}</h2><p><strong>Top-Begriffe:</strong> {words}</p><p><strong>Anzahl Betreffe:</strong> {count}</p></div>"

lda_html += "</div></body></html>"

with open("lda_top5.html", "w", encoding="utf-8") as f:
    f.write(lda_html)

print("LDA abgeschlossen. HTML gespeichert als 'lda_top5.html'.")
```

---

📌 Änderungen gegenüber deinem letzten Stand:

* Multilinguales Embeddingmodell `paraphrase-multilingual-MiniLM-L12-v2`.
* Kombination deutscher + englischer Stopwortlisten.
* HTML-Ausgabe fullscreen, mit **deutscher Erklärung pro Modell**.
* Ausgabe:

  * `bertopic_top5.html`
  * `lda_top5.html`

---

👉 Soll ich dir auch eine **kleine Vorschau** geben, wie die beiden HTML-Dokumente visuell aussehen (z. B. ein Screenshot-ähnliches Mockup im Chat)?
