import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from gensim import corpora
from gensim.models.ldamodel import LdaModel
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# NLTK Ressourcen
nltk.download('stopwords')

# deutsche Stopwörter verwenden
german_stopwords = set(stopwords.words('german'))

# CSV laden
df = pd.read_csv("multilingual_support_tickets.csv")
print("Spalten:", df.columns)

# Nur deutsche Einträge
df = df[df["language"] == "de"]

# Texte bereinigen
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zäöüß ]', ' ', text)
    tokens = [w for w in text.split() if w not in german_stopwords and len(w) > 2]
    return " ".join(tokens)

df["clean_body"] = df["body"].astype(str).apply(clean_text)
print("Anzahl deutschsprachiger Beschwerden:", len(df))

# BERTopic
print("Starte BERTopic")
embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
topic_model = BERTopic(embedding_model=embedding_model, language="german")

topics, _ = topic_model.fit_transform(df["clean_body"])
freq = topic_model.get_topic_freq().sort_values("Count", ascending=False).head(5)

# Balkendiagramm BERTopic
plt.figure(figsize=(8, 5))
plt.barh([f"Thema {i+1}" for i in range(len(freq))], freq["Count"], color="steelblue")
plt.xlabel("Anzahl Beschwerden")
plt.title("Top 5 Themen – BERTopic")
plt.tight_layout()
buf = BytesIO()
plt.savefig(buf, format="png")
plt.close()
bertopic_chart = base64.b64encode(buf.getvalue()).decode("utf-8")

# HTML für BERTopic
bertopic_html = """
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>Top 5 Themen – BERTopic</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
    .container {{ width: 100vw; height: 100vh; padding: 40px; box-sizing: border-box; }}
    h1 {{ text-align: center; }}
    .topic {{ margin: 20px 0; padding: 15px; border: 1px solid #ccc; border-radius: 8px; }}
    .explain {{ background: #eef; padding: 10px; margin: 20px 0; border-radius: 8px; }}
</style>
</head>
<body>
<div class="container">
    <h1>Top 5 Themen – BERTopic</h1>
    <div class="explain">
        <strong>Erklärung:</strong> BERTopic nutzt semantische Embeddings und Clustering,
        um ähnliche Texte automatisch zu gruppieren.
    </div>
    <img src="data:image/png;base64,{chart}" alt="Balkendiagramm">
""".format(chart=bertopic_chart)

# Durchnummerierung 1–5
for i, row in enumerate(freq.itertuples(), start=1):
    top_words = [w for w, _ in topic_model.get_topic(row.Topic)[:5]]
    label = " / ".join(top_words[:2])  # Automatisches Label
    bertopic_html += f"<div class='topic'><h2>Thema {i}: {label}</h2><p><strong>Top-Begriffe:</strong> {', '.join(top_words)}</p><p><strong>Anzahl Beschwerden:</strong> {row.Count}</p></div>"

bertopic_html += "</div></body></html>"

with open("bertopic_top5.html", "w", encoding="utf-8") as f:
    f.write(bertopic_html)

print("BERTopic abgeschlossen. HTML gespeichert als 'bertopic_top5.html'.")

# LDA
print("Starte LDA")
texts = [t.split() for t in df["clean_body"]]
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, passes=10, random_state=42)

# Bestes Thema
doc_topics = [max(lda_model.get_document_topics(bow), key=lambda x: x[1])[0] for bow in corpus]
df["lda_topic"] = doc_topics
lda_freq = df["lda_topic"].value_counts().head(5)

# Balkendiagramm LDA
plt.figure(figsize=(8, 5))
plt.barh([f"Thema {i+1}" for i in range(len(lda_freq))], lda_freq.values, color="seagreen")
plt.xlabel("Anzahl Beschwerden")
plt.title("Top 5 Themen – LDA")
plt.tight_layout()
buf = BytesIO()
plt.savefig(buf, format="png")
plt.close()
lda_chart = base64.b64encode(buf.getvalue()).decode("utf-8")

# HTML für LDA
lda_html = """
<!DOCTYPE html>
<html lang="de">
<head>
<meta charset="UTF-8">
<title>Top 5 Themen – LDA</title>
<style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; }}
    .container {{ width: 100vw; height: 100vh; padding: 40px; box-sizing: border-box; }}
    h1 {{ text-align: center; }}
    .topic {{ margin: 20px 0; padding: 15px; border: 1px solid #ccc; border-radius: 8px; }}
    .explain {{ background: #efe; padding: 10px; margin: 20px 0; border-radius: 8px; }}
</style>
</head>
<body>
<div class="container">
    <h1>Top 5 Themen – LDA</h1>
    <div class="explain">
        <strong>Erklärung:</strong> LDA (Latent Dirichlet Allocation)
    </div>
    <img src="data:image/png;base64,{chart}" alt="Balkendiagramm">
""".format(chart=lda_chart)

# Durchnummerierung 1–5
for i, (idx, count) in enumerate(lda_freq.items(), start=1):
    words = [w for w, _ in lda_model.show_topic(idx, topn=5)]
    label = " / ".join(words[:2])
    lda_html += f"<div class='topic'><h2>Thema {i}: {label}</h2><p><strong>Top-Begriffe:</strong> {', '.join(words)}</p><p><strong>Anzahl Beschwerden:</strong> {count}</p></div>"

lda_html += "</div></body></html>"

with open("lda_top5.html", "w", encoding="utf-8") as f:
    f.write(lda_html)

print("LDA abgeschlossen. HTML gespeichert als 'lda_top5.html'.")