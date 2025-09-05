import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import base64
from io import BytesIO

# ----------------- NLTK -----------------
nltk.download('stopwords')
german_stopwords = set(stopwords.words('german'))

# ----------------- Textbereinigung -----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zäöüß ]', ' ', text)
    tokens = [w for w in text.split() if w not in german_stopwords and len(w) > 2]
    return " ".join(tokens)

# ----------------- BERTopic -----------------
def run_bertopic(df):
    print("Starte BERTopic")
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    topic_model = BERTopic(embedding_model=embedding_model, language="german")
    topics, _ = topic_model.fit_transform(df["clean_body"])
    freq = topic_model.get_topic_freq().sort_values("Count", ascending=False).head(5)

    # Balkendiagramm
    plt.figure(figsize=(8, 5))
    plt.barh([f"Thema {i+1}" for i in range(len(freq))], freq["Count"], color="steelblue")
    plt.xlabel("Anzahl Beschwerden")
    plt.title("Top 5 Themen – BERTopic")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    chart = base64.b64encode(buf.getvalue()).decode("utf-8")

    # HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="de">
    <head><meta charset="UTF-8"><title>Top 5 Themen – BERTopic</title></head>
    <body>
    <h1>Top 5 Themen – BERTopic</h1>
    <img src="data:image/png;base64,{chart}" alt="Balkendiagramm">
    """
    for i, row in enumerate(freq.itertuples(), start=1):
        top_words = [w for w, _ in topic_model.get_topic(row.Topic)[:5]]
        label = " / ".join(top_words[:2])
        html += f"<div><h2>Thema {i}: {label}</h2><p>Top-Begriffe: {', '.join(top_words)}</p><p>Anzahl Beschwerden: {row.Count}</p></div>"
    html += "</body></html>"

    with open("bertopic_top5.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("BERTopic abgeschlossen. HTML gespeichert als 'bertopic_top5.html'.")
    return topic_model

# ----------------- LDA -----------------
def run_lda(df):
    print("Starte LDA")
    texts = [t.split() for t in df["clean_body"]]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # Coherence Score
    coherence_scores = []
    for k in range(3, 11):
        lda_tmp = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=10, random_state=42)
        coherence_model = CoherenceModel(model=lda_tmp, texts=texts, dictionary=dictionary, coherence='c_v')
        score = coherence_model.get_coherence()
        coherence_scores.append((k, score))
    optimal_k = max(coherence_scores, key=lambda x: x[1])[0]
    print("Optimale Themenanzahl:", optimal_k)

    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=optimal_k, passes=10, random_state=42)

    # Themen extrahieren
    doc_topics = [max(lda_model.get_document_topics(bow), key=lambda x: x[1])[0] for bow in corpus]
    df["lda_topic"] = doc_topics
    lda_freq = df["lda_topic"].value_counts().head(5)

    # Balkendiagramm
    plt.figure(figsize=(8, 5))
    plt.barh([f"Thema {i+1}" for i in range(len(lda_freq))], lda_freq.values, color="seagreen")
    plt.xlabel("Anzahl Beschwerden")
    plt.title("Top 5 Themen – LDA")
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    chart = base64.b64encode(buf.getvalue()).decode("utf-8")

    # HTML
    html = f"""
    <!DOCTYPE html>
    <html lang="de">
    <head><meta charset="UTF-8"><title>Top Themen – LDA</title></head>
    <body>
    <h1>Top 5 Themen – LDA</h1>
    <img src="data:image/png;base64,{chart}" alt="Balkendiagramm">
    """
    for i, (idx, count) in enumerate(lda_freq.items(), start=1):
        words = [w for w, _ in lda_model.show_topic(idx, topn=5)]
        label = " / ".join(words[:2])
        html += f"<div><h2>Thema {i}: {label}</h2><p>Top-Begriffe: {', '.join(words)}</p><p>Anzahl Beschwerden: {count}</p></div>"
    html += "</body></html>"

    with open("lda_top5.html", "w", encoding="utf-8") as f:
        f.write(html)
    print("LDA abgeschlossen. HTML gespeichert als 'lda_top5.html'.")
    return lda_model

# ----------------- Main -----------------
if __name__ == "__main__":
    # CSV laden und bereinigen
    df = pd.read_csv("multilingual_support_tickets.csv")
    df = df[df["language"] == "de"]
    df["clean_body"] = df["body"].astype(str).apply(clean_text)

    # BERTopic
    topic_model = run_bertopic(df)

    # LDA
    lda_model = run_lda(df)
