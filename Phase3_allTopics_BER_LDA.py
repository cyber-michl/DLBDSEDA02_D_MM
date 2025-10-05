import pandas as pd
import re
import nltk
import spacy 
from nltk.corpus import stopwords
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from typing import Dict, Any, Union 

# --- Globale Konfiguration und Setup ---

#Stoppwörter herunterladen, falls nicht vorhanden
try:
    nltk.data.find('corpora/stopwords')
except LookupError: 
    nltk.download('stopwords', quiet=True) 

german_stopwords = set(stopwords.words('german'))

#spaCy Modell für deutsche Lemmatisierung
try:
    #Nur notwendige Komponenten
    nlp = spacy.load("de_core_news_sm", disable=['ner', 'senter'])
except OSError:
    #Warnung
    print("Warnung: spaCy-Modell 'de_core_news_sm' nicht geladen. Lemmatisierung wird übersprungen.")
    nlp = None 


def export_all_topics(model: Union[BERTopic, LdaModel], model_name: str, filename: str, df: pd.DataFrame):
    """
    Extrahiert ALLE erkannten Themen (mit Top 10 Wörtern) aus dem Modell 
    und speichert sie in einer CSV-Datei.
    """
    topic_data = []
    
    if model_name == "BERTopic":
        topic_info = model.get_topic_info()
        topic_info = topic_info[topic_info['Topic'] != -1].sort_values("Count", ascending=False)
        
        for index, row in topic_info.iterrows():
            topic_id = row['Topic']
            top_words_info = model.get_topic(topic_id)
            top_words = {f"Wort {i+1}": w for i, (w, _) in enumerate(top_words_info[:10])}
            
            data = {
                "Modell": model_name,
                "Thema_ID": topic_id,
                "Anzahl_Dokumente": row['Count'],
                "Label_Intern": row['Name'],
                **top_words
            }
            topic_data.append(data)

    elif model_name == "LDA":
        all_topics = model.show_topics(num_topics=-1, num_words=10, formatted=False)
        
        texts_lda = [t.split() for t in df["clean_body"].tolist() if t] 
        dictionary_lda = corpora.Dictionary(texts_lda)
        dictionary_lda.filter_extremes(no_below=5, no_above=0.5) 
        corpus_lda = [dictionary_lda.doc2bow(text) for text in texts_lda]

        doc_counts = pd.Series([
            max(model.get_document_topics(bow), key=lambda x: x[1], default=(None, 0.0))[0] 
            for bow in corpus_lda
        ]).value_counts()
        
        for topic_id, words_info in all_topics:
            top_words = {f"Wort {i+1}": w for i, (w, _) in enumerate(words_info)}
            
            data = {
                "Modell": model_name,
                "Thema_ID": topic_id,
                "Anzahl_Dokumente": doc_counts.get(topic_id, 0),
                **top_words
            }
            topic_data.append(data)

    if topic_data:
        df_export = pd.DataFrame(topic_data)
        df_export.to_csv(filename, index=False, encoding="utf-8")
        print(f"Export erfolgreich: '{filename}'")
        return filename
    else:
        return None

# --- Hilfsfunktion für Diagramme ---
def create_bar_chart(data: Dict[str, Any], xlabel: str, title: str, color: str) -> str:
    """Erstellt ein horizontales Balkendiagramm und gibt es als Base64-String zurück."""
    labels = list(data.keys())
    counts = list(data.values())
    fig_height = max(5, len(labels) * 0.8) 
    plt.figure(figsize=(10, fig_height)) 
    plt.barh(labels, counts, color=color)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.gca().invert_yaxis() 
    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format="png")
    plt.close()
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# --- Vorverarbeitung ---
def preprocess_text(text: str) -> str:
    """Führt Bereinigung, Stoppwortentfernung und optionale spaCy-Lemmatisierung durch."""
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    if nlp:
        doc = nlp(text.lower())
        tokens = [
            token.lemma_ for token in doc 
            if token.is_alpha and 
            token.lemma_ not in german_stopwords and 
            len(token.lemma_) > 2
        ]
        return " ".join(tokens)
    else:
        text = text.lower()
        text = re.sub(r'[^a-zäöüß ]', ' ', text)
        tokens = [
            w for w in text.split() 
            if w not in german_stopwords and 
            len(w) > 2
        ]
        return " ".join(tokens)


# --- BERTopic Analyse ---
def run_bertopic(df: pd.DataFrame) -> BERTopic:
    embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2") 
    topic_model = BERTopic(embedding_model=embedding_model, language="german", min_topic_size=10, verbose=False) 
    docs = df["clean_body"].tolist()
    
    # BERTopic gibt bei verbose=False keine Ausgabe
    topics, _ = topic_model.fit_transform(docs)
    
    export_all_topics(topic_model, "BERTopic", "bertopic_alle_themen.csv", df)

    freq = topic_model.get_topic_freq().sort_values("Count", ascending=False)
    freq = freq[freq.Topic != -1].head(5) 
    
    if freq.empty: return topic_model

    chart_title = "Top 5 Themen – BERTopic"
    chart_data = {
        f"ID {row.Topic}: {' / '.join([w for w, _ in topic_model.get_topic(row.Topic)[:2]])}": row.Count 
        for row in freq.itertuples()
    }
    chart = create_bar_chart(dict(sorted(chart_data.items(), key=lambda item: item[1], reverse=False)), chart_title, "Anzahl Beschwerden", "steelblue")

    #HTML-Report
    html = f"""
    <!DOCTYPE html>..."""
    for i, row in enumerate(freq.itertuples(), start=1):
        topic_id = row.Topic
        top_words = [w for w, _ in topic_model.get_topic(topic_id)[:5]]
        label = f"ID {topic_id}: {' / '.join(top_words[:2])}"
        html += f"<div><h2>Thema {i} ({label})</h2><p>Schlüsselwörter: {', '.join(top_words)}</p><p>Anzahl: {row.Count}</p></div>"
        
    html += "</body></html>"
    with open("bertopic_top5.html", "w", encoding="utf-8") as f:
        f.write(html)
        
    print("Export erfolgreich: 'bertopic_top5.html'")
    return topic_model

# --- LDA Analyse ---
def run_lda(df: pd.DataFrame) -> LdaModel:
    """Führt die LDA-Analyse durch, bestimmt optimales K, exportiert alle Themen und generiert den HTML-Report."""
    texts = [t.split() for t in df["clean_body"].tolist() if t] 
    if not texts: return None

    dictionary = corpora.Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5) 
    corpus = [dictionary.doc2bow(text) for text in texts]

    #optimale Themenanzahl (K) mittels Coherence Score
    coherence_scores = []
    min_k, max_k_test = 5, 10 
    
    for k in range(min_k, max_k_test + 1):
        lda_tmp = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, passes=10, random_state=42)
        coherence_model = CoherenceModel(model=lda_tmp, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_scores.append((k, coherence_model.get_coherence()))

    optimal_k, _ = max(coherence_scores, key=lambda x: x[1])
    
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=optimal_k, passes=20, random_state=42)
    
    export_all_topics(lda_model, "LDA", "lda_alle_themen.csv", df)

    def get_dominant_topic(bow):
        return max(lda_model.get_document_topics(bow, minimum_probability=0.01), key=lambda x: x[1], default=(None, 0.0))
                   
    doc_topics = [get_dominant_topic(bow) for bow in corpus]
    dominant_topic_ids = [topic_id for topic_id, score in doc_topics if topic_id is not None]
    lda_freq = pd.Series(dominant_topic_ids).value_counts().head(5)

    if lda_freq.empty: return lda_model
        
    chart_title = "Top 5 Themen – LDA"
    topic_labels = {}
    for idx, count in lda_freq.items():
        words = [w for w, _ in lda_model.show_topic(idx, topn=3)]
        topic_labels[f"Thema {idx}: {' / '.join(words)}"] = count
    
    chart = create_bar_chart(dict(sorted(topic_labels.items(), key=lambda item: item[1], reverse=False)), chart_title, "Anzahl Beschwerden", "seagreen")

    #Generiere HTML-Report
    html = f"""
    <!DOCTYPE html>..."""
    for i, (idx, count) in enumerate(lda_freq.items(), start=1):
        words = [w for w, _ in lda_model.show_topic(idx, topn=5)]
        label = f"ID {idx}: {' / '.join(words[:2])}"
        html += f"<div><h2>Thema {i} ({label})</h2><p>Schlüsselwörter: {', '.join(words)}</p><p>Anzahl: {count}</p></div>"
        
    html += "</body></html>"
    with open("lda_top.html", "w", encoding="utf-8") as f:
        f.write(html)
        
    print("Export erfolgreich: 'lda_top.html'")
    return lda_model

# --- MAIN ---

if __name__ == "__main__":
    print("--- Starte NLP Themenanalyse ---") # Start
    try:
        df = pd.read_csv("multilingual_support_tickets.csv")
        df = df[df["language"] == "de"].copy() 
        
        if df.empty:
            print("FEHLER: Keine deutschsprachigen Einträge im Datensatz gefunden.")
        else:
            #Datenverarbeitung
            df["clean_body"] = df["body"].astype(str).apply(preprocess_text)
            df = df[df["clean_body"].str.len() > 0]
            df = df.drop_duplicates(subset=['clean_body'])
            df = df.reset_index(drop=True)

            topic_model = run_bertopic(df)
            lda_model = run_lda(df)
            
            print("--- Analyse erfolgreich abgeschlossen ---")
            
    except FileNotFoundError:
        print("FEHLER: Die Datei 'multilingual_support_tickets.csv' wurde nicht gefunden.")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
