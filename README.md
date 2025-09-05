# DLBDSEDA02_D_MM
IU International University

1.1. Aufgabe 1: NLP-Techniken anwenden, um eine Textsammlung zu analysieren


Im Rahmen des Portfolios wurde ein Datenanalyseprojekt realisiert, das darauf abzielt, häufig genannte Themen in deutschsprachigen Beschwerdetexten mittels Natural Language Processing (NLP) zu identifizieren. Ziel war es, auf Grundlage der identifizierten Inhalte datenbasierte Erkenntnisse über die größten Unzufriedenheiten zu gewinnen. Die Ergebnisse sollen exemplarisch als Input für kommunale Entscheidungsträger dienen.

Die Umsetzung erfolgte in einer **virtuellen Python-Umgebung mit conda**.  
Zur Verwaltung und Reproduzierbarkeit der Abhängigkeiten wurde eine separate **`environment.yml`** verwendet.  

Als Datengrundlage diente der Kaggle-Datensatz **„Multilingual Customer Support Tickets“**.  
Daraus wurden gezielt die **deutschsprachigen Einträge** extrahiert und für die Analyse genutzt.

Ziel war es, mit Hilfe von **NLP-Techniken** die am häufigsten angesprochenen Themen aus den Beschwerden
zu extrahieren und diese für Entscheidungsträger verständlich aufzubereiten.

Es wurden zwei unterschiedliche semantische Analysetechniken eingesetzt:
- **BERTopic** (Clustering basierend auf Sentence-Transformers)
- **LDA (Latent Dirichlet Allocation)**

Die Ergebnisse werden sowohl in Form von **HTML-Reports** mit den Top 5 Themen als auch
in **Balkendiagrammen** dargestellt.

Die Vorverarbeitung beinhaltete:
- Unicode- und Zeichensäuberung
- Entfernung von Stoppwörtern mit nltk
- Lemmatisierung mit spaCy (Modell de_core_news_sm)

Zur numerischen Repräsentation der Texte wurden zwei Verfahren eingesetzt:
1. TF-IDF-Vektorisierung über scikit-learn
2. Sentence Embeddings mittels sentence-transformers (Modell: paraphrase-multilingual-MiniLM-L12-v2)

Themenextraktionstechniken:
- Latent Dirichlet Allocation (LDA) (gensim)
- BERTopic (semantisches Clustering basierend auf Embeddings + HDBSCAN)

Umgebung automatisch aufsetzen:
- environment.yml (conda)

conda env create -f environment.yml

conda activate nlp-beschwerden

import nltk

nltk.download("stopwords")


- requirements.txt

pip install -r requirements.txt

python -m spacy download de_core_news_sm

import nltk

nltk.download("stopwords")


Verwendete Bibliotheken:
pandas, nltk, spaCy, scikit-learn, gensim, BERTopic, sentence-transformers, matplotlib
