# DLBDSEDA02_D_MM
IU International University

1.1. Aufgabe 1: NLP-Techniken anwenden, um eine Textsammlung zu analysieren


Im Rahmen des Portfolios wurde ein Datenanalyseprojekt realisiert, das darauf abzielt, häufig genannte Themen in deutschsprachigen Beschwerdetexten mittels Natural Language Processing (NLP) zu identifizieren. Ziel war es, auf Grundlage der identifizierten Inhalte datenbasierte Erkenntnisse über die größten Unzufriedenheiten zu gewinnen. Die Ergebnisse sollen exemplarisch als Input für kommunale Entscheidungsträger dienen.

Die Umsetzung erfolgte in einer virtuellen Python-Umgebung mit venv und einer separaten requirements.txt zur Bibliotheksverwaltung. Verwendet wurde der Kaggle-Datensatz „Multilingual Customer Support Tickets“. Daraus wurden gezielt die deutschsprachigen Einträge extrahiert.

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
  
Verwendete Bibliotheken:
pandas, nltk, spaCy, scikit-learn, gensim, BERTopic, sentence-transformers, matplotlib
