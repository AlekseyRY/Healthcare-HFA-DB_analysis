import numpy as np
import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


df = pd.read_csv('Data_base/pivioted_fully_preprocessed_70-20.csv', infer_datetime_format = True, parse_dates=[1])
df = df.set_index(['country_region', 'year'])
body = list(df.columns)
body_cleaned = pd.read_csv('Data_base/body_cleaned_v1.csv').columns
spcy = spacy.load("ru_core_news_md")

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(body_cleaned)

def find_most_similar(query, top_n=10):
    similars = []
    # Обрабатываем входной запрос
    new_doc = spcy(query)
    new_text_cleaned = [" ".join(token.lemma_ for token in new_doc)]
    # Векторизуем обработанный запрос
    query_vec = vectorizer.transform(new_text_cleaned)
    # Вычисляем косинусное сходство между запросом и всеми текстами в корпусе
    similarities = cosine_similarity(query_vec, tfidf_matrix)

    # Получаем индексы наиболее похожих текстов
    similar_indices = similarities.argsort()[0][::-1][:top_n]

    # Выводим наиболее похожие тексты и их близость
    for idx in similar_indices:
        similars.append(body[idx])
    return similars