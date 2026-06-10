# -*- coding: utf-8 -*-
"""Сохранение_артефактов_для_бота
"""

!pip install vk_api

!pip install pymorphy3

"""#СОХРАНЕНИЕ АРТЕФАКТОВ
Так как Лог.регрессия показала лучшие метрики, берем ее для сохранения артефактов
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import pymorphy3
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

# Загружаем стоп-слова (используем готовый ручной набор из пункта 2)
RUSSIAN_STOP_WORDS = {
    'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'но', 'его',
    'по', 'из', 'у', 'от', 'о', 'же', 'за', 'бы', 'мне', 'меня', 'мной', 'ты', 'тебе', 'тебя', 'тобой', 'мы',
    'нас', 'нам', 'нами', 'вы', 'вас', 'вам', 'вами', 'они', 'их', 'им', 'ими', 'себя', 'себе', 'собой',
    'это', 'этот', 'эта', 'эти', 'того', 'тому', 'тем', 'том', 'этом', 'этом', 'этой', 'этою', 'тех', 'этим',
    'кто', 'когда', 'где', 'какой', 'какая', 'какое', 'какие', 'какого', 'какому', 'каким', 'каком', 'какой',
    'какую', 'какою', 'какие', 'каких', 'каким', 'какими', 'который', 'которая', 'которое', 'которые',
    'которого', 'которому', 'которым', 'котором', 'которую', 'которою', 'которые', 'которых', 'которым',
    'которыми', 'есть', 'быть', 'был', 'была', 'было', 'были', 'буду', 'будет', 'будем', 'будут', 'для', 'да',
    'нет', 'или', 'если', 'уже', 'ещё', 'тот', 'там', 'тут', 'весь', 'все', 'всё', 'свой', 'своя', 'своё',
    'свои', 'мой', 'твой', 'наш', 'ваш', 'её', 'его', 'их', 'один', 'одна', 'одно', 'одни', 'два', 'три',
    'много', 'мало', 'очень', 'бы', 'ли', 'уж', 'ни', 'раз', 'такой', 'такая', 'такое', 'такие', 'также',
    'тоже', 'это', 'до', 'после', 'между', 'над', 'под', 'при', 'про', 'без', 'через', 'чтобы', 'чтоб',
    'к', 'ко', 'об', 'обо', 'из-за', 'из-под', 'около', 'вокруг', 'вроде', 'вместо', 'внутри', 'вне',
    'лишь', 'только', 'ещё', 'уже', 'всегда', 'иногда', 'сейчас', 'теперь', 'тогда', 'потом', 'пока',
    'даже', 'ведь', 'хотя', 'может', 'можно', 'надо', 'нужно', 'нельзя', 'сегодня', 'вчера', 'завтра',
    'тут', 'там', 'здесь', 'везде', 'нигде', 'всего', 'ничего', 'кто-то', 'что-то', 'где-то', 'когда-то'
}
stop_words = RUSSIAN_STOP_WORDS

# Инициализируем морфологический анализатор
morph = pymorphy3.MorphAnalyzer()

# Функция предобработки текста
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^а-яё\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    lemmas = []
    for word in words:
        if word not in stop_words and len(word) > 1:
            lemma = morph.parse(word)[0].normal_form
            lemmas.append(lemma)
    return ' '.join(lemmas)

# 1. Загрузка данных
df = pd.read_excel('balanced_categorized.xlsx')
print("Исходное распределение классов:")
print(df['category'].value_counts())

# 2. Предобработка текста
df['clean_text'] = df['text'].apply(preprocess_text)
df = df.dropna(subset=['clean_text', 'category'])

# 3. Удаляем классы с количеством примеров меньше 2 (для стратификации)
class_counts = df['category'].value_counts()
rare_classes = class_counts[class_counts < 2].index.tolist()
if rare_classes:
    print(f"Удаляем классы с <2 примерами: {rare_classes}")
    df = df[~df['category'].isin(rare_classes)]
    print("Новое распределение классов:")
    print(df['category'].value_counts())

# 4. Кодирование категорий в числа (LabelEncoder)
le = LabelEncoder()
y = le.fit_transform(df['category'])
cat_mapping = {i: name for i, name in enumerate(le.classes_)}
print("Маппинг категорий:", cat_mapping)

X = df['clean_text']

# 5. Разделение на обучающую и тестовую выборки (стратифицированно)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 6. Векторизация TF-IDF (как в пункте 1)
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"Размерность признаков: {X_train_tfidf.shape[1]}")

# 7. Модель логистической регрессии (как в пункте 1: saga, max_iter=1000)
logreg = LogisticRegression(
    solver='saga',
    max_iter=1000,
    random_state=42
)

# Подбор гиперпараметра C (как в пункте 1)
param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
grid = GridSearchCV(logreg, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid.fit(X_train_tfidf, y_train)

print(f"\nЛучший параметр C: {grid.best_params_['C']}")
print(f"Лучший F1 (взвешенный) на кросс-валидации: {grid.best_score_:.4f}")

best_model = grid.best_estimator_

# Оценка на тестовой выборке
y_pred = best_model.predict(X_test_tfidf)

if len(y_test) > 0 and len(set(y_test)) > 0:
    target_names = [str(c) for c in le.classes_]
    report = classification_report(
        y_test, y_pred,
        target_names=target_names,
        zero_division=0
    )
    print("\nОтчёт по тестовой выборке:")
    print(report)
else:
    print("\n⚠️ Тестовая выборка пуста или состоит из одного класса – отчёт не может быть сформирован.")

# 8. Сохранение артефактов
with open('model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('tfidf.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

with open('cat_mapping.json', 'w', encoding='utf-8') as f:
    clean_mapping = {int(k): str(v) for k, v in cat_mapping.items()}
    json_mapping = {str(k): v for k, v in clean_mapping.items()}
    json.dump(json_mapping, f, ensure_ascii=False, indent=2)

print("\nАртефакты сохранены: model.pkl, tfidf.pkl, cat_mapping.json")

