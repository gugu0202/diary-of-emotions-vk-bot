# -*- coding: utf-8 -*-
"""Полная категоризация текстов с SMOTE на TF-IDF (исправленная версия)"""

# 1. Установка необходимых библиотек (если ещё не установлены)
!pip install pandas numpy scikit-learn pymystem3 nltk ruwordnet openpyxl imbalanced-learn

# 2. Импорт модулей
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

from pymystem3 import Mystem
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter

import urllib.request
import os
import subprocess
import sys
import shutil
import glob
from ruwordnet import RuWordNet

# 3. Загрузка данных (укажите правильные пути к файлам)
labeled_df = pd.read_excel('/content/размеченные тексты.xlsx')
unlabeled_df = pd.read_excel('/content/оставшиеся_тексты.xlsx')

X_labeled_raw = labeled_df['text'].tolist()
y_labeled = labeled_df['cat'].tolist()
X_unlabeled_raw = unlabeled_df['text'].tolist()

# 4. Предобработка текста (лемматизация, очистка, стоп-слова)
mystem = Mystem()
stop_words = set(stopwords.words('russian'))
# Дополнительные стоп-слова при необходимости
extra_stop = {'это', 'все', 'так', 'вот', 'был', 'еще', 'его', 'она', 'они',
              'меня', 'тебя', 'нас', 'вас', 'который', 'эта', 'эти', 'этот'}
stop_words.update(extra_stop)

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^а-яё\s]', ' ', text)
    lemmas_raw = mystem.lemmatize(text)
    lemmas = []
    for word in lemmas_raw:
        word = word.strip()
        if word and re.match(r'^[а-яё]+$', word) and len(word) > 1 and word not in stop_words:
            lemmas.append(word)
    return lemmas

X_labeled_clean = [' '.join(preprocess(text)) for text in X_labeled_raw]
X_unlabeled_clean = [' '.join(preprocess(text)) for text in X_unlabeled_raw]

# 5. Выделение характерных лексем (top-50 на категорию) и расширение синонимами
vectorizer = CountVectorizer(ngram_range=(1, 2), min_df=5, max_features=5000)
X_train_counts = vectorizer.fit_transform(X_labeled_clean)
feature_names = vectorizer.get_feature_names_out()
le = LabelEncoder()
y_enc = le.fit_transform(y_labeled)

# MI scores
mi_scores = mutual_info_classif(X_train_counts, y_enc, random_state=42)

n_top_features = 50
categories = le.classes_
category_top_features = {}

for cat_idx, cat_name in enumerate(categories):
    mask = (y_enc == cat_idx)
    mean_in_cat = X_train_counts[mask].mean(axis=0).A1
    mean_other = X_train_counts[~mask].mean(axis=0).A1
    mean_other[mean_other == 0] = 1e-6
    ratio = mean_in_cat / mean_other
    threshold = np.percentile(mi_scores, 70)
    candidate_indices = np.where((mi_scores > threshold) & (ratio > 1.2))[0]
    sorted_indices = candidate_indices[np.argsort(-ratio[candidate_indices])]
    top_indices = sorted_indices[:n_top_features]
    top_features = [feature_names[i] for i in top_indices]
    category_top_features[cat_name] = top_features

# 6. Подключение RuWordNet (автоматическая загрузка базы)
def setup_ruwordnet():
    db_dir = os.path.join(os.path.expanduser('~'), '.ruwordnet')
    os.makedirs(db_dir, exist_ok=True)
    target = os.path.join(db_dir, 'ruwordnet-2021.db')
    if os.path.exists(target):
        return RuWordNet()
    try:
        import ruwordnet
        pkg_dir = os.path.dirname(ruwordnet.__file__)
        candidates = glob.glob(os.path.join(pkg_dir, '**', '*.db'), recursive=True)
        if candidates:
            shutil.copy2(candidates[0], target)
            print(f'База скопирована из пакета: {candidates[0]}')
            return RuWordNet()
    except Exception:
        pass
    try:
        print("Загружаем базу командой 'ruwordnet download'...")
        subprocess.run([sys.executable, '-m', 'ruwordnet', 'download'], check=True)
        if os.path.exists(target):
            return RuWordNet()
        legacy = os.path.join(db_dir, 'ruwordnet.db')
        if os.path.exists(legacy):
            os.rename(legacy, target)
            return RuWordNet()
    except Exception as e:
        print(f"Команда не удалась: {e}")
    url = "https://github.com/avidale/python-ruwordnet/raw/master/ruwordnet/static/ruwordnet-2021.db"
    print(f"Скачиваем базу с {url}...")
    try:
        urllib.request.urlretrieve(url, target)
        if os.path.exists(target):
            return RuWordNet()
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
    raise FileNotFoundError("Не удалось получить базу ruwordnet-2021.db")

wordnet = setup_ruwordnet()

def get_synonyms(word):
    synonyms = set()
    try:
        for synset in wordnet.get_synsets(word):
            for sense in synset.senses:
                synonyms.add(sense.lemma)
    except:
        pass
    return synonyms

# Расширяем топ-фичи синонимами (только униграммы)
category_expanded_units = {}
for cat_name, top_feats in category_top_features.items():
    expanded = set()
    for feat in top_feats:
        expanded.add(feat)
        if ' ' not in feat:  # униграммы
            expanded.update(get_synonyms(feat))
    category_expanded_units[cat_name] = list(expanded)

# 7. Построение словаря и BOW-представление
all_units = set()
for units in category_expanded_units.values():
    all_units.update(units)
vocabulary = {word: idx for idx, word in enumerate(all_units)}
bow_vectorizer = CountVectorizer(vocabulary=vocabulary)
X_labeled_bow = bow_vectorizer.fit_transform(X_labeled_clean)
X_unlabeled_bow = bow_vectorizer.transform(X_unlabeled_clean)

# 8. Применение SMOTE (если минимальный размер класса >= 3)
label_counts = Counter(y_enc)
min_class_size = min(label_counts.values())
print(f"Минимальный размер класса в размеченных данных: {min_class_size}")

if min_class_size >= 3:
    k = min(5, min_class_size - 1)
    print(f"Используем SMOTE с k_neighbors={k}")
    smote = SMOTE(k_neighbors=k, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_labeled_bow, y_enc)
    print("SMOTE применён. Размер выборки после балансировки:", X_resampled.shape[0])
else:
    print("Слишком мало образцов в одном из классов. SMOTE не используется, применяем class_weight='balanced'.")
    X_resampled, y_resampled = X_labeled_bow, y_enc

# 9. Обучение классификатора
clf = LogisticRegression(multi_class='multinomial', max_iter=1000,
                         class_weight='balanced', random_state=42)
clf.fit(X_resampled, y_resampled)
y_pred_enc = clf.predict(X_unlabeled_bow)
y_pred = le.inverse_transform(y_pred_enc)

# 10. Сохранение результатов
output_df = pd.DataFrame({'text': X_unlabeled_raw, 'category': y_pred})
output_df.to_excel('balanced_categorized.xlsx', index=False)
print("Категоризация завершена. Результаты сохранены в balanced_categorized.xlsx")