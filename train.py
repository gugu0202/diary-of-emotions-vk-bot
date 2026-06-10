# -*- coding: utf-8 -*-

#  УСТАНОВКА БИБЛИОТЕК 
# Обновляем проблемные библиотеки
!pip install --upgrade tensorflow
!pip install --upgrade pymorphy3

# Устанавливаем остальные библиотеки
!pip install pandas openpyxl scikit-learn imbalanced-learn nltk gensim sentence-transformers xgboost ruwordnet

#  ИМПОРТЫ 
import pandas as pd
import numpy as np
import re
import warnings
import random
import nltk
import os
import sys
import subprocess
import shutil
import glob
import urllib.request
from collections import Counter

# Предобработка текста
from nltk.corpus import stopwords
import pymorphy3  # Заменяем pymorphy2 на pymorphy3

# ML
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.class_weight import compute_class_weight

# SMOTE и пайплайны
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Embedding, LSTM, Bidirectional, Dense, Dropout,
                                     Conv1D, GlobalMaxPooling1D, Concatenate, BatchNormalization)
from tensorflow.keras.callbacks import EarlyStopping

# Sentence Transformers
from sentence_transformers import SentenceTransformer

# XGBoost
import xgboost as xgb

# RuWordNet
from ruwordnet import RuWordNet

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
nltk.download('stopwords')

#  ПАРАМЕТРЫ 
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_FEATURES_TFIDF = 10000
MAX_SEQUENCE_LENGTH = 200
MAX_NB_WORDS = 10000
EMBEDDING_DIM = 128

#  ФУНКЦИЯ ЗАГРУЗКИ RUWORDNET 
def setup_ruwordnet():
    """Обеспечивает наличие файла ruwordnet-2021.db в ~/.ruwordnet/ и возвращает объект RuWordNet."""
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

    raise FileNotFoundError(
        "Не удалось получить базу ruwordnet-2021.db. "
        "Скачайте вручную с https://github.com/nlpub/ruwordnet/releases/tag/v2021.01 "
        "и положите файл в ~/.ruwordnet/"
    )

print("Инициализация RuWordNet...")
try:
    wordnet = setup_ruwordnet()
    print("RuWordNet успешно загружен")
except Exception as e:
    print(f"Ошибка загрузки RuWordNet: {e}")
    print("Аугментация синонимами будет недоступна, используется fallback-аугментация.")
    wordnet = None

#  ЗАГРУЗКА И ПРЕДОБРАБОТКА ДАННЫХ 
print("Загрузка данных...")
df = pd.read_excel('balanced_categorized.xlsx')

if 'text' not in df.columns or 'category' not in df.columns:
    raise ValueError("Файл должен содержать колонки 'text' и 'category'")

# Фильтрация классов с количеством записей < 2
class_counts = df['category'].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df = df[df['category'].isin(valid_classes)].copy()
print("Распределение классов после фильтрации (<2 удалены):")
print(df['category'].value_counts())

# Предобработка текста (лемматизация через pymorphy3)
stop_words = set(stopwords.words('russian'))
morph = pymorphy3.MorphAnalyzer()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^а-яё\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    lemmas = []
    for w in words:
        if w not in stop_words and len(w) > 1:
            # Получаем нормальную форму (лемму) через pymorphy3
            lemma = morph.parse(w)[0].normal_form
            lemmas.append(lemma)
    return ' '.join(lemmas)

print("Лемматизация текстов...")
df['clean_text'] = df['text'].apply(preprocess_text)
df = df[df['clean_text'].str.strip() != '']
print("Предобработка завершена.")

# Кодирование меток
le = LabelEncoder()
df['label'] = le.fit_transform(df['category'])
target_names = [str(cls) for cls in le.classes_]
n_classes = len(target_names)
print(f"Классы: {target_names}")

X = df['clean_text']
y = df['label']

# Разделение на train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

#  ФУНКЦИЯ БЕЗОПАСНОГО SMOTE 
def get_smote(k_neighbors=5, random_state=RANDOM_STATE):
    min_class_size = min(Counter(y_train).values())
    if min_class_size >= k_neighbors + 1:
        print(f"SMOTE применяется с k_neighbors={k_neighbors}")
        return SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    else:
        print(f"Внимание: минимальный размер класса {min_class_size} < {k_neighbors+1}. SMOTE не используется. Будет использован class_weight='balanced'.")
        return None

#  ФУНКЦИИ АУГМЕНТАЦИИ (с ruwordnet) 
def get_synonyms_ruwordnet(word):
    """Возвращает список синонимов для слова (лемма) из RuWordNet."""
    if wordnet is None:
        return []
    try:
        synsets = wordnet.get_synsets(word)
        synonyms = set()
        for synset in synsets:
            for sense in synset.senses:
                synonyms.add(sense.lemma)
        return list(synonyms)
    except:
        return []

def synonym_replace_ruwordnet(text, prob=0.3):
    """Заменяет слова на синонимы с вероятностью prob."""
    words = text.split()
    if len(words) < 2:
        return text
    new_words = []
    for w in words:
        if random.random() < prob:
            syns = get_synonyms_ruwordnet(w)
            if syns:
                new_words.append(random.choice(syns))
                continue
        new_words.append(w)
    if new_words == words and len(words) >= 2:
        idx = random.randint(0, len(words)-2)
        words[idx], words[idx+1] = words[idx+1], words[idx]
        return ' '.join(words)
    return ' '.join(new_words)

def augment_text(text, shuffle_prob=0.3, delete_prob=0.2, synonym_prob=0.3):
    """Комбинированная аугментация: синонимы, перестановка, удаление."""
    text = synonym_replace_ruwordnet(text, prob=synonym_prob)
    words = text.split()
    if len(words) < 2:
        return text
    if random.random() < shuffle_prob:
        i1, i2 = random.sample(range(len(words)), 2)
        words[i1], words[i2] = words[i2], words[i1]
    if len(words) > 1 and random.random() < delete_prob:
        del words[random.randrange(len(words))]
    return ' '.join(words)

#  1. NAIVE BAYES 
print("\n" + "="*60)
print("1. NAIVE BAYES (MultinomialNB)")
print("="*60)

tfidf = TfidfVectorizer(max_features=MAX_FEATURES_TFIDF, ngram_range=(1,2))
X_train_tf = tfidf.fit_transform(X_train)
X_test_tf = tfidf.transform(X_test)

smote_nb = get_smote(k_neighbors=5)
if smote_nb is not None:
    X_train_res, y_train_res = smote_nb.fit_resample(X_train_tf, y_train)
else:
    X_train_res, y_train_res = X_train_tf, y_train

nb = MultinomialNB()
nb.fit(X_train_res, y_train_res)
y_pred_nb = nb.predict(X_test_tf)
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb, target_names=target_names, zero_division=0))

#  2. SVM 
print("\n" + "="*60)
print("2. SVM")
print("="*60)

tfidf_svm = TfidfVectorizer(max_features=MAX_FEATURES_TFIDF, ngram_range=(1,2))
X_train_tf_svm = tfidf_svm.fit_transform(X_train)
X_test_tf_svm = tfidf_svm.transform(X_test)

smote_svm = get_smote(k_neighbors=5)
if smote_svm is not None:
    X_train_res_svm, y_train_res_svm = smote_svm.fit_resample(X_train_tf_svm, y_train)
else:
    X_train_res_svm, y_train_res_svm = X_train_tf_svm, y_train

param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']}
svm = SVC(class_weight='balanced' if smote_svm is None else None, random_state=RANDOM_STATE)
grid_svm = GridSearchCV(svm, param_grid_svm, cv=3, scoring='f1_weighted', n_jobs=-1)
grid_svm.fit(X_train_res_svm, y_train_res_svm)
print("Лучшие параметры:", grid_svm.best_params_)
y_pred_svm = grid_svm.predict(X_test_tf_svm)
print("Accuracy:", accuracy_score(y_test, y_pred_svm))
print(classification_report(y_test, y_pred_svm, target_names=target_names, zero_division=0))

#  3. LOGISTIC REGRESSION 
print("\n" + "="*60)
print("3. LOGISTIC REGRESSION")
print("="*60)

tfidf_lr = TfidfVectorizer(max_features=MAX_FEATURES_TFIDF, ngram_range=(1,2))
X_train_tf_lr = tfidf_lr.fit_transform(X_train)
X_test_tf_lr = tfidf_lr.transform(X_test)

smote_lr = get_smote(k_neighbors=5)
if smote_lr is not None:
    X_train_res_lr, y_train_res_lr = smote_lr.fit_resample(X_train_tf_lr, y_train)
else:
    X_train_res_lr, y_train_res_lr = X_train_tf_lr, y_train

lr = LogisticRegression(solver='saga', max_iter=1000,
                        class_weight='balanced' if smote_lr is None else None,
                        random_state=RANDOM_STATE)
param_grid_lr = {'C': [0.01, 0.1, 1, 10, 100]}
grid_lr = GridSearchCV(lr, param_grid_lr, cv=5, scoring='f1_weighted', n_jobs=-1)
grid_lr.fit(X_train_res_lr, y_train_res_lr)
print("Лучший C:", grid_lr.best_params_['C'])
y_pred_lr = grid_lr.predict(X_test_tf_lr)
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr, target_names=target_names, zero_division=0))

#  4. CNN 
print("\n" + "="*60)
print("4. CNN (1D)")
print("="*60)

tokenizer_cnn = Tokenizer(num_words=MAX_NB_WORDS, oov_token='<OOV>')
tokenizer_cnn.fit_on_texts(X_train)
X_train_seq = tokenizer_cnn.texts_to_sequences(X_train)
X_test_seq = tokenizer_cnn.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
vocab_size_cnn = min(MAX_NB_WORDS, len(tokenizer_cnn.word_index) + 1)

model_cnn = Sequential([
    Embedding(vocab_size_cnn, EMBEDDING_DIM, input_length=MAX_SEQUENCE_LENGTH),
    Conv1D(filters=128, kernel_size=5, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])
model_cnn.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_cnn.summary()

class_weights_cnn = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict_cnn = dict(enumerate(class_weights_cnn))
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history_cnn = model_cnn.fit(X_train_pad, y_train, validation_split=0.2, epochs=15, batch_size=32,
                            class_weight=class_weight_dict_cnn, callbacks=[early_stop], verbose=1)
y_pred_cnn = np.argmax(model_cnn.predict(X_test_pad), axis=1)
print("Accuracy:", accuracy_score(y_test, y_pred_cnn))
print(classification_report(y_test, y_pred_cnn, target_names=target_names, zero_division=0))

#  5. LSTM С АУГМЕНТАЦИЕЙ 
print("\n" + "="*60)
print("5. LSTM с аугментацией (ruwordnet)")
print("="*60)

train_counts = y_train.value_counts()
max_count = train_counts.max()
aug_texts = []
aug_labels = []
for cls in train_counts.index:
    count = train_counts[cls]
    texts_cls = X_train[y_train == cls].tolist()
    aug_texts.extend(texts_cls)
    aug_labels.extend([cls] * count)
    needed = max_count - count
    generated = 0
    while generated < needed:
        for t in texts_cls:
            if generated >= needed:
                break
            aug_texts.append(augment_text(t, shuffle_prob=0.4, delete_prob=0.3, synonym_prob=0.5))
            aug_labels.append(cls)
            generated += 1
    print(f'Класс {cls}: было {count}, добавлено {generated} (стало {count+generated})')

X_train_aug = pd.Series(aug_texts)
y_train_aug = pd.Series(aug_labels).astype(int)

tokenizer_lstm = Tokenizer(num_words=MAX_NB_WORDS, oov_token='<OOV>')
tokenizer_lstm.fit_on_texts(X_train_aug)
X_train_seq_lstm = tokenizer_lstm.texts_to_sequences(X_train_aug)
X_test_seq_lstm = tokenizer_lstm.texts_to_sequences(X_test)
X_train_pad_lstm = pad_sequences(X_train_seq_lstm, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
X_test_pad_lstm = pad_sequences(X_test_seq_lstm, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
vocab_size_lstm = min(MAX_NB_WORDS, len(tokenizer_lstm.word_index) + 1)

model_lstm = Sequential([
    Embedding(vocab_size_lstm, 128, input_length=MAX_SEQUENCE_LENGTH),
    Bidirectional(LSTM(64, return_sequences=True)),
    Bidirectional(LSTM(32)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])
model_lstm.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model_lstm.summary()

class_weights_lstm = compute_class_weight('balanced', classes=np.unique(y_train_aug), y=y_train_aug)
class_weight_dict_lstm = dict(enumerate(class_weights_lstm))
early_stop_lstm = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
history_lstm = model_lstm.fit(X_train_pad_lstm, y_train_aug, validation_split=0.2, epochs=20, batch_size=32,
                              class_weight=class_weight_dict_lstm, callbacks=[early_stop_lstm], verbose=1)
y_pred_lstm = np.argmax(model_lstm.predict(X_test_pad_lstm), axis=1)
print("Accuracy:", accuracy_score(y_test, y_pred_lstm))
print(classification_report(y_test, y_pred_lstm, target_names=target_names, zero_division=0))

#  6. XGBoost + SENTENCE EMBEDDINGS 
print("\n" + "="*60)
print("6. XGBoost + Sentence Embeddings + SMOTE")
print("="*60)

model_emb = SentenceTransformer('distiluse-base-multilingual-cased')
print("Генерация эмбеддингов для train...")
X_train_emb = model_emb.encode(X_train.tolist(), show_progress_bar=True)
print("Генерация эмбеддингов для test...")
X_test_emb = model_emb.encode(X_test.tolist(), show_progress_bar=True)

smote_xgb = get_smote(k_neighbors=5)
if smote_xgb is not None:
    X_train_res_xgb, y_train_res_xgb = smote_xgb.fit_resample(X_train_emb, y_train)
else:
    X_train_res_xgb, y_train_res_xgb = X_train_emb, y_train

xgb_model = xgb.XGBClassifier(objective='multi:softprob', eval_metric='mlogloss',
                              random_state=RANDOM_STATE, use_label_encoder=False)
param_grid_xgb = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
grid_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=3, scoring='f1_weighted', n_jobs=-1, verbose=1)
grid_xgb.fit(X_train_res_xgb, y_train_res_xgb)
print("Лучшие параметры XGBoost:", grid_xgb.best_params_)
y_pred_xgb = grid_xgb.predict(X_test_emb)
print("Accuracy:", accuracy_score(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb, target_names=target_names, zero_division=0))

#  ВИЗУАЛИЗАЦИЯ 
best_pred = y_pred_xgb
cm = confusion_matrix(y_test, best_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('Confusion Matrix - XGBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.show()

print("\n=== ВСЕ МОДЕЛИ ОБУЧЕНЫ И ОЦЕНЕНЫ ===")