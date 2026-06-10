# -*- coding: utf-8 -*-
"""1. ВК ЗАГРУЗКА И ПАРСИНГ"""

!pip install vk_api

from vk_api import VkApi
from vk_api.exceptions import ApiError

# Ваш токен доступа
token = ''


# ID сообщества, стену которого вы хотите парсить
group_id = ''
#например,'natural_faces'

post_count = 100
offset = 0

# Создаем объект VKSession
vk_session = VkApi(token=token)

# Получаем объект VK_API
vk = vk_session.get_api()

# Структура для хранения данных всех постов
posts_data = []

try:
    # Цикл на 35 раз
    for i in range(35):
        # Сдвиг offset увеличивается на 100 с каждой итерацией
        offset = i * 100

        # Получение постов со стены сообщества
        wall_posts = vk.wall.get(count=post_count, domain=group_id, offset=offset)

        # Извлечение данных из каждого поста
        for post in wall_posts['items']:
            post_data = {'text': post['text']}
            posts_data.append(post_data)

    # Вывод данных всех собранных постов
    for post in posts_data:
        print(post)

except ApiError as e:
    print(f"Ошибка VK API: {e}")





"""2. ТЕКСТЫ КАЖДОГО СООБЩЕСТВА ИЛИ СУММАРНО ВСЕ ОЧИЩАЕМ ОТ ОРГАНЗАЦИОННЫХ ПОСТОВ И РЕКЛАМЫ"""

import pandas as pd
import re

# Загрузка исходного файла
df = pd.read_excel('/content/натуральные лица 3500.xlsx')

# Приведение имён колонок к стандартному виду
df.columns = [str(c).strip().lower() for c in df.columns]

# Ключевые слова и фразы для удаления рекламных и организационных текстов
stop_patterns = [
   r'требуются',  r'ищем',  r'ваканс',  r'команда',  r'оплат',  r'цена',  r'тариф',  r'инструкц', r'миф',
    r'курс',  r'вебинар', r'регистр', r'ответим выборочн', r'гарантированн.*ответ', r'администратор',

    r'проекте «психолог онлайн»',  r'психолог дарья киселёва', r'бесплатный онлайн-мини-курс', r'международный диплом',
    r'открыть частную практику',
    r'группа вконтакте',
    r'telegram-канал',
    r'site helphelp24.ru',
    r'подробности у администратора',
    r'пишите администратору',
    r'Что вы об этом думаете?',
    r'Четверг - рыбный день',
    r'🌿',
]
stop_re = re.compile('|'.join(stop_patterns), re.IGNORECASE)

def is_unwanted(text):
    if pd.isnull(text):
        return False
    return bool(stop_re.search(str(text)))

# --- НОВЫЙ КОД: Фильтрация по количеству слов ---
# Считаем количество слов в тексте (разделитель - пробел)
df['word_count'] = df['text'].astype(str).apply(lambda x: len(x.split()))

# Оставляем только строки, где количество слов >= 10
filtered_by_words = df[df['word_count'] >= 10].copy()

# --- ОСНОВНАЯ ФИЛЬТРАЦИЯ ПО СПИСКУ СТОП-СЛОВ ---
# Применяем функцию фильтрации к колонке 'text'
final_filtered = filtered_by_words[~filtered_by_words['text'].apply(is_unwanted)].copy()

# Сохранение результата в Excel
final_filtered.to_excel('Без_рекламы_и_орг.xlsx', index=False, sheet_name='Без рекламы и орг.')

print(f'Файл успешно сохранён: Без_рекламы_и_орг.xlsx')
print(f'Строк в исходном файле: {len(df)}')
print(f'Строк после фильтрации по словам: {len(filtered_by_words)}')
print(f'Строк после полной фильтрации: {len(final_filtered)}')





"""3. ПАРСИНГ ДЛЯ СООБЩЕСТВА "ПСИХОЛОГ ОНЛАЙН"
Посты сообщества "Психолог онлайн" строятся по форме: текст пользователя - ответ психолога. ПОэтому нужно вытащить текст пользователя отдельно, используем разделители, которые выделили в ходе анализа постов (как начинается ответ психолога).
"""

import pandas as pd
import re

def split_text(message: str) -> tuple:

    if not isinstance(message, str):
        return '', ''

    # Ищем вхождение слова "ответ" как отдельного слова (с пробелами или в начале/конце)
    match_otvet = re.search(r'\bответ\b', message)
    if match_otvet:
        split_pos = match_otvet.start()
        question = message[:split_pos].strip()
        advice = message[split_pos:].strip()
        return question, advice

    idx_1 = message.find("уважаемый автор")
    idx_2 = message.find("дорогой автор")

    if idx_1 != -1 or idx_2 != -1:
        if idx_1 != -1 and idx_2 != -1:
            split_pos = min(idx_1, idx_2)
        elif idx_1 != -1:
            split_pos = idx_1
        else:
            split_pos = idx_2

        question = message[:split_pos].strip()
        advice = message[split_pos:].strip()
        return question, advice

    tg_phrase = "в нашем телеграм-канале психолог онлайн - t.me/psonline24"
    tg_pos = message.find(tg_phrase)
    if tg_pos != -1:
        question = message[:tg_pos].strip()
        advice = message[tg_pos + len(tg_phrase):].strip()
        return question, advice

    greetings = ["здравствуйте", "добрый день", "доброго времени суток", "добрый вечер"]
    matches = []
    for greet in greetings:
        start = 0
        while True:
            pos = message.find(greet, start)
            if pos == -1:
                break
            end = pos + len(greet)
            matches.append((pos, end, greet))
            start = pos + 1

    # Проверяем условие: ровно одно приветствие и оно не в самом начале текста
    if len(matches) == 1:
        first_start, first_end, _ = matches[0]
        if first_start > 0:
            question = message[:first_start].strip()
            advice = message[first_end:].strip()
            return question, advice

    # Правило для нескольких приветствий
    if len(matches) >= 2:
        second_start, second_end, _ = matches[1]
        question = message[:second_start].strip()
        advice = message[second_end:].strip()
        return question, advice

    elif len(matches) == 1:
        first_start, first_end, _ = matches[0]
        underline = "___________________________"
        underline_pos = message.find(underline)
        if first_start > 0 and underline_pos != -1 and underline_pos < first_start:
            question = message[:first_start].strip()
            advice = message[first_end:].strip()
            return question, advice
        else:
            return message.strip(), ''
    else:
        # Проверяем наличие ссылки на VK Market
        vk_link_pattern = r'(https://vk\.com/market)|(https://vk\.com/market/product/)'
        link_match = re.search(vk_link_pattern, message)
        if link_match:
            # Если ссылка найдена — возвращаем пустой текст
            return '', ''  # Полностью отбрасываем сообщение
        else:
            return message.strip(), ''


# Обрабатываем наши тексты
input_file = "психолог онлайн 3500.xlsx"
output_file = "новые тексты.xlsx"
sheet_name = "text"

try:
    df = pd.read_excel(input_file, sheet_name=sheet_name)

    if 'text' not in df.columns:
        print(f"Ошибка: В файле нет колонки 'text'. Доступные колонки: {df.columns.tolist()}")
    else:
        # Приводим к нижнему регистру для корректного поиска
        df['text'] = df['text'].str.lower()

        # Применяем функцию парсинга
        df[['question', 'advice']] = df['text'].apply(lambda msg: pd.Series(split_text(msg)))

        # Считаем количество слов в исходном тексте (до парсинга)
        df['word_count'] = df['text'].apply(lambda x: len(str(x).split()))

        # Фильтруем строки, где в исходном тексте было меньше 10 слов
        df_filtered = df[df['word_count'] >= 10].copy()

        # Сохраняем результат без лишних колонок и индекса
        df_filtered.drop(columns=['text', 'word_count'], inplace=True)
        df_filtered.to_excel(output_file, index=False)

        print(f"✅ Обработка завершена. Файл сохранён как: {output_file}")
        print(f"Строк обработано всего: {len(df)}")
        print(f"Строк после фильтрации (>=10 слов): {len(df_filtered)}")

except FileNotFoundError:
    print(f"❌ Ошибка: Файл '{input_file}' не найден.")
except Exception as e:
    print(f"⚠️  Произошла ошибка при обработке файла: {e}")



""" 4. ИСКЛЮЧЕНИЕ ТЕКСТОВ БЕЗ "ЭМОЦИЙ"
#Среди всех текстов все еще остаются тексты без "эмоций". Будем их исключать, прогоняя лексику сообщения через словарь тональности КартаСловСент
"""

!python -m spacy download ru_core_news_sm

import csv
import re
import pandas as pd
import spacy

# Загружаем модель spaCy
nlp = spacy.load('ru_core_news_sm')
print('✅ spaCy загружен')

def load_kartaslov_lemmas(csv_path, nlp):
    """
    Загружает словарь КартаСловСент и лемматизирует все слова.
   Возвращает множество уникальных лемм.
    """
    lemmas = set()
    raw_words = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=';')
        next(reader, None)
        for row in reader:
            if row:
                word = row[0].strip().lower()
                if word and len(word) >= 2:
                    raw_words.append(word)

    print(f'Загружено слов из словаря: {len(raw_words)}')

    # Лемматизируем все слова
    for word in raw_words:
        doc = nlp(word)
        if doc and doc[0].lemma_:
            lemma = doc[0].lemma_
            if len(lemma) >= 2:
                lemmas.add(lemma)

    print(f'Уникальных лемм: {len(lemmas)}')
    return lemmas

# Загружаем леммы
lemmas = load_kartaslov_lemmas('kartaslov_sentiment.csv', nlp)

# Функция проверки текста на наличие эмоциональных лемм

def has_emotion_lemmas(text, lemmas, nlp):

    text_clean = str(text).strip()
    if not text_clean:
        return False

    # Обрабатываем текст через spaCy
    doc = nlp(text_clean.lower())

    # Проверяем каждое слово
    for token in doc:
        if token.is_alpha and token.lemma_ in lemmas:
            return True

    return False

def get_emotion_lemmas(text, lemmas, nlp):

    text_clean = str(text).strip()
    if not text_clean:
        return []

    doc = nlp(text_clean.lower())
    found_lemmas = []

    for token in doc:
        if token.is_alpha and token.lemma_ in lemmas:
            found_lemmas.append(token.lemma_)

    return list(set(found_lemmas))  # убираем дубликаты

# Обрабатываем наш общий файл

input_file = 'итоговые фильтр.xlsx'

try:
    df = pd.read_excel(input_file)
    print(f'✅ Загружен файл: {input_file}')
    print(f'   Строк: {len(df)}')
    print(f'   Столбцы: {list(df.columns)}')

    # Проверяем наличие столбца 'text'
    if 'text' not in df.columns:
        print('\n❌ ВНИМАНИЕ: Столбец "text" не найден!')
        print('   Доступные столбцы:', list(df.columns))
        # Пробуем найти похожий столбец
        text_columns = [col for col in df.columns if 'text' in col.lower() or 'текст' in col.lower()]
        if text_columns:
            print(f'   Найдены похожие столбцы: {text_columns}')
            text_column = text_columns[0]
            print(f'   Буду использовать столбец: {text_column}')
            df['text'] = df[text_column]  # создаем копию с именем 'text'
        else:
            print('   Подходящих столбцов не найдено. Проверьте файл.')

except FileNotFoundError:
    print(f'❌ Файл {input_file} не найден!')
    print('   Убедитесь, что файл находится в той же папке, что и ноутбук.')
    print('   Или укажите полный путь к файлу.')

    # Показываем текущую директорию
    import os
    print(f'\n   Текущая папка: {os.getcwd()}')
    print('   Доступные файлы:')
    for file in os.listdir():
        if file.endswith('.xlsx') or file.endswith('.xls'):
            print(f'      - {file}')

# Фильтрация текстов

if 'text' in df.columns:
    total = len(df)

    print('🔍 Анализ текстов на наличие эмоциональной лексики...')

    # Создаем маску для фильтрации
    mask = df['text'].apply(lambda x: has_emotion_lemmas(x, lemmas, nlp))

    # Разделяем на два датафрейма
    df_with_emotions = df[mask].copy()
    df_without_emotions = df[~mask].copy()

    # Добавляем колонку с найденными леммами
    df_with_emotions['found_lemmas'] = df_with_emotions['text'].apply(
        lambda x: get_emotion_lemmas(x, lemmas, nlp)
    )
    df_with_emotions['lemmas_count'] = df_with_emotions['found_lemmas'].apply(len)

    kept = len(df_with_emotions)
    removed = len(df_without_emotions)

    print(f'\n📊 РЕЗУЛЬТАТЫ ФИЛЬТРАЦИИ:')
    print(f'   📝 Всего текстов:          {total}')
    print(f'   ✅ С эмоциями:             {kept} ({kept/total*100:.1f}%)')
    print(f'   ❌ Без эмоций:             {removed} ({removed/total*100:.1f}%)')

    # Статистика по количеству найденных слов
    if kept > 0:
        avg_lemmas = df_with_emotions['lemmas_count'].mean()
        max_lemmas = df_with_emotions['lemmas_count'].max()
        print(f'\n📈 СТАТИСТИКА ПО ЭМОЦИОНАЛЬНЫМ СЛОВАМ:')
        print(f'   Среднее количество на текст: {avg_lemmas:.1f}')
        print(f'   Максимальное количество:     {max_lemmas}')

else:
    print('❌ Невозможно выполнить фильтрацию: столбец "text" не найден')

# Сохранение результатов

output_emotional = 'тексты_с_эмоциями.xlsx'
output_neutral = 'тексты_без_эмоций.xlsx'

if 'text' in df.columns:
    # Сохраняем текст с эмоциями
    # Убираем список лемм для сохранения (оставляем только количество)
    df_save = df_with_emotions.drop('found_lemmas', axis=1)
    df_save.to_excel(output_emotional, index=False)
    print(f'✅ Тексты с эмоциями сохранены: {output_emotional}')
    print(f'   Количество: {len(df_save)}')

    # Сохраняем тексты без эмоций
    df_without_emotions.to_excel(output_neutral, index=False)
    print(f'✅ Тексты без эмоций сохранены: {output_neutral}')
    print(f'   Количество: {len(df_without_emotions)}')

# Просмотр примеров
#  !Для наглядности, можно не запускать
if 'text' in df.columns:
    print('═' * 60)
    print('📋 ПРИМЕРЫ ТЕКСТОВ С ЭМОЦИЯМИ:')
    print('═' * 60)

    for i, (_, row) in enumerate(df_with_emotions.head(3).iterrows()):
        text = str(row['text'])[:150]
        lemmas_list = row.get('found_lemmas', [])
        count = row.get('lemmas_count', 0)

        print(f'\n{i+1}. [Найдено слов: {count}]')
        print(f'   Леммы: {", ".join(lemmas_list[:10])}')
        print(f'   Текст: {text}...' if len(str(row['text'])) > 150 else f'   Текст: {text}')

    if removed > 0:
        print('\n' + '═' * 60)
        print('📋 ПРИМЕРЫ ТЕКСТОВ БЕЗ ЭМОЦИЙ:')
        print('═' * 60)

        for i, (_, row) in enumerate(df_without_emotions.head(3).iterrows()):
            text = str(row['text'])[:150]
            print(f'\n{i+1}. {text}...' if len(str(row['text'])) > 150 else f'\n{i+1}. {text}')

# Самые частые эмоциональные леммы
#  !Для статистики, можно не запускать
if 'text' in df.columns and kept > 0:
    from collections import Counter

    # Собираем все найденные леммы
    all_lemmas = []
    for lemmas_list in df_with_emotions['found_lemmas']:
        all_lemmas.extend(lemmas_list)

    # Считаем частоты
    lemma_counts = Counter(all_lemmas)

    print('🔝 ТОП-20 САМЫХ ЧАСТЫХ ЭМОЦИОНАЛЬНЫХ СЛОВ:\n')
    print(f'{"№":<4} {"Лемма":<20} {"Количество":<12} {"% текстов"}')
    print('-' * 50)

    for i, (lemma, count) in enumerate(lemma_counts.most_common(20), 1):
        pct = count / kept * 100
        print(f'{i:<4} {lemma:<20} {count:<12} {pct:.1f}%')





""" 5. ВЫБИРАЕМ РАНДОМНЫЕ ТЕКСТЫ ДЛЯ РУЧНОЙ КАТЕГОРИЗАЦИИ
# ОТБИРАЕМ 500 текстов для ручной категоризации, отдельно выводим оставшиеся"""

import pandas as pd

# Настройки
excel_file = 'оставшиеся_тексты.xlsx'
sheet_name = 'Sheet1'
text_column = 'text'
sample_size = 1000
output_file = 'random_1000_texts.xlsx'

# Загрузка данных
df = pd.read_excel(excel_file, sheet_name=sheet_name)

# Проверка наличия столбца
if text_column not in df.columns:
    raise ValueError(f"В файле нет столбца '{text_column}'. Доступные столбцы: {list(df.columns)}")

# Перемешиваем и выбираем 500 текстов
sampled_df = df.sample(n=sample_size, random_state=42)[[text_column]]

# Сохраняем результат
sampled_df.to_excel(output_file, index=False)

print(f"Готово: выбрано {len(sampled_df)} текстов. Результат сохранён в '{output_file}'.")

import pandas as pd
import re

# НАСТРОЙКИ
BIG_FILE = 'итоговые фильтр.xlsx'          # Исходный файл со всеми текстами
SAMPLE_FILE = 'random_500_texts.xlsx'   # Файл с 500 случайными текстами
SHEET_NAME = 'Лист1'
TEXT_COLUMN = 'text'

OUTPUT_FILE = 'оставшиеся_тексты.xlsx'  # Куда сохранить результат

# Загрузка данных
try:
    big_df = pd.read_excel(BIG_FILE, sheet_name=SHEET_NAME)
    sample_df = pd.read_excel(SAMPLE_FILE)
except FileNotFoundError as e:
    print(f"Ошибка: не найден файл. Проверьте пути к {e.filename}")
    exit()

# Проверка наличия столбцов
for df, name in [(big_df, BIG_FILE), (sample_df, SAMPLE_FILE)]:
    if TEXT_COLUMN not in df.columns:
        print(f"В файле '{name}' нет столбца '{TEXT_COLUMN}'. Доступные: {list(df.columns)}")
        exit()

#Подготовка: функция для нормализации текста
def normalize(text):
    if pd.isna(text):
        return set()
    # Оставляем только буквы и цифры, разбиваем на слова
    words = re.findall(r'\w+', str(text).lower())
    return set(words)

# Создаём множество "отпечатков" для 500 выбранных текстов
sample_signatures = set()
for text in sample_df[TEXT_COLUMN]:
    sample_signatures.add(frozenset(normalize(text)))

# Фильтрация большого файла
# Оставляем только те строки, которых НЕТ среди выбранных 500
mask = big_df[TEXT_COLUMN].apply(lambda x: frozenset(normalize(x)) not in sample_signatures)
filtered_df = big_df[mask]

# Сохранение результата
filtered_df.to_excel(OUTPUT_FILE, index=False)

print(f"Готово.")
print(f"- Всего текстов в большом файле: {len(big_df)}")
print(f"Результат сохранён в '{OUTPUT_FILE}'")

"""Можно было сразу вытащить два файла: 500 + оставшиеся, но пришлось возвращаться к оставшимся уже после проведения ручной разметки"""