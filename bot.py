import vk_api
from vk_api.bot_longpoll import VkBotLongPoll, VkBotEventType
from vk_api.keyboard import VkKeyboard, VkKeyboardColor
from vk_api.utils import get_random_id
import sqlite3
import json
import pickle
import re
import random
import pymorphy3
from collections import Counter
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

#  ПОДКЛЮЧЕНИЕ RUWORDNET ДЛЯ СИНОНИМОВ (опционально) 
USE_SYNONYM_EXPANSION = False  # По умолчанию отключено для производительности
ruwordnet_available = False
try:
    if USE_SYNONYM_EXPANSION:
        from ruwordnet import RuWordNet
        wordnet = RuWordNet()
        ruwordnet_available = True
        print("RuWordNet загружен")
except Exception as e:
    print(f"RuWordNet не загружен: {e}")
    ruwordnet_available = False

def expand_with_synonyms(lemmas, max_synonyms=3):
    """Расширяет список лемм синонимами через RuWordNet"""
    if not ruwordnet_available:
        return lemmas
    expanded = set(lemmas)
    for lemma in lemmas:
        try:
            synsets = wordnet.get_synsets(lemma)
            for synset in synsets:
                for sense in synset.senses:
                    syn = sense.lemma.lower()
                    if syn != lemma and len(syn) > 2:
                        expanded.add(syn)
                        if len(expanded) - len(lemmas) >= max_synonyms:
                            break
                if len(expanded) - len(lemmas) >= max_synonyms:
                    break
        except:
            continue
    return list(expanded)

#  ФУНКЦИЯ ДЛЯ КОРРЕКТНОЙ ЗАГРУЗКИ METHODS.JSON 
def load_methods_safe(filepath='methods.json'):
    with open(filepath, 'r', encoding='utf-8') as f:
        raw = f.read()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"Первый парсинг не удался: {e}")
        print("Пытаюсь автоматически исправить неэкранированные переносы строк...")
        fixed = []
        in_string = False
        escape = False
        for ch in raw:
            if escape:
                fixed.append(ch)
                escape = False
                continue
            if ch == '\\':
                escape = True
                fixed.append(ch)
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                fixed.append(ch)
                continue
            if in_string and ch == '\n':
                fixed.append('\\n')
                continue
            if in_string and ch == '\r':
                fixed.append('\\r')
                continue
            if in_string and ch == '\t':
                fixed.append('\\t')
                continue
            fixed.append(ch)
        fixed_str = ''.join(fixed)
        try:
            return json.loads(fixed_str)
        except json.JSONDecodeError as e2:
            print(f"Исправление не помогло: {e2}")
            raise

#  ЗАГРУЗКА ЭМОТИВНОГО ЛЕКСИКОНА 
def load_emotive_lexicon(filepath='EMOTIVE_LEXICON.csv'):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        if content.startswith('EMOTIVE_LEXICON ='):
            dict_str = content[len('EMOTIVE_LEXICON ='):].strip()
            lexicon = eval(dict_str)
        else:
            lexicon = eval(content)
        return lexicon
    except Exception as e:
        print(f"Ошибка загрузки EMOTIVE_LEXICON: {e}")
        return {}

EMOTIVE_LEXICON = load_emotive_lexicon()
lexicon_available = bool(EMOTIVE_LEXICON)

#  ЗАГРУЗКА МЕТОДИК 
try:
    METHODS_LIST = load_methods_safe('methods.json')
    methods_available = True
except Exception as e:
    print(f"Ошибка загрузки methods.json: {e}")
    METHODS_LIST = []
    methods_available = False

# Индекс для быстрого поиска (кроме категорий 1 и 3, для них особая логика)
methods_index = {}
special_methods = {}
if methods_available:
    for method in METHODS_LIST:
        raw_cat = method.get('category_raw', '')
        if not raw_cat:
            continue
        if raw_cat in ('1', '3'):
            special_methods[raw_cat] = method
            continue
        intensity_str = method.get('intensity', '')
        intensities = []
        if intensity_str:
            parts = [p.strip().lower() for p in intensity_str.split(',')]
            intensities = [p for p in parts if p in ('низкая', 'средняя', 'высокая')]
        else:
            intensities = ['низкая', 'средняя', 'высокая']
        for intensity in intensities:
            methods_index.setdefault(raw_cat, {}).setdefault(intensity, []).append(method)

#  МОРФОЛОГИЯ 
morph = pymorphy3.MorphAnalyzer()

#  СТОП-СЛОВА 
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

#  ПРЕДОБРАБОТКА 
def preprocess_text(text, use_synonyms=False):
    text = text.lower()
    text = re.sub(r'[^а-яё\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    lemmas = []
    for word in words:
        if word not in stop_words and len(word) > 1:
            lemma = morph.parse(word)[0].normal_form
            lemmas.append(lemma)
    if use_synonyms and ruwordnet_available:
        lemmas = expand_with_synonyms(lemmas)
    return ' '.join(lemmas)

#  ЭМОТИВНАЯ НАСЫЩЕННОСТЬ 
def is_emotive_lemma(lemma, lexicon_set):
    for base in lexicon_set:
        if base in lemma:
            return True
    return False

def count_emotive_words(text, category):
    if not lexicon_available:
        return 0
    words = text.lower().split()
    count = 0
    lexicon_set = EMOTIVE_LEXICON.get(category, set())
    if not lexicon_set:
        return 0
    for w in words:
        parsed = morph.parse(w)
        if parsed:
            lemma = parsed[0].normal_form
            if is_emotive_lemma(lemma, lexicon_set):
                count += 1
    return count

def compute_intensity(text, category):
    total_words = len(text.split())
    if total_words == 0:
        return 'низкая'
    emotive_cnt = count_emotive_words(text, category)
    density = emotive_cnt / total_words
    if density < 0.2:
        return 'низкая'
    elif density < 0.5:
        return 'средняя'
    else:
        return 'высокая'

#  ЗАГРУЗКА МОДЕЛИ 
model_available = False
tfidf_available = False
model = None
tfidf = None
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    model_available = True
except Exception as e:
    print(f"Ошибка загрузки model.pkl: {e}")

try:
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    tfidf_available = True
except Exception as e:
    print(f"Ошибка загрузки tfidf.pkl: {e}")

#  ЗАГРУЗКА МАППИНГА КАТЕГОРИЙ 
CAT_MAPPING = {}
try:
    with open('cat_mapping.json', 'r', encoding='utf-8') as f:
        CAT_MAPPING = json.load(f)
    CAT_MAPPING = {int(k): v for k, v in CAT_MAPPING.items()}
except Exception as e:
    print(f"Ошибка загрузки cat_mapping.json: {e}")

# Преобразование имени категории -> raw (для сопоставления с методиками)
CATEGORY_NAME_TO_RAW = {
    'Бодрость': '1',
    'Уныние': '2',
    'Спокойствие': '3',
    'Тревога': '4',
    'Негативный образ себя/жизни': '5'
}

def classify_mood(text):
    """Возвращает словарь с категорией и scores. При отсутствии модели - спокойствие по умолчанию."""
    if not model_available or not tfidf_available or not CAT_MAPPING:
        return {
            'raw_category': '3',
            'category_name': 'Спокойствие',
            'scores': {}
        }
    cleaned = preprocess_text(text)
    if not cleaned.strip():
        return {
            'raw_category': '3',
            'category_name': 'Спокойствие',
            'scores': {}
        }
    vec = tfidf.transform([cleaned])
    proba = model.predict_proba(vec)[0]
    best_idx = proba.argmax()
    category_name = CAT_MAPPING.get(best_idx, 'Спокойствие')
    raw_category = CATEGORY_NAME_TO_RAW.get(category_name, '3')
    scores = {CAT_MAPPING.get(i, f'idx{i}'): proba[i] for i in range(len(proba))}
    return {'raw_category': raw_category, 'category_name': category_name, 'scores': scores}

#  ПОДБОР МЕТОДИКИ 
def get_method_for_category(raw_category, intensity):
    if not methods_available:
        return None
    # Для категорий 1 и 3 используем единственную методику
    if raw_category in ('1', '3'):
        return special_methods.get(raw_category)
    if raw_category not in methods_index:
        return None
    if intensity in methods_index[raw_category]:
        methods = methods_index[raw_category][intensity]
        if methods:
            return random.choice(methods)
    # fallback: любые методы этой категории
    all_methods = []
    for int_level, methods_list in methods_index[raw_category].items():
        all_methods.extend(methods_list)
    if all_methods:
        return random.choice(all_methods)
    return None

def format_method_text(method):
    lines = []
    if method.get('type'):
        lines.append(f"📌 Тип: {method['type']}")
    if method.get('method_and_instructions'):
        lines.append(f"📖 Методика:\n{method['method_and_instructions']}")
    if method.get('link'):
        lines.append(f"🔗 Ссылка: {method['link']}")
    return '\n\n'.join(lines)

def get_recommendation(user_id, text, raw_category, intensity, entry_id):
    method = get_method_for_category(raw_category, intensity)
    if not method:
        rec_text = ("К сожалению, подходящая методика не найдена.\n"
                    "Рекомендуем обратиться к специалисту или опишите свои ощущения подробнее.")
    else:
        rec_text = format_method_text(method)
    save_recommendation(user_id, entry_id, rec_text)
    return rec_text

#  АНАЛИЗ ТРЕНДА 
def analyze_trend(user_db_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute('''SELECT mood_category FROM entries 
                   WHERE user_id = ? ORDER BY timestamp DESC LIMIT 7''', (user_db_id,))
    rows = cur.fetchall()
    conn.close()
    if len(rows) < 3:
        return None
    cats = [r[0] for r in rows]
    # Порядок: 1=бодрость (лучше), 5=негативный образ (хуже)
    mood_order = {'Бодрость': 1, 'Уныние': 2, 'Спокойствие': 3, 'Тревога': 4, 'Негативный образ себя/жизни': 5}
    scores = [mood_order.get(c, 3) for c in cats]
    # Проверка устойчивого ухудшения: последние 3 значения не меньше предыдущих
    if len(scores) >= 4 and scores[0] >= scores[1] >= scores[2] >= scores[3]:
        return ("📉 За последние дни наблюдается устойчивое ухудшение настроения.\n"
                "Рекомендуем пройти шкалу депрессии Бека (BDI) или шкалу тревоги STAI.\n"
                "Горячая линия МЧС: 8-800-775-17-17")
    return None

#  БАЗА ДАННЫХ 
DB_NAME = 'bot_mood.db'

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            vk_id INTEGER UNIQUE,
            first_name TEXT,
            last_name TEXT,
            state TEXT DEFAULT 'start'
        );
        CREATE TABLE IF NOT EXISTS entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            text TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            mood_category TEXT,
            model_output TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        CREATE TABLE IF NOT EXISTS recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            entry_id INTEGER,
            recommendation_text TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id),
            FOREIGN KEY(entry_id) REFERENCES entries(id)
        );
    ''')
    conn.commit()
    conn.close()

def get_or_create_user(vk_id, first_name='', last_name=''):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute('SELECT id, state FROM users WHERE vk_id = ?', (vk_id,))
    user = cur.fetchone()
    if user is None:
        cur.execute('INSERT INTO users (vk_id, first_name, last_name) VALUES (?, ?, ?)',
                    (vk_id, first_name, last_name))
        conn.commit()
        user_id = cur.lastrowid
        state = 'start'
    else:
        user_id, state = user
    conn.close()
    return user_id, state

def update_user_state(user_id, state):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute('UPDATE users SET state = ? WHERE id = ?', (state, user_id))
    conn.commit()
    conn.close()

def save_entry(user_id, text, mood_category, model_output):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute('INSERT INTO entries (user_id, text, mood_category, model_output) VALUES (?, ?, ?, ?)',
                (user_id, text, mood_category, json.dumps(model_output, ensure_ascii=False)))
    entry_id = cur.lastrowid
    conn.commit()
    conn.close()
    return entry_id

def save_recommendation(user_id, entry_id, rec_text):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute('INSERT INTO recommendations (user_id, entry_id, recommendation_text) VALUES (?, ?, ?)',
                (user_id, entry_id, rec_text))
    conn.commit()
    conn.close()

def get_entry_count(user_id):
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()
    cur.execute('SELECT COUNT(*) FROM entries WHERE user_id = ?', (user_id,))
    count = cur.fetchone()[0]
    conn.close()
    return count

#  КЛАВИАТУРА 
def create_keyboard(buttons_text, one_time=False):
    keyboard = VkKeyboard(one_time=one_time)
    for i, text in enumerate(buttons_text):
        if i > 0 and i % 2 == 0:
            keyboard.add_line()
        keyboard.add_button(text, color=VkKeyboardColor.PRIMARY)
    return keyboard.get_keyboard()

#  VK БОТ 
VK_TOKEN = ''                               # Заполнить токен
GROUP_ID =                                  # Заполнить ID группы

vk_session = vk_api.VkApi(token=VK_TOKEN)
vk = vk_session.get_api()
longpoll = VkBotLongPoll(vk_session, group_id=GROUP_ID)
user_state = {}

def send_message(peer_id, message, keyboard=None):
    vk.messages.send(
        peer_id=peer_id,
        message=message,
        random_id=get_random_id(),
        keyboard=keyboard
    )

def process_new_message(event):
    msg = event.obj.message
    user_vk_id = msg['from_id']
    text = msg.get('text', '').strip()
    peer_id = msg['peer_id']

    # Получаем данные пользователя
    user_info = vk.users.get(user_ids=user_vk_id, fields=['first_name', 'last_name'])
    first_name = user_info[0].get('first_name', '') if user_info else ''
    last_name = user_info[0].get('last_name', '') if user_info else ''

    user_db_id, db_state = get_or_create_user(user_vk_id, first_name, last_name)
    current_state = user_state.get(user_vk_id, db_state)

    # Состояние ожидания текста
    if current_state in ('start', 'ready_for_text'):
        # Валидация: пустой текст или слишком короткий
        if not text:
            send_message(peer_id, "Пожалуйста, напишите текст вашего состояния.")
            # Остаёмся в том же состоянии
            user_state[user_vk_id] = 'ready_for_text'
            update_user_state(user_db_id, 'ready_for_text')
            return
        if len(text.split()) < 3:
            send_message(peer_id, "Опишите своё состояние подробнее (минимум 3 слова).")
            user_state[user_vk_id] = 'ready_for_text'
            update_user_state(user_db_id, 'ready_for_text')
            return

        mood = classify_mood(text)
        raw_cat = mood['raw_category']
        cat_name = mood['category_name']
        intensity = compute_intensity(text, cat_name)
        entry_id = save_entry(user_db_id, text, cat_name, mood['scores'])
        rec_text = get_recommendation(user_db_id, text, raw_cat, intensity, entry_id)

        keyboard = create_keyboard(["Ещё методика", "Новая запись", "Статистика", "Завершить"])
        send_message(peer_id, rec_text, keyboard=keyboard)

        user_state[user_vk_id] = 'choosing'
        update_user_state(user_db_id, 'choosing')

        # Анализ тренда (если есть достаточно записей)
        if get_entry_count(user_db_id) >= 3:
            trend_msg = analyze_trend(user_db_id)
            if trend_msg:
                send_message(peer_id, trend_msg)

    # Состояние выбора действия после получения методики
    elif current_state == 'choosing':
        if text == "Ещё методика":
            conn = sqlite3.connect(DB_NAME)
            cur = conn.cursor()
            cur.execute('SELECT text, mood_category, id FROM entries WHERE user_id = ? ORDER BY timestamp DESC LIMIT 1',
                        (user_db_id,))
            last = cur.fetchone()
            conn.close()
            if last:
                last_text, last_cat, last_entry_id = last
                last_mood = classify_mood(last_text)
                last_raw_cat = last_mood['raw_category']
                last_intensity = compute_intensity(last_text, last_cat)
                new_rec = get_recommendation(user_db_id, last_text, last_raw_cat, last_intensity, last_entry_id)
                send_message(peer_id, f"✨ Дополнительная методика:\n{new_rec}")
            else:
                send_message(peer_id, "Сначала отправьте текст вашего состояния.")
                user_state[user_vk_id] = 'ready_for_text'
                update_user_state(user_db_id, 'ready_for_text')
        elif text == "Новая запись":
            send_message(peer_id, "Опишите, что вы чувствуете сейчас.")
            user_state[user_vk_id] = 'ready_for_text'
            update_user_state(user_db_id, 'ready_for_text')
        elif text == "Статистика":
            conn = sqlite3.connect(DB_NAME)
            cur = conn.cursor()
            cur.execute('SELECT mood_category FROM entries WHERE user_id = ? ORDER BY timestamp DESC LIMIT 5',
                        (user_db_id,))
            rows = cur.fetchall()
            conn.close()
            if rows:
                cats = [r[0] for r in rows]
                common = Counter(cats).most_common(1)[0][0]
                send_message(peer_id, f"Последние состояния: {', '.join(cats)}\nПреобладающее: {common}")
            else:
                send_message(peer_id, "Нет сохранённых записей.")
        elif text == "Завершить":
            send_message(peer_id, "Сеанс завершён. Если захотите продолжить, просто напишите мне.")
            user_state[user_vk_id] = 'start'
            update_user_state(user_db_id, 'start')
        else:
            send_message(peer_id, "Выберите действие с помощью кнопок.")
    else:
        send_message(peer_id, "Начнём сначала. Опишите своё состояние.")
        user_state[user_vk_id] = 'ready_for_text'
        update_user_state(user_db_id, 'ready_for_text')

def main():
    init_db()
    print("Бот запущен.")
    # Проверка доступности компонентов при старте
    if not model_available or not tfidf_available:
        print("ВНИМАНИЕ: Модель или TF-IDF не загружены. Классификация будет работать в упрощённом режиме.")
    if not methods_available:
        print("ВНИМАНИЕ: methods.json не загружен. Рекомендации будут недоступны.")
    if not lexicon_available:
        print("ВНИМАНИЕ: EMOTIVE_LEXICON не загружен. Интенсивность будет всегда 'низкая'.")
    try:
        for event in longpoll.listen():
            if event.type == VkBotEventType.MESSAGE_NEW:
                process_new_message(event)
    except KeyboardInterrupt:
        print("Бот остановлен.")
    finally:
        for vk_id, state in user_state.items():
            user_db_id, _ = get_or_create_user(vk_id)
            update_user_state(user_db_id, state)

if __name__ == '__main__':
    main()
