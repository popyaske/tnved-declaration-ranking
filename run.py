import json
import os
import re
import argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Стоп-слова из nltk
STOPWORDS = {'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она', 'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о', 'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'вдруг', 'ли', 'если', 'уже', 'или', 'ни', 'быть', 'был', 'него', 'до', 'вас', 'нибудь', 'опять', 'уж', 'вам', 'ведь', 'там', 'потом', 'себя', 'ничего', 'ей', 'может', 'они', 'тут', 'где', 'есть', 'надо', 'ней', 'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто', 'чего', 'раз', 'тоже', 'себе', 'под', 'будет', 'ж', 'тогда', 'кто', 'этот', 'того', 'потому', 'этого', 'какой', 'совсем', 'ним', 'здесь', 'этом', 'один', 'почти', 'мой', 'тем', 'чтобы', 'нее', 'сейчас', 'были', 'куда', 'зачем', 'всех', 'никогда', 'можно', 'при', 'наконец', 'два', 'об', 'другой', 'хоть', 'после', 'над', 'больше', 'тот', 'через', 'эти', 'нас', 'про', 'всего', 'них', 'какая', 'много', 'разве', 'три', 'эту', 'моя', 'впрочем', 'хорошо', 'свою', 'этой', 'перед', 'иногда', 'лучше', 'чуть', 'том', 'нельзя', 'такой', 'им', 'более', 'всегда', 'конечно', 'всю', 'между'}
                                                  
# ------------------- Загрузка данных -------------------
def load_declarations(data_dir):
    decls = []
    with open(os.path.join(data_dir, 'declarations.jsonl'), 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            parts = []
            for field in ['G31_1', 'desc_extention']:
                val = item.get(field, '')
                if val:
                    parts.append(str(val))
            country = item.get('G34', '')
            if country:
                parts.append(f"страна {country}")
            amount = item.get('G42', '')
            if amount:
                parts.append(f"сумма {amount}")
            text = ' '.join(parts).strip()
            decls.append({'id': item['declaration_id'], 'text': text})
    return decls

def load_regulations(data_dir):
    regs = []
    with open(os.path.join(data_dir, 'regulations.jsonl'), 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            desc = item.get('description', '') or ''
            notes = item.get('notes', '') or ''
            expl = item.get('explanation', '') or ''
            code = item.get('code', '') or ''
            text = (desc + ' ' + notes + ' ' + expl).strip()
            regs.append({
                'id': item['regulation_id'],
                'code': code,
                'text': text
            })
    return regs

# ------------------- Парсинг tnved_knowledge.txt -------------------
def parse_tnved_knowledge(knowledge_path):
    """
    Извлекает из файла структуру: для каждого кода (первые 2-4 цифры) сохраняет
    заголовок группы/позиции. Используется для обогащения регуляций.
    """
    group_info = {}
    with open(knowledge_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or not re.match(r'^\d', line):
                continue
            # Ищем код и описание (формат: "XXXX | описание" или "XXXX - описание")
            m = re.match(r'^(\d{2,4})\s*[|\-]\s*(.+)', line)
            if m:
                code_prefix = m.group(1)
                desc = m.group(2).strip()
                group_info[code_prefix] = desc
    return group_info

def enrich_regulations(regs, group_info):
    """Добавляем к тексту регуляции описание её группы (первые 2 цифры кода)."""
    for reg in regs:
        code = reg['code']
        if code and len(code) >= 2:
            prefix2 = code[:2]
            prefix4 = code[:4] if len(code) >= 4 else prefix2
            # Сначала ищем по 4, потом по 2
            group_desc = group_info.get(prefix4) or group_info.get(prefix2) or ''
            if group_desc:
                reg['text'] = reg['text'] + ' ' + group_desc
    return regs

# ------------------- Гибридный поиск -------------------
def tokenize(text):
    # Простая токенизация с удалением стоп-слов и пунктуации
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return [t for t in text.split() if t not in STOPWORDS and len(t) > 1]

def compute_hybrid_scores(query_text, reg_texts, bm25_model, embeddings, model, weight_bm25=0.3, weight_sem=0.7):
    """
    Вычисляет комбинированный скор: BM25 + косинусное сходство эмбеддингов.
    """
    # BM25
    query_tokens = tokenize(query_text)
    bm25_scores = bm25_model.get_scores(query_tokens)  # сырые скоры
    # Нормализация BM25 (min-max)
    if bm25_scores.max() > bm25_scores.min():
        bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())
    else:
        bm25_scores = np.zeros_like(bm25_scores)
    
    # Эмбеддинги
    if query_text.strip():
        query_emb = model.encode([query_text], normalize_embeddings=True)[0]
        sem_scores = np.dot(embeddings, query_emb)  # косинусное (нормализовано)
    else:
        sem_scores = np.zeros(len(reg_texts))
    
    # Комбинация
    combined = weight_bm25 * bm25_scores + weight_sem * sem_scores
    return combined

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    
    data_dir = args.data
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    
    # Загрузка
    decls = load_declarations(data_dir)
    regs = load_regulations(data_dir)
    
    # Парсим tnved_knowledge
    knowledge_path = os.path.join(data_dir, 'tnved_knowledge.txt')
    if os.path.exists(knowledge_path):
        group_info = parse_tnved_knowledge(knowledge_path)
        regs = enrich_regulations(regs, group_info)
    
    # Подготовка текстов для BM25 и эмбеддингов
    reg_texts = [r['text'] for r in regs]
    tokenized_corpus = [tokenize(t) for t in reg_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Модель эмбеддингов
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    model.eval()
    embeddings = model.encode(reg_texts, normalize_embeddings=True, show_progress_bar=True)
    
    # Сбор результатов
    out_rows = []
    for decl in decls:
        query = decl['text']
        # Если текст пустой, используем заглушку "unknown"
        if not query:
            query = "unknown"
        scores = compute_hybrid_scores(query, reg_texts, bm25, embeddings, model)
        top_indices = np.argsort(scores)[::-1][:10]
        for rank, idx in enumerate(top_indices, start=1):
            out_rows.append([
                decl['id'],
                rank,
                regs[idx]['id'],
                float(scores[idx])
            ])
    
    # Сохранение
    df = pd.DataFrame(out_rows, columns=['declaration_id', 'rank', 'regulation_id', 'score'])
    df.to_csv(os.path.join(out_dir, 'predictions.csv'), index=False, float_format='%.8f')
    print(f"Результат сохранён в {out_dir}/predictions.csv")

if __name__ == '__main__':
    main()