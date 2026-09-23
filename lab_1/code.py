import nltk
import pymorphy3
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

m = pymorphy3.MorphAnalyzer()

file_path = "sample_text.txt"
target_words = None

def is_agree(p1, p2):
    if p1.tag.number != p2.tag.number:
        return False
    
    if p1.tag.case != p2.tag.case:
        return False
    
    # Во множественном числе прилагательные не имеют категории рода
    if p1.tag.number != "plur" and p1.tag.gender != p2.tag.gender:
        return False
    
    return True

with open(file_path, "r", encoding="utf-8") as f:
    text = f.read()

allowed_pos = {"NOUN", "ADJF"}
pairs = []

for sent in sent_tokenize(text):
    words = [w.lower() for w in word_tokenize(sent) if w.isalpha()]
    for w1, w2 in zip(words[:-1], words[1:]):
        p1 = m.parse(w1)[0]
        p2 = m.parse(w2)[0]
        
        if p1.tag.POS in allowed_pos and p2.tag.POS in allowed_pos:
            if target_words and (p1.normal_form not in target_words and p2.normal_form not in target_words):
                continue
            
            if is_agree(p1, p2):
                pairs.append((p1.normal_form, p2.normal_form, f"{w1} {w2}"))

print(f"Найдено пар: {len(pairs)}\n")
for lemma1, lemma2, orig in pairs:
    print(f"{orig:30} -> {lemma1} {lemma2}")


def tag_summary(p):
    return (
        f"POS={p.tag.POS}, number={p.tag.number}, "
        f"case={p.tag.case}, gender={p.tag.gender}"
    )


def first_failure(p1, p2):
    if p1.tag.POS not in allowed_pos or p2.tag.POS not in allowed_pos:
        return "POS"
    
    if p1.tag.number != p2.tag.number:
        return "number"
    
    if p1.tag.case != p2.tag.case:
        return "case"
    
    if p1.tag.number != "plur" and p1.tag.gender != p2.tag.gender:
        return "gender"
    
    return "FOUND"


def agreeing_parse_pairs(w1, w2):
    pairs = []
    
    for p1 in m.parse(w1):
        for p2 in m.parse(w2):
            if p1.tag.POS not in allowed_pos or p2.tag.POS not in allowed_pos:
                continue
            
            if is_agree(p1, p2):
                pairs.append((p1, p2))
    
    return pairs

missed_candidates = {
    ("каждый", "студент"),
    ("глубокие", "знания"),
    ("солнечный", "день"),
    ("просторные", "поля"),
    ("удивительное", "полотно"),
    ("этот", "университет"),
}

print("Проверка пропущенных согласованных пар:\n")

for sent in sent_tokenize(text):
    words = [w.lower() for w in word_tokenize(sent) if w.isalpha()]
    
    for w1, w2 in zip(words[:-1], words[1:]):
        if (w1, w2) not in missed_candidates:
            continue
        
        p1 = m.parse(w1)[0]
        p2 = m.parse(w2)[0]
        alternatives = agreeing_parse_pairs(w1, w2)
        
        print(f"{w1} {w2}")
        print(f"  Причина отказа первого разбора: {first_failure(p1, p2)}")
        print(f"  Первый разбор: {tag_summary(p1)} | {tag_summary(p2)}")
        
        for p1_alt, p2_alt in alternatives:
            print(f"  Согласованный вариант: {tag_summary(p1_alt)} | {tag_summary(p2_alt)}")
        
        print()

w1, w2 = "холодные", "ветры"
adjective_noun_pairs = []

for p1 in m.parse(w1):
    for p2 in m.parse(w2):
        if p1.tag.POS == "ADJF" and p2.tag.POS == "NOUN" and is_agree(p1, p2):
            adjective_noun_pairs.append((p1, p2))

print("Проверка найденной пары с ошибочной частью речи:\n")
print(f"{w1} {w2}")
print(f"  Первый разбор: {tag_summary(m.parse(w1)[0])} | {tag_summary(m.parse(w2)[0])}")

for p1, p2 in adjective_noun_pairs:
    print(f"  Корректный POS-вариант: {tag_summary(p1)} | {tag_summary(p2)}")

print("\nКоличество пропущенных пар:", len(missed_candidates))


w1, w2 = "необычайных", "университетов"

p1 = m.parse(w1)[0]
p2 = m.parse(w2)[0]

print(f"{w1}: лемма = {p1.normal_form}, тег = {p1.tag}")
print(f"{w2}: лемма = {p2.normal_form}, тег = {p2.tag}")
print("Согласованы:", is_agree(p1, p2))
print("Результат:", p1.normal_form, p2.normal_form)

def tag_summary(p):
    return (
        f"POS={p.tag.POS}, number={p.tag.number}, "
        f"case={p.tag.case}, gender={p.tag.gender}"
    )


def first_failure(p1, p2):
    if p1.tag.POS not in allowed_pos or p2.tag.POS not in allowed_pos:
        return "POS"
    
    if p1.tag.number != p2.tag.number:
        return "number"
    
    if p1.tag.case != p2.tag.case:
        return "case"
    
    if p1.tag.number != "plur" and p1.tag.gender != p2.tag.gender:
        return "gender"
    
    return "FOUND"


def agreeing_parse_pairs(w1, w2):
    pairs = []
    
    for p1 in m.parse(w1):
        for p2 in m.parse(w2):
            if p1.tag.POS not in allowed_pos or p2.tag.POS not in allowed_pos:
                continue
            
            if is_agree(p1, p2):
                pairs.append((p1, p2))
    
    return pairs

missed_candidates = {
    ("каждый", "студент"),
    ("глубокие", "знания"),
    ("солнечный", "день"),
    ("просторные", "поля"),
    ("удивительное", "полотно"),
    ("этот", "университет"),
}

for sent in sent_tokenize(text):
    words = [w.lower() for w in word_tokenize(sent) if w.isalpha()]
    
    for w1, w2 in zip(words[:-1], words[1:]):
        if (w1, w2) not in missed_candidates:
            continue
        
        p1 = m.parse(w1)[0]
        p2 = m.parse(w2)[0]
        alternatives = agreeing_parse_pairs(w1, w2)
        
        print(f"{w1} {w2}")
        print(f"  Причина отказа первого разбора: {first_failure(p1, p2)}")
        print(f"  Первый разбор: {tag_summary(p1)} | {tag_summary(p2)}")
        
        for p1_alt, p2_alt in alternatives:
            print(f"  Согласованный вариант: {tag_summary(p1_alt)} | {tag_summary(p2_alt)}")
        
        print()

w1, w2 = "холодные", "ветры"
adjective_noun_pairs = []

for p1 in m.parse(w1):
    for p2 in m.parse(w2):
        if p1.tag.POS == "ADJF" and p2.tag.POS == "NOUN" and is_agree(p1, p2):
            adjective_noun_pairs.append((p1, p2))

print("Проверка найденной пары с ошибочной частью речи:\n")
print(f"{w1} {w2}")
print(f"  Первый разбор: {tag_summary(m.parse(w1)[0])} | {tag_summary(m.parse(w2)[0])}")

for p1, p2 in adjective_noun_pairs:
    print(f"  Корректный POS-вариант: {tag_summary(p1)} | {tag_summary(p2)}")
