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


w1, w2 = "необычайных", "университетов"

p1 = m.parse(w1)[0]
p2 = m.parse(w2)[0]

print(f"{w1}: лемма = {p1.normal_form}, тег = {p1.tag}")
print(f"{w2}: лемма = {p2.normal_form}, тег = {p2.tag}")
print("Согласованы:", is_agree(p1, p2))
print("Результат:", p1.normal_form, p2.normal_form)
