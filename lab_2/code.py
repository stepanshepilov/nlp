import os
import re

import gensim

model_path = "cbow.txt" if os.path.exists("cbow.txt") else "lab_2/cbow.txt"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)

pat = re.compile(r"(.*)_NOUN")

def get_top_nouns(pos, neg=None, topn=10):
    neg = neg or []
    pos_tags = [f"{w}_NOUN" if not w.endswith("_NOUN") else w for w in pos]
    neg_tags = [f"{w}_NOUN" if not w.endswith("_NOUN") else w for w in neg]
    
    dist = word2vec.most_similar(positive=pos_tags, negative=neg_tags, topn=topn * 5)
    
    nouns = []
    for item in dist:
        match = pat.match(item[0])
        if match:
            nouns.append((match.group(1), item[1]))
            if len(nouns) == topn:
                break
    return nouns

target1 = "безделушка"
target2 = "комод"

pos = ["шкаф", "несессер", "мебель"]
neg = []

results = get_top_nouns(pos, neg, topn=10)
words = [w for w, _ in results]

print("Топ-10 ближайших существительных:")
for rank, (word, score) in enumerate(results, 1):
    mark = "Выданное слово" if word in (target1, target2) else ""
    print(f"{rank:2d}. {word:15} ({score:.4f}){mark}")

in_top1 = target1 in words
in_top2 = target2 in words
print(f"\n{target1} в топ-10: {'Да' if in_top1 else 'Нет'}")
print(f"{target2} в топ-10: {'Да' if in_top2 else 'Нет'}")
