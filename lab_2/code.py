import os
import re

import gensim

model_path = "cbow.txt" if os.path.exists("cbow.txt") else "lab_2/cbow.txt"
word2vec = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=False)
pat = re.compile(r"(.*)_NOUN")


def noun(word):
    return word if word.endswith("_NOUN") else f"{word}_NOUN"


def get_top_nouns(pos, neg=None, topn=10):
    neg = neg or []
    pos_tags = [noun(w) for w in pos]
    neg_tags = [noun(w) for w in neg]
    dist = word2vec.most_similar(positive=pos_tags, negative=neg_tags, topn=topn * 5)

    return [
        (pat.match(word).group(1), score)
        for word, score in dist
        if pat.match(word)
    ][:topn]


def get_top_nouns_for_vector(vector, topn=10, exclude=()):
    dist = word2vec.similar_by_vector(vector, topn=topn * 5)
    excluded = {noun(word) for word in exclude}
    return [
        (pat.match(word).group(1), score)
        for word, score in dist
        if pat.match(word) and word not in excluded
    ][:topn]


def find_combination(target1, target2):
    target1, target2 = noun(target1), noun(target2)
    target1_name = pat.match(target1).group(1)
    target2_name = pat.match(target2).group(1)
    nearby = get_top_nouns([target1], topn=20)
    nearby += get_top_nouns([target2], topn=20)
    nearby += get_top_nouns([target1, target2], topn=20)
    target_names = {target1_name, target2_name}
    candidates = list(
        dict.fromkeys(
            word for word, _ in nearby if word not in target_names
        )
    )
    shared = [
        word
        for word, _ in get_top_nouns([target1, target2], topn=15)
        if word not in target_names
    ]
    best = None
    best_score = float("inf")

    def check(pos, neg, vector, exclude=()):
        nonlocal best, best_score
        results = get_top_nouns_for_vector(vector, exclude=exclude)
        words = [word for word, _ in results]
        if target1_name in words and target2_name in words:
            score = words.index(target1_name) + words.index(target2_name)
            if score < best_score:
                best = (pos, neg, results)
                best_score = score

    for word in candidates:
        vector = word2vec[noun(word)]
        check([word], [], vector, [word])

    for i, word1 in enumerate(shared[:15]):
        for word2 in shared[i + 1:]:
            vector1_candidate = word2vec[noun(word1)]
            vector2_candidate = word2vec[noun(word2)]
            check(
                [word1, word2],
                [],
                vector1_candidate + vector2_candidate,
                [word1, word2],
            )
            check(
                [word2],
                [word1],
                vector2_candidate - vector1_candidate,
                [word1, word2],
            )
            check(
                [word1],
                [word2],
                vector1_candidate - vector2_candidate,
                [word1, word2],
            )

    return best

target1 = "безделушка"
target2 = "комод"

pos = ["шкаф", "несессер", "мебель"]
neg = []

results = get_top_nouns(pos, neg, topn=10)
words = [w for w, _ in results]
used_search = False

if target1 not in words or target2 not in words:
    combination = find_combination(target1, target2)
    if combination:
        pos, neg, results = combination
        words = [w for w, _ in results]
        used_search = True

print(f"Целевая пара слов: {target1}, {target2}")
print(f"Сложение (pos): {pos}")
print(f"Вычитание (neg): {neg}")
if used_search:
    print("Комбинация подобрана автоматически.")
print("Топ-10 ближайших существительных:")
for rank, (word, score) in enumerate(results, 1):
    mark = " <--- ВЫДАННОЕ СЛОВО" if word in (target1, target2) else ""
    print(f"{rank:2d}. {word:15} ({score:.4f}){mark}")

in_top1 = target1 in words
in_top2 = target2 in words
print(f"\n{target1} в топ-10: {'Да' if in_top1 else 'Нет'}")
print(f"{target2} в топ-10: {'Да' if in_top2 else 'Нет'}")
if not in_top1 or not in_top2:
    print("Не удалось подобрать комбинацию среди проверенных вариантов.")

demo_target1 = "безделушка"
demo_target2 = "комод"
combination = find_combination(demo_target1, demo_target2)

if combination is None:
    print("Не удалось подобрать комбинацию среди проверенных вариантов.")
else:
    demo_pos, demo_neg, demo_results = combination
    formula = " + ".join(demo_pos)
    if demo_neg:
        formula += (" - " if formula else "") + " - ".join(demo_neg)

    print(f"Целевая пара: {demo_target1}, {demo_target2}")
    print(f"Найденная комбинация: {formula}")
    print("Топ-10 ближайших существительных:")
    for rank, (word, score) in enumerate(demo_results, 1):
        mark = " <--- ЦЕЛЬ" if word in (demo_target1, demo_target2) else ""
        print(f"{rank:2d}. {word:15} ({score:.4f}){mark}")
