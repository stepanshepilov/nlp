import torch
from torch.nn import functional as F
from transformers import BertTokenizer, BertForMaskedLM

name = "bert-base-multilingual-uncased"
tokenizer = BertTokenizer.from_pretrained(name)
model = BertForMaskedLM.from_pretrained(name, return_dict=True)
model.eval()

def predict_mask(text, topn=10):
    text = text.replace("___", tokenizer.mask_token)
    inputs = tokenizer.encode_plus(text, return_tensors="pt")
    mask_positions = torch.where(
        inputs["input_ids"][0] == tokenizer.mask_token_id
    )[0]

    if len(mask_positions) == 0:
        raise ValueError("В предложении должен быть хотя бы один маркер [MASK].")

    with torch.no_grad():
        output = model(**inputs)

    probabilities = F.softmax(output.logits, dim=-1)
    predictions_by_position = []

    for mask_position in mask_positions:
        mask_probabilities = probabilities[0, mask_position, :]
        top = torch.topk(mask_probabilities, topn)
        predictions = []
        for token, prob in zip(top.indices, top.values):
            predictions.append((tokenizer.decode([token.item()]).strip(), prob.item()))
        predictions_by_position.append(predictions)

    return predictions_by_position


target_words = ["выходит", "играть"]
sentence = "Каждый день он [MASK] из дома, чтобы [MASK]."
predictions_by_position = predict_mask(sentence, topn=10)

if len(predictions_by_position) != len(target_words):
    raise ValueError("Число масок должно совпадать с числом целевых слов.")

results = []
print(f"Единое предложение: {sentence}\n")

for position, (target, predictions) in enumerate(
    zip(target_words, predictions_by_position), start=1
):
    predicted_words = [word for word, _ in predictions]
    found = target in predicted_words
    results.append(found)

    print(f"Маска {position}; целевое слово: {target}")
    for rank, (word, probability) in enumerate(predictions, 1):
        mark = " <--- ЦЕЛЕВОЕ СЛОВО" if word == target else ""
        print(f"{rank:2d}. {word:15} (вероятность: {probability:.4f}){mark}")
    print(f"Целевое слово в топ-10: {'Да' if found else 'Нет'}\n")

print(
    "Результат:",
    "ОБА СЛОВА В ТОП-10 ОДНОГО ПРЕДЛОЖЕНИЯ"
    if all(results)
    else "НЕ ВСЕ ЦЕЛЕВЫЕ СЛОВА ПОПАЛИ В ТОП-10"
)
