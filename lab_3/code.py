import torch
from torch.nn import functional as F
from transformers import BertTokenizer, BertForMaskedLM

name = "bert-base-multilingual-uncased"
tokenizer = BertTokenizer.from_pretrained(name)
model = BertForMaskedLM.from_pretrained(name, return_dict=True)
model.eval()

def predict_mask(text, topn=10):
    if tokenizer.mask_token not in text:
        text = text.replace("___", tokenizer.mask_token)
        
    inputs = tokenizer.encode_plus(text, return_tensors="pt")
    mask_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)
    
    with torch.no_grad():
        output = model(**inputs)
        
    logits = output.logits
    softmax = F.softmax(logits, dim=-1)
    mask_word = softmax[0, mask_index[0], :]
    top = torch.topk(mask_word, topn)
    
    results = []
    for token, prob in zip(top[-1][0].data, top[0][0].data):
        results.append((tokenizer.decode([token]).strip(), prob.item()))
    return results

target1 = "выходит"
target2 = "играть"

# Для каждого слова подобран свой контекст
sentences = {
    target1: "Каждый день он [MASK] из дома.",
    target2: "Он выходит из дома, чтобы [MASK].",
}

results = {}
for target, sentence in sentences.items():
    predictions = predict_mask(sentence, topn=10)
    predicted_words = [word for word, _ in predictions]
    results[target] = target in predicted_words

    print(f"Целевое слово: {target}")
    print(f"Предложение: {sentence}")
    print("Топ-10 предсказаний модели:")
    for rank, (word, prob) in enumerate(predictions, 1):
        mark = " <--- ЦЕЛЕВОЕ СЛОВО" if word == target else ""
        print(f"{rank:2d}. {word:15} (вероятность: {prob:.4f}){mark}")
    print(f"Слово в топ-10: {'Да' if results[target] else 'Нет'}\n")

print("Результат:", "ОБА СЛОВА НАЙДЕНЫ В СВОИХ КОНТЕКСТАХ" if all(results.values()) else "ТРЕБУЕТСЯ КОРРЕКЦИЯ ОКРУЖЕНИЯ")
