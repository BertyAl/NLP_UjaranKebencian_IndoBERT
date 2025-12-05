import torch
import torch.nn.functional as F
import re
import string
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# KONFIGURASI MODEL
# ==========================================
AVAILABLE_MODELS = {
    "toxic_bert": {
        "name": "Dzeisonov/indobert-toxic-classifier",
        "desc": "IndoBERT (Base) - Toxic Classifier"
    },
    "toxic_roberta": {
        "name": "Dzeisonov/indoroberta-toxic-classifier",
        "desc": "IndoRoBERTa (Base) - Toxic Classifier"
    }
}

# Cache global
loaded_models = {}

def get_model_and_tokenizer(model_key):
    """Load model secara lazy loading."""
    if model_key not in AVAILABLE_MODELS:
        model_key = "toxic_bert"

    if model_key in loaded_models:
        return loaded_models[model_key]['tokenizer'], loaded_models[model_key]['model']

    config = AVAILABLE_MODELS[model_key]
    print(f"⏳ Sedang memuat model baru: {config['name']} ...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(config['name'])
        model = AutoModelForSequenceClassification.from_pretrained(config['name'])
        loaded_models[model_key] = {'tokenizer': tokenizer, 'model': model}
        print("✅ Model berhasil dimuat!")
        return tokenizer, model
    except Exception as e:
        print(f"❌ Gagal memuat model: {e}")
        return None, None

def preprocess_text(text):
    if not isinstance(text, str) or not text: return ""
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+|@\w+|#|\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_text(text, model_key):
    """Prediksi satu kalimat."""
    tokenizer, model = get_model_and_tokenizer(model_key)

    if not model or not tokenizer:
        return {"original_text": text, "label": "ERROR", "score": "0%"}

    clean_text = preprocess_text(text)
    if not clean_text:
        return {"original_text": text, "label": "Kosong", "score": "0%"}

    inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        label_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][label_id].item()

    predicted_label = model.config.id2label[label_id]
    
    # Standarisasi Label (Toxic / Non-Toxic)
    if predicted_label in ["LABEL_1", "Toxic", "toxic", "1"]:
        final_label = "Toxic"
    else:
        final_label = "Non-Toxic"

    return {
        "original_text": text,
        "text_clean": clean_text,
        "label": final_label,
        "score": f"{confidence:.1%}" # Mengembalikan persentase (misal: 98.5%)
    }

def process_file(file_obj, model_key):
    """Memproses file CSV, Excel, atau TXT."""
    results = []
    texts = []

    try:
        filename = file_obj.filename.lower()
        
        # 1. Jika file CSV
        if filename.endswith('.csv'):
            df = pd.read_csv(file_obj)
            texts = df.iloc[:, 0].astype(str).tolist() # Ambil kolom pertama
            
        # 2. Jika file Excel (.xlsx / .xls)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_obj)
            texts = df.iloc[:, 0].astype(str).tolist() # Ambil kolom pertama
            
        # 3. Jika file TXT (fallback)
        else:
            content = file_obj.read().decode("utf-8")
            texts = content.splitlines()

        # Batasi 50 baris agar server tidak hang
        for text in texts[:50]:
            if text.strip():
                res = predict_text(text, model_key)
                results.append(res)
                
    except Exception as e:
        print(f"Error processing file: {e}")
        return []

    return results