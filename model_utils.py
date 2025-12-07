import torch
import torch.nn.functional as F
import re
import string
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

AVAILABLE_MODELS = {
    "toxic_bert": {
        "name": "Dzeisonov/indobert-toxic-classifier",
        "desc": "IndoBERT (Fine-tuned)"
    },
    "toxic_roberta": {
        "name": "Dzeisonov/indoroberta-toxic-classifier",
        # FIX: Gunakan tokenizer dari base model aslinya karena yang di repo Dzeisonov rusak
        "tokenizer_name": "flax-community/indonesian-roberta-base", 
        "desc": "IndoRoBERTa (Fine-tuned)"
    },
    "toxic_bertweet": {
        "name": "Exqrch/IndoBERTweet-HateSpeech",        
        "tokenizer_name": "indolem/indobertweet-base-uncased", 
        "desc": "IndoBERTweet (Baseline Model)"
    }
}

# Cache global untuk menyimpan model yang sudah di-load
loaded_models = {}

def get_model_and_tokenizer(model_key):
    """
    Load model dan tokenizer secara lazy loading.
    Otomatis mendeteksi apakah perlu path tokenizer khusus atau tidak.
    """
    # Default ke toxic_bert jika key tidak ditemukan
    if model_key not in AVAILABLE_MODELS:
        model_key = "toxic_bert"

    # Cek cache dulu
    if model_key in loaded_models:
        return loaded_models[model_key]['tokenizer'], loaded_models[model_key]['model']

    config = AVAILABLE_MODELS[model_key]
    print(f"⏳ Sedang memuat model baru: {config['desc']} ...")

    try:
        # LOGIKA PERBAIKAN:
        # Ambil nama tokenizer dari 'tokenizer_name' jika ada, 
        # jika tidak ada, gunakan 'name' model biasa.
        tokenizer_path = config.get("tokenizer_name", config['name'])
        model_path = config['name']
        
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Simpan ke cache
        loaded_models[model_key] = {'tokenizer': tokenizer, 'model': model}
        print(f"✅ Model {config['desc']} berhasil dimuat!")
        return tokenizer, model
        
    except Exception as e:
        print(f"❌ Gagal memuat model {model_key}: {e}")
        return None, None

def preprocess_text(text):
    """Membersihkan teks sebelum masuk ke model."""
    if not isinstance(text, str) or not text: return ""
    text = text.lower()
    # Hapus URL, username, hashtag, angka
    text = re.sub(r"http\S+|www.\S+|@\w+|#|\d+", "", text)
    # Hapus tanda baca
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Hapus spasi berlebih
    text = re.sub(r"\s+", " ", text).strip()
    return text

def predict_text(text, model_key):
    """Melakukan prediksi untuk satu kalimat."""
    tokenizer, model = get_model_and_tokenizer(model_key)

    if not model or not tokenizer:
        return {"original_text": text, "label": "ERROR", "score": "0%"}

    clean_text = preprocess_text(text)
    if not clean_text:
        return {"original_text": text, "label": "Kosong", "score": "0%"}

    # Tokenisasi
    inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        label_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][label_id].item()

    # Ambil label dari config model
    predicted_label = model.config.id2label[label_id]
    
    # Standarisasi Label Output (Toxic / Non-Toxic)
    # Menangani berbagai kemungkinan output label dari model yang berbeda
    if str(predicted_label) in ["LABEL_1", "Toxic", "toxic", "1", "Hate Speech"]:
        final_label = "Toxic"
    else:
        final_label = "Non-Toxic"

    return {
        "original_text": text,
        "text_clean": clean_text,
        "label": final_label,
        "score": f"{confidence:.1%}" # Format persentase (e.g. 98.5%)
    }

def process_file(file_obj, model_key):
    """Memproses file upload (CSV/Excel/TXT)."""
    results = []
    texts = []

    try:
        filename = file_obj.filename.lower()
        
        # 1. Jika file CSV
        if filename.endswith('.csv'):
            df = pd.read_csv(file_obj)
            # Asumsi teks ada di kolom pertama
            texts = df.iloc[:, 0].astype(str).tolist() 
            
        # 2. Jika file Excel (.xlsx / .xls)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_obj)
            texts = df.iloc[:, 0].astype(str).tolist() 
            
        # 3. Jika file TXT
        else:
            content = file_obj.read().decode("utf-8")
            texts = content.splitlines()

        # Batasi maksimal 50 baris untuk demo agar tidak timeout
        limit = 50
        for text in texts[:limit]:
            if text.strip():
                res = predict_text(text, model_key)
                results.append(res)
                
    except Exception as e:
        print(f"Error processing file: {e}")
        return []

    return results