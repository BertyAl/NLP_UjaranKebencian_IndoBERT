import torch
import torch.nn.functional as F
import re
import string
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Menggunakan model publik dari Hugging Face
MODEL_NAME = "Dzeisonov/indobert-toxic-classifier"

print(f"⏳ Sedang menghubungkan ke Hugging Face: {MODEL_NAME}...")
print("   (Jika ini pertama kali, proses download akan memakan waktu 1-2 menit)")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    print("✅ Model Berhasil Dimuat dan Siap Digunakan!")
except Exception as e:
    print(f"❌ Gagal memuat model. Periksa koneksi internet Anda.\nError: {e}")
    tokenizer = None
    model = None


def preprocess_text(text):
    if not text: return ""
    text = text.lower()
    text = re.sub(r"http\S+|www.\S+|@\w+|#|\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_toxicity(text):
    if not model or not tokenizer:
        return {"text": text, "label": "ERROR", "score": "Model gagal dimuat"}

    clean_text = preprocess_text(text)

    # Tokenisasi
    inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)

        # Ambil ID kelas tertinggi
        label_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][label_id].item()

    # Kita coba ambil label asli dari config model
    predicted_label = model.config.id2label[label_id]

    # Terjemahkan ke Bahasa Indonesia
    if predicted_label in ["LABEL_1", "Toxic", "toxic", "1"]:
        final_label = "Toxic"
    else:
        final_label = "Non-Toxic"

    return {
        "text_clean": clean_text,
        "label": final_label,
        "score": f"{confidence:.1%}"
    }