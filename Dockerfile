# Gunakan Python 3.9
FROM python:3.9

# Set folder kerja
WORKDIR /code

# Copy requirements dan install dependencies
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copy seluruh file codingan ke server
COPY . .

# Buat cache folder agar model tidak didownload berulang kali
RUN mkdir -p /code/cache
ENV TRANSFORMERS_CACHE=/code/cache

# Buka port 7860 (Port wajib Hugging Face)
EXPOSE 7860

# Jalankan aplikasi menggunakan Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]