FROM python:3.10-slim

WORKDIR /app

# Установка зависимостей системы
RUN apt-get update && apt-get install -y git ffmpeg libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*

# Установка lama-cleaner
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN git clone https://github.com/Sanster/lama-cleaner.git
WORKDIR /app/lama-cleaner
RUN pip install .

# Возврат в корень
WORKDIR /app

# Копируем скрипты
COPY predict.py .
COPY requirements.txt .
RUN pip install -r requirements.txt

CMD ["python", "predict.py"]