from fastapi import FastAPI, WebSocket
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Инициализация FastAPI
app = FastAPI()

# Загрузка модели GPT-Neo
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

@app.websocket("/ws/ai")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Получаем сообщение от клиента
            data = await websocket.receive_text()
            print(f"Received: {data}")

            # Генерация ответа
            inputs = tokenizer.encode(data, return_tensors="pt")
            outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Отправляем обратно
            await websocket.send_text(response)
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()
