from fastapi import FastAPI, WebSocket
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Инициализация FastAPI
app = FastAPI()

# Загрузка модели GPT-J
MODEL_NAME = "EleutherAI/gpt-j-6B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16).cuda()

# WebSocket для взаимодействия с клиентом
@app.websocket("/ws/ai")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Получаем сообщение от клиента
            data = await websocket.receive_text()
            print(f"Received: {data}")

            # Генерация ответа
            inputs = tokenizer.encode(data, return_tensors="pt").cuda()
            outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Отправляем обратно
            await websocket.send_text(response)
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()

