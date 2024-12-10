from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Инициализация FastAPI
app = FastAPI()

# Загрузка модели GPT-Neo
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Устройство для работы модели (CPU, так как нет GPU)
device = torch.device("cpu")
model = model.to(device)

# WebSocket для взаимодействия с клиентом
@app.websocket("/ws/ai")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Получение сообщения
            data = await websocket.receive_text()
            print(f"Received: {data}")

            # Проверка длины сообщения
            if len(data) > 500:
                await websocket.send_text("Message is too long. Please send a shorter message.")
                continue

            # Очистка сообщения
            sanitized_data = data.strip()

            # Генерация ответа
            inputs = tokenizer(sanitized_data, return_tensors="pt", truncation=True).to(device)
            outputs = model.generate(
                inputs["input_ids"],
                max_length=150,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=inputs["attention_mask"]
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # Отправка ответа клиенту
            await websocket.send_text(response)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Unexpected error: {e}")
        await websocket.close(code=1001)
