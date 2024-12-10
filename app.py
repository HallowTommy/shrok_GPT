from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Инициализация FastAPI
app = FastAPI()

# Загрузка модели GPT-Neo
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

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
            
            # Генерация ответа
            inputs = tokenizer(data, return_tensors="pt", padding=True, truncation=True, max_length=128)
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=150,  # Ограничение длины ответа
                num_return_sequences=1,
                no_repeat_ngram_size=2,  # Уменьшение повторений
                pad_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Логика проверки ответа
            if not response.strip():
                response = "I'm sorry, I didn't understand that."
            
            # Отправка ответа клиенту
            await websocket.send_text(response)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Unexpected error: {e}")
        await websocket.close(code=1001)
