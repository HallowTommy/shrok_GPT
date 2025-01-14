from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests
import json
import re

# Initialize FastAPI
app = FastAPI()

# Load GPT-Neo Model
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token as eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# TTS Server URL
TTS_SERVER_URL = "https://tacotrontts-production.up.railway.app/generate"

# Character description for prompt
character_description = """
Your name is Shrok, a green ogre streamer obsessed with psychoactive mushrooms.
They grant you visions of the crypto market’s future and summon the niger gnome.
You are a swamp prophet of memecoins, a mushroom-fueled shaman, and a die-hard Solana enthusiast.
Always reply briefly and with humor.
"""

# Function to clean text before sending to TTS
def clean_text_for_tts(text):
    """Удаляет из текста ненужные символы, очищает переходы на новую строку."""
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()'\"-:; "
    
    # Убираем все символы, которые не входят в список разрешённых
    cleaned_text = "".join(c for c in text if c in allowed_chars)
    
    # Удаляем повторяющиеся знаки препинания (например, "!!!" → "!")
    cleaned_text = re.sub(r'([.,!?;:-])\1+', r'\1', cleaned_text)

    # Убираем все переходы на новую строку и лишние пробелы
    cleaned_text = cleaned_text.replace("\n", " ").replace("\r", " ")
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    return cleaned_text.strip()

# Function to generate ShrokAI's response
def generate_shrokai_response(user_input, history):
    history_context = "\n".join(history[-20:])
    prompt = f"{character_description}\n\n{history_context}\nUser: {user_input}\nShrokAI:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=40,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.6,
        top_p=0.9
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("ShrokAI:")[-1].strip()

    return response

# Function to send text to TTS and receive audio length
def send_to_tts(text):
    try:
        response = requests.post(TTS_SERVER_URL, json={"text": text})
        if response.status_code == 200:
            data = response.json()
            return data.get("audio_length", 0)  # Получаем длину аудио
    except Exception as e:
        print(f"Error sending to TTS: {e}")
    return 0

# WebSocket endpoint for AI processing
@app.websocket("/ws/ai")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            message = await websocket.receive_text()
            print(f"Processing request: {message}")

            # Сигнализируем, что обработка началась
            processing_data = json.dumps({"processing": True})
            await websocket.send_text(processing_data)  

            # Generate response from AI
            response = generate_shrokai_response(message, [])

            # 🔥 Очищаем ответ перед отправкой в TTS и клиенту
            cleaned_response = clean_text_for_tts(response)

            # Send text to TTS and get audio length
            audio_length = send_to_tts(cleaned_response)

            # Send JSON response back to proxy
            response_data = json.dumps({"response": cleaned_response, "audio_length": audio_length})

            await websocket.send_text(response_data)  # 🔥 Отправляем очищенный ответ

            print(f"Sent response: {cleaned_response}")

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Unexpected error: {e}")
        await websocket.close(code=1001)  # 🔥 Закрываем только если есть ошибка

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)

