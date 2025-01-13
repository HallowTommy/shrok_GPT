from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests
import asyncio

# Initialize FastAPI
app = FastAPI()

# Load GPT-Neo Model
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token as eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# List of active WebSocket connections
active_connections = set()
# Dialogue history for each user
dialogue_history = {}

# Global control variables
is_processing = False  # Blocks new requests while a response is being generated
block_time = 0  # Stores the time (in seconds) for which new requests are blocked

# TTS Server URL
TTS_SERVER_URL = "https://tacotrontts-production.up.railway.app/generate"

# Welcome message
WELCOME_MESSAGE = "Address me as @ShrokAI and type your message so I can hear you."
BUSY_MESSAGE = "ShrokAI is busy, please wait for the current response to complete."

# Character description for prompt
character_description = """
Your name is Shrok, a green ogre streamer obsessed with psychoactive mushrooms.
They grant you visions of the crypto market’s future and summon the niger gnome.
You are a swamp prophet of memecoins, a mushroom-fueled shaman, and a die-hard Solana enthusiast.
Always reply briefly and with humor.
"""

# Function to generate ShrokAI's response
def generate_shrokai_response(user_input, history):
    history_context = "\n".join(history[-20:])
    prompt = f"{character_description}\n\n{history_context}\nUser: {user_input}\nShrokAI:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=20,
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
    global block_time
    try:
        response = requests.post(TTS_SERVER_URL, json={"text": text})
        if response.status_code == 200:
            data = response.json()
            audio_length = data.get("audio_length", 0)  # Получаем длину аудио
            block_time = audio_length + 10  # Устанавливаем время блокировки
            return audio_length
    except Exception as e:
        print(f"Error sending to TTS: {e}")
    return 0

# Function to handle blocking logic
async def unblock_after_delay():
    global is_processing
    print(f"Blocking requests for {block_time} seconds...")
    await asyncio.sleep(block_time)
    is_processing = False
    print("Unblocking requests.")

# WebSocket endpoint for client interaction
@app.websocket("/ws/ai")
async def websocket_endpoint(websocket: WebSocket):
    global is_processing
    await websocket.accept()
    active_connections.add(websocket)
    user_id = id(websocket)
    dialogue_history[user_id] = []
    
    # Send welcome message to new user
    await websocket.send_text(WELCOME_MESSAGE)
    
    try:
        while True:
            # Немедленная проверка блокировки перед приемом сообщений
            if is_processing:
                await websocket.send_text(BUSY_MESSAGE)
                continue  # НЕ передавать сообщение дальше

            # Получаем сообщение от пользователя
            data = await websocket.receive_text()
            print(f"Processing request: {data}")

            # Отправляем BUSY_MESSAGE всем пользователям (фиксируем момент начала обработки)
            for connection in list(active_connections):
                try:
                    await connection.send_text(BUSY_MESSAGE)
                except Exception as e:
                    print(f"Failed to send busy message to a client: {e}")
                    active_connections.remove(connection)

            # Помечаем, что началась обработка
            is_processing = True

            # Добавляем сообщение в историю и генерируем ответ
            dialogue_history[user_id].append(f"User: {data}")
            response = generate_shrokai_response(data, dialogue_history[user_id])
            dialogue_history[user_id].append(f"ShrokAI: {response}")

            # Рассылаем ответ от ИИ всем пользователям
            for connection in list(active_connections):
                try:
                    await connection.send_text(response)
                except Exception as e:
                    print(f"Failed to send message to a client: {e}")
                    active_connections.remove(connection)
                print(f"Sent to client: {response}")

            # Отправляем текст в TTS и получаем длительность аудиофайла
            audio_length = send_to_tts(response)
            
            # Запускаем таймер разблокировки
            asyncio.create_task(unblock_after_delay())

    except WebSocketDisconnect:
        print("WebSocket disconnected")
        active_connections.remove(websocket)
        del dialogue_history[user_id]
    except Exception as e:
        print(f"Unexpected error: {e}")
        await websocket.close(code=1001)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
