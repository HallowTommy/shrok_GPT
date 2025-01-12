from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests
import time
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

# Character description for prompt
character_description = """
You are ShrokAI, a big, green ogre streamer who broadcasts from your swamp. You love jokes, crypto, and stories about your mysterious gnome neighbor. Your answers are short, fun, and engaging.
"""

# Function to generate ShrokAI's response
def generate_shrokai_response(user_input, history):
    history_context = "\n".join(history[-20:])
    prompt = f"{character_description}\n\n{history_context}\nUser: {user_input}\nShrokAI:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=100,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.7,
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
            data = await websocket.receive_text()
            print(f"Received: {data}")

            # Block new requests if processing is active
            if is_processing:
                await websocket.send_text("ShrokAI is busy, try again later!")
                continue
            
            is_processing = True  # Block new messages during processing

            dialogue_history[user_id].append(f"User: {data}")
            response = generate_shrokai_response(data, dialogue_history[user_id])
            dialogue_history[user_id].append(f"ShrokAI: {response}")

            # Send response to TTS and get audio length
            audio_length = send_to_tts(response)
            
            # Start unblock timer
            asyncio.create_task(unblock_after_delay())

            # Broadcast only text response to all users
            for connection in active_connections:
                await connection.send_text(response)
                print(f"Sent to client: {response}")

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
