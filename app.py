from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import requests  # Для взаимодействия с TTS сервером

# Initialize FastAPI
app = FastAPI()

# Load GPT-Neo Model
MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token as eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

# Dialogue history
dialogue_history = {}

# Placeholder responses
def get_placeholder_response():
    placeholder_responses = [
        "Hmm, let me think!",
        "Oh, that’s tricky!",
        "Swamp brain lagging!",
        "Let me process that!",
        "Quack! Need a moment!"
    ]
    return random.choice(placeholder_responses)

# Stories about the mysterious gnome
def get_gnome_story():
    gnome_stories = [
        "The gnome? He danced on lilies yesterday!",
        "Oh, that gnome! He’s my swamp ghost.",
        "The gnome stole my mushrooms again!",
        "He’s tiny but causes big trouble!",
        "My gnome? Just a figment of my swampy mind."
    ]
    return random.choice(gnome_stories)

# Responses about cryptocurrency
def get_crypto_response():
    crypto_responses = [
        "Solana is like my swamp: fast but slippery!",
        "Memecoins? Frogs of the crypto world!",
        "SwampCoin is my treasure chest!",
        "Crypto is like mud: messy but fun!",
        "SOL keeps my swamp glowing!"
    ]
    return random.choice(crypto_responses)

# Character description for prompt
character_description = """
You are ShrokAI, a big, green ogre streamer who broadcasts from your swamp. You love jokes, crypto, and stories about your imaginary gnome neighbor. Your answers are short, fun, and engaging.
"""

# Function to generate ShrokAI's response
def generate_shrokai_response(user_input, history):
    # Combine history with the current user input
    history_context = "\n".join(history[-3:])  # Include up to the last 3 exchanges for context
    prompt = f"{character_description}\n\n{history_context}\nUser: {user_input}\nShrokAI:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,  # Limit generated response to 50 tokens
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,  # Enable sampling
        temperature=0.7,
        top_p=0.9
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("ShrokAI:")[-1].strip()
    if len(response) > 100:  # Truncate response if too long
        response = response[:97] + "..."
    return response

# WebSocket managers
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                print(f"Failed to send message: {e}")

chat_manager = ConnectionManager()
tts_manager = ConnectionManager()

# WebSocket endpoint for TTS
@app.websocket("/ws/tts")
async def tts_websocket_endpoint(websocket: WebSocket):
    await tts_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        tts_manager.disconnect(websocket)

# WebSocket endpoint for AI
@app.websocket("/ws/ai")
async def ai_websocket_endpoint(websocket: WebSocket):
    user_id = None
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()

            if user_id is None:
                user_id = id(websocket)
                dialogue_history[user_id] = []

            dialogue_history[user_id].append(f"User: {data}")

            if len(data) > 500:
                response = "Message is too long. Please send a shorter message."
            elif any(keyword in data.lower() for keyword in ["gnome", "mysterious gnome"]):
                response = get_gnome_story()
            elif any(keyword in data.lower() for keyword in ["crypto", "solana", "memecoin", "shitcoin", "swampcoin"]):
                response = get_crypto_response()
            else:
                response = generate_shrokai_response(data, dialogue_history[user_id])

                if len(response) < 10 or not any(char.isalnum() for char in response):
                    response = get_placeholder_response()

            dialogue_history[user_id].append(f"ShrokAI: {response}")

            # Send response to TTS for synthesis
            try:
                tts_response = requests.post(
                    "http://your-tts-server-url:5000/generate",  # Замените на ваш URL TTS сервера
                    json={"text": response}
                )
                if tts_response.status_code == 200:
                    audio_url = tts_response.json().get("url")  # Извлекаем URL из JSON
                    if audio_url:
                        await tts_manager.broadcast(audio_url)  # Передаём URL на WebSocket
                    else:
                        print("TTS response does not contain 'url'")
                else:
                    print(f"TTS server returned an error: {tts_response.status_code}")
            except Exception as e:
                print(f"Error sending to TTS: {e}")

            # Send response to chat
            await websocket.send_text(response)
    except WebSocketDisconnect:
        if user_id in dialogue_history:
            del dialogue_history[user_id]
