from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import requests
import json

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

# Function to send text to TTS server and get audio URL
def send_to_tts_server(text):
    try:
        tts_url = "https://tacotrontts-production.up.railway.app/generate"
        headers = {"Content-Type": "application/json"}
        payload = {"text": text}
        print(f"Sending text to TTS: {text}")  # Debug log
        response = requests.post(tts_url, headers=headers, json=payload)

        if response.status_code == 200:
            audio_url = response.json().get("url")
            print(f"Received audio URL from TTS: {audio_url}")  # Debug log
            return audio_url
        else:
            print(f"TTS server error: {response.text}")  # Log error response
            return None
    except Exception as e:
        print(f"Error sending to TTS server: {e}")  # Log exception
        return None

# WebSocket endpoint for client interaction
@app.websocket("/ws/ai")
async def websocket_endpoint(websocket: WebSocket):
    user_id = None
    await websocket.accept()
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            print(f"Received: {data}")

            # Initialize dialogue history for this user
            if user_id is None:
                user_id = id(websocket)
                dialogue_history[user_id] = []

            # Add user message to history
            dialogue_history[user_id].append(f"User: {data}")

            # Check message length
            if len(data) > 500:
                response = "Message is too long. Please send a shorter message."
            elif any(keyword in data.lower() for keyword in ["gnome", "mysterious gnome"]):
                response = get_gnome_story()
            elif any(keyword in data.lower() for keyword in ["crypto", "solana", "memecoin", "shitcoin", "swampcoin"]):
                response = get_crypto_response()
            else:
                response = generate_shrokai_response(data, dialogue_history[user_id])

                # Check for meaningless response
                if len(response) < 10 or not any(char.isalnum() for char in response):
                    response = get_placeholder_response()

            # Add ShrokAI's response to history
            dialogue_history[user_id].append(f"ShrokAI: {response}")

            # Send response to TTS and get audio URL
            audio_url = send_to_tts_server(response)
            if audio_url:
                # Broadcast audio URL to all connected clients
                await websocket.send_text(json.dumps({"text": response, "audio_url": audio_url}))
            else:
                # Fallback to text-only response if TTS fails
                await websocket.send_text(json.dumps({"text": response}))

    except WebSocketDisconnect:
        print("WebSocket disconnected")
        if user_id in dialogue_history:
            del dialogue_history[user_id]  # Remove history for the disconnected user
    except Exception as e:
        print(f"Unexpected error: {e}")
        await websocket.close(code=1001)
