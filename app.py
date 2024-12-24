from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
import requests
import time

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
TTS_DELETE_URL = "https://tacotrontts-production.up.railway.app/delete"

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
def generate_shrokai_response(user_input):
    prompt = f"{character_description}\nUser: {user_input}\nShrokAI:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=50,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("ShrokAI:")[-1].strip()
    return response if response else get_placeholder_response()

# Function to send text to TTS and get audio URL
def send_to_tts(text):
    try:
        response = requests.post(TTS_SERVER_URL, json={"text": text})
        if response.status_code == 200:
            data = response.json()
            return data.get("audio_url", "")
        else:
            print(f"TTS server error: {response.status_code}")
            return ""
    except Exception as e:
        print(f"Error sending to TTS: {e}")
        return ""

# WebSocket endpoint for client interaction
@app.websocket("/ws/ai")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        latest_audio_url = ""  # Stores the latest audio URL

        while True:
            data = await websocket.receive_text()
            print(f"Received: {data}")

            if "gnome" in data.lower():
                response = get_gnome_story()
            elif "crypto" in data.lower():
                response = get_crypto_response()
            else:
                response = generate_shrokai_response(data)

            try:
                audio_url = send_to_tts(response)
                if audio_url:
                    latest_audio_url = audio_url
                    await websocket.send_json({"audio_url": audio_url})
                    print(f"Sent audio URL: {audio_url}")
                else:
                    print("No audio_url received from TTS.")
            except Exception as e:
                print(f"Error in TTS integration: {e}")

            # Send the latest audio to newly connected clients
            if latest_audio_url:
                await websocket.send_json({"audio_url": latest_audio_url})

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Unexpected error: {e}")
        await websocket.close(code=1001)

