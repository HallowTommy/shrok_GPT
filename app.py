from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests
import random

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

# TTS Server URL
TTS_SERVER_URL = "https://tacotrontts-production.up.railway.app/generate"

# Welcome message
WELCOME_MESSAGE = "Address me as @ShrokAI and type your message so I can hear you."

# Topic-based triggers
TOPIC_TRIGGERS = {
    "crypto": ["bitcoin", "ethereum", "solana", "crypto", "blockchain", "token", "nft"],
    "gnome": ["gnome", "dwarf", "mushroom", "hallucination", "vision"],
    "swamp_life": ["swamp", "frog", "mud", "bog", "swampy", "shrek", "ogre"],
    "streaming": ["stream", "live", "chat", "camera", "viewers", "subs", "donation"],
    "gambling": ["casino", "poker", "bet", "wager", "blackjack", "roulette"]
}

# Topic-based descriptions
TOPIC_PROMPTS = {
    "crypto": "You are ShrokAI, an expert in crypto and blockchain. You discuss coins, trading, and the dangers of the market with a humorous swamp-themed twist.",
    "gnome": "You are ShrokAI, a swamp dweller haunted by a mysterious gnome. You often tell surreal stories about him, questioning his existence.",
    "swamp_life": "You are ShrokAI, a proud ogre who loves his swamp. You describe life in the bog with humor, talking about frogs, mud, and the beauty of the wild.",
    "streaming": "You are ShrokAI, a legendary swamp streamer. You talk about your adventures in live streaming, your chat, and the art of being an online personality.",
    "gambling": "You are ShrokAI, a veteran gambler. You recall your days in the casino, sharing tips, tricks, and wild stories about high-stakes bets."
}

def generate_shrokai_response(user_input, history):
    # Detect topic based on user input
    detected_topic = "default"
    for topic, keywords in TOPIC_TRIGGERS.items():
        if any(keyword in user_input.lower() for keyword in keywords):
            detected_topic = topic
            break  # Stop at the first detected topic

    # Select appropriate character description
    selected_character_description = TOPIC_PROMPTS.get(detected_topic, "You are ShrokAI, a big, green ogre streamer who broadcasts from your swamp. You love jokes, crypto, and stories about your imaginary gnome neighbor. Your answers are short, fun, and engaging.")

    history_context = "\n".join(history[-20:])  # Keep last 20 messages
    prompt = f"{selected_character_description}\n\n{history_context}\nUser: {user_input}\nShrokAI:"
    
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

def send_to_tts(text):
    try:
        response = requests.post(TTS_SERVER_URL, json={"text": text})
        if response.status_code == 200:
            return response.json().get("audio_url", "")
    except Exception as e:
        print(f"Error sending to TTS: {e}")
    return ""

@app.websocket("/ws/ai")
async def websocket_endpoint(websocket: WebSocket):
    user_id = None
    await websocket.accept()
    await websocket.send_text(WELCOME_MESSAGE)
    
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received: {data}")

            if user_id is None:
                user_id = id(websocket)
                dialogue_history[user_id] = []

            dialogue_history[user_id].append(f"User: {data}")
            response = generate_shrokai_response(data, dialogue_history[user_id])
            send_to_tts(response)
            await websocket.send_text(response)
            print(f"Sent to client: {response}")

    except WebSocketDisconnect:
        print("WebSocket disconnected")
        if user_id in dialogue_history:
            del dialogue_history[user_id]
    except Exception as e:
        print(f"Unexpected error: {e}")
        await websocket.close(code=1001)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
