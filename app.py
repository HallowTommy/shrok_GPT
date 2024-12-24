import requests  # Для отправки HTTP-запросов
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import random
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

# TTS server URL
TTS_SERVER_URL = "tacotrontts-production.up.railway.app/generate"

# Clean text for TTS
def clean_text_for_tts(text):
    """
    Удаляет неподдерживаемые символы (@, # и т.д.) и обрезает длину текста.
    """
    # Удаляем все символы, кроме латинских букв, цифр, знаков пунктуации и пробелов
    text = re.sub(r"[^a-zA-Z0-9,.!? ]", "", text)

    # Убираем двойные пробелы, если они есть
    text = re.sub(r"\s+", " ", text)

    return text.strip()

# Send text to TTS server
def send_to_tts(text):
    try:
        print(f"Sending text to TTS: {text}")
        response = requests.post(TTS_SERVER_URL, json={"text": text})
        print(f"TTS Response: {response.status_code}, {response.text}")
        if response.status_code == 200:
            data = response.json()
            audio_url = data.get("audio_url", "")
            print(f"Extracted audio_url: {audio_url}")
            return audio_url
        else:
            print(f"TTS server error: {response.status_code}")
            return ""
    except Exception as e:
        print(f"Error sending to TTS: {e}")
        return ""

# Function to generate ShrokAI's response
def generate_shrokai_response(user_input, history):
    history_context = "\n".join(history[-3:])  # Include up to the last 3 exchanges for context
    prompt = f"{character_description}\n\n{history_context}\nUser: {user_input}\nShrokAI:"
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
    response = response[:50]  # Limit response to 50 characters
    return response

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

            # Generate ShrokAI's response
            response = generate_shrokai_response(data, dialogue_history[user_id])

            # Add ShrokAI's response to history
            dialogue_history[user_id].append(f"ShrokAI: {response}")

            # Send response text and audio link
            audio_url = send_to_tts(response)
            if audio_url:
                await websocket.send_json({"text": response, "audio_url": audio_url})
            else:
                await websocket.send_json({"text": response, "error": "TTS generation failed."})
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        if user_id in dialogue_history:
            del dialogue_history[user_id]
    except Exception as e:
        print(f"Unexpected error: {e}")
        await websocket.close(code=1001)
