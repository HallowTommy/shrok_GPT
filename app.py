from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
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

# TTS Server URL
TTS_SERVER_URL = "https://tacotrontts-production.up.railway.app/generate"

# Character description for prompt
character_description = """
Your name is Shrok, a green ogre streamer obsessed with psychoactive mushrooms.
They grant you visions of the crypto market’s future and summon the black gnome.
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

            # Generate response from AI
            response = generate_shrokai_response(message, [])

            # Send text to TTS and get audio length
            audio_length = send_to_tts(response)

            # Send JSON response back to proxy
            response_data = json.dumps({"response": response, "audio_length": audio_length})
            await websocket.send_text(response_data)

            print(f"Sent response: {response}")

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"Unexpected error: {e}")
        await websocket.close(code=1001)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
