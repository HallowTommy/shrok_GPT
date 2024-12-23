from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import httpx
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

# Function to generate ShrokAI's response
def generate_shrokai_response(user_input):
    prompt = f"You are ShrokAI, a big, green ogre streamer who broadcasts from your swamp. You love jokes, crypto, and stories about your imaginary gnome neighbor.\nUser: {user_input}\nShrokAI:"
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
    return response[:100] if len(response) > 100 else response

# Async function to send text to TTS server and get audio URL
async def send_to_tts_server(text):
    tts_url = "https://tacotrontts-production.up.railway.app/generate"
    headers = {"Content-Type": "application/json"}
    payload = {"text": text}

    try:
        print(f"[DEBUG] Sending text to TTS: {text}")
        async with httpx.AsyncClient() as client:
            response = await client.post(tts_url, headers=headers, json=payload)

        if response.status_code == 200:
            audio_url = response.json().get("url")
            print(f"[DEBUG] Received audio URL from TTS: {audio_url}")
            return audio_url
        else:
            print(f"[ERROR] TTS server error: {response.text}")
            return None
    except Exception as e:
        print(f"[ERROR] Error sending to TTS server: {e}")
        return None

# WebSocket endpoint for client interaction
@app.websocket("/ws/ai")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            print(f"[DEBUG] Received from client: {data}")

            # Generate AI response
            response = generate_shrokai_response(data)
            print(f"[DEBUG] Generated response: {response}")

            # Send response to TTS
            audio_url = await send_to_tts_server(response)

            # Send response back to the client
            if audio_url:
                await websocket.send_text(json.dumps({"text": response, "audio_url": audio_url}))
            else:
                await websocket.send_text(json.dumps({"text": response}))
    except WebSocketDisconnect:
        print("[DEBUG] WebSocket disconnected")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        await websocket.close(code=1001)
