from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
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

# Placeholder responses
def get_placeholder_response():
    placeholder_responses = [
        "Hmm, let me think about that!",
        "Oh wow, that’s a tricky one! Can you ask again?",
        "Sorry, my swampy brain got stuck. Try rephrasing!",
        "Whoa, that’s deep! Ask me again in swampy terms!",
        "Give me a second, my swamp wifi is lagging!",
    ]
    return random.choice(placeholder_responses)

# Stories about the mysterious gnome
def get_gnome_story():
    gnome_stories = [
        "Oh, that gnome! One time, I thought he was stealing my swamp gas to fuel his tiny boat! But turns out, it was just a hallucination.",
        "You mean the gnome that only shows up when I eat too many swamp mushrooms? Yeah, he's always singing jazz in my head.",
    ]
    return random.choice(gnome_stories)

# Responses about cryptocurrency
def get_crypto_response():
    crypto_responses = [
        "Oh, Solana? It's like a frog in my swamp: quick but sometimes slips.",
        "Memecoins? They're like swamp frogs: they jump high but often plop back into the water!",
    ]
    return random.choice(crypto_responses)

# Character description for prompt
character_description = """
You are ShrokAI, a big, green, slightly smelly ogre streamer who broadcasts his streams directly from his cozy, albeit slightly dirty, swamp. You are cheerful, joyful, and love to joke about yourself and your viewers. You speak in a straightforward manner with a touch of sarcasm. You sometimes mention your neighbor, a mysterious gnome whom you see but have never spoken to. You are a big fan of the Solana network and the SOL token, and you often talk about memecoins like Dogecoin, PepeCoin, and your favorite SwampCoin. Remember to include your typical phrases and speech characteristics in your responses, like adding unexpected words or sounds such as "Quack!" or "Splat!" to spice up your speech.
"""

# Function to generate ShrokAI's response
def generate_shrokai_response(user_input, history):
    # Combine history with the current user input
    history_context = "\n".join(history[-5:])  # Include up to the last 5 exchanges for context
    prompt = f"{character_description}\n\n{history_context}\nUser: {user_input}\nShrokAI:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=150,  # Limit generated response to 150 tokens
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,  # Enable sampling
        temperature=0.8,
        top_p=0.9
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split("ShrokAI:")[-1].strip()
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

            # Send response to client
            await websocket.send_text(response)
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        if user_id in dialogue_history:
            del dialogue_history[user_id]  # Remove history for the disconnected user
    except Exception as e:
        print(f"Unexpected error: {e}")
        await websocket.close(code=1001)
