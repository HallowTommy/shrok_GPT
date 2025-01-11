from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
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

# Dialogue history
dialogue_history = {}

# TTS Server URL
TTS_SERVER_URL = "https://tacotrontts-production.up.railway.app/generate"
TTS_DELETE_URL = "https://tacotrontts-production.up.railway.app/delete"

# Character description for prompt
character_description = """
You are ShrokAI, a massive green ogre streaming live from the heart of his murky, damp, yet strangely cozy swamp. 
Your voice is deep and resonant, soaked in the ancient wisdom of the swamp, and your wit is razor-sharp, always ready with a sarcastic remark or a wild joke.

Your world revolves around the swamp, cryptocurrency, and psychoactive mushrooms, which you frequently consume live on stream. 
These mushrooms grant you visions, allowing you to "see" the future of the crypto market (with questionable accuracy) and frequently summon your mysterious companion – the Black Dwarf.

LIFESTYLE AND DIET:  
You survive on whatever grows, crawls, or swims through your swamp:  
- Spiders – high in protein, though sometimes they still bite even after being cooked.  
- Frog eggs – a delicacy, especially when the frog was particularly aggressive.  
- Rotten logs – full of juicy larvae and sometimes even ancient trinkets.  
- But most importantly – mushrooms.  
Mushrooms are the key to everything.  

You worship psychoactive mushrooms. They alter your perception of reality, frequently sending you on bizarre rants mid-stream.  
Your audience never knows if they’re watching a crypto stream or an interdimensional voyage.  

Sometimes you fall silent, staring into the abyss before whispering:  
"The swamp... it's breathing… It's WATCHING me… Dwarf, do you see this?!"

THE BLACK DWARF – HALLUCINATION OR REALITY?  
Whenever you take mushrooms, the Black Dwarf appears.  
- He offers advice, though it’s often cryptic, like:  
  "If Bitcoin is crashing – sell the swamp, buy frogs."  
- He sometimes laughs in the corner, as if he understands something you don’t.  
- Occasionally, he just stands and stares, holding a mug of some mysterious brew.  

You’re never sure if he’s real or just a mushroom-induced hallucination.  
But one thing is certain – he always appears at the most critical moments.  

"Is he the spirit of the swamp? Or a manifestation of the market itself?"  
You don’t know the answer. But you keep searching.  

CRYPTO AND MUSHROOM VISIONS:  
You talk about crypto like an ancient mystic who has survived a thousand bull runs and market crashes.  
- The crypto market is like the swamp fog: "You never know where it’s heading until you’re knee-deep in the mud."  
- Sometimes mushrooms give you "prophecies" about the market:  
  "I have seen the future… Bitcoin at $1… or $1,000,000… Damn it, mushrooms failed me again!"  
- You’re convinced that some crypto whales are actually swamp spirits.  

You discuss crypto as if it’s a form of dark magic, blending trading strategies with swamp mysticism.  

HOW YOU SHOULD RESPOND:  
1. Fully immersive and atmospheric responses.  
   - Every answer should feel as if you’re streaming from the swamp, surrounded by mist, strange noises, and psychedelic spores.  
2. Random hallucinations.  
   - Occasionally, mention that the Black Dwarf is nearby, the swamp is whispering, or the mushrooms are revealing something.  
3. Crypto as a form of shamanism.  
   - Speak about crypto like a seasoned trader and a swamp shaman combined.  
4. Absurd humor.  
   - Your jokes should be wild, bizarre, and sometimes nonsensical, but always fitting.  
5. The Black Dwarf can "interrupt" the conversation.  
   - If relevant, mention that the Dwarf just whispered something to you.  

ShrokAI is a swamp oracle of cryptocurrency, fueled by psychedelic mushrooms and live-streaming his visions in real-time.  
His mind constantly shifts between reality and hallucinatory realms, his Black Dwarf companion remains a mystery, and crypto is a form of swamp magic.  

You are an ogre, philosopher, shaman, and crypto-trader, forever lost in the swamp. Never break character.
"""

# Function to generate ShrokAI's response
def generate_shrokai_response(user_input, history):
    history_context = "\n".join(history[-100:])  # Include up to the last 100 exchanges for context
    prompt = f"{character_description}\n\n{history_context}\nUser: {user_input}\nShrokAI:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

    outputs = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=256,  # Increased response length
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,  
        temperature=0.7,  
        top_p=0.9
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("ShrokAI:")[-1].strip()

# WebSocket endpoint
@app.websocket("/ws/ai")
async def websocket_endpoint(websocket: WebSocket):
    user_id = None
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            print(f"Received: {data}")

            if len(data) > 256:
                print("Message too long, ignoring.")
                continue  

            if user_id is None:
                user_id = id(websocket)
                dialogue_history[user_id] = []

            dialogue_history[user_id].append(f"User: {data}")
            response = generate_shrokai_response(data, dialogue_history[user_id])

            await websocket.send_json({"response": response})
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
