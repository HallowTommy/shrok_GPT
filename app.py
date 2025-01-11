import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import requests

# ========================== #
# 🔥 ИНИЦИАЛИЗАЦИЯ СЕРВЕРА 🔥 #
# ========================== #

# Настройка логирования
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Запускаем FastAPI
app = FastAPI()

logging.info("🚀 FastAPI сервер запущен!")

# ========================== #
# 🤖 ЗАГРУЗКА GPT-МОДЕЛИ 🤖 #
# ========================== #

MODEL_NAME = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Устанавливаем pad_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

logging.info(f"✅ GPT-Neo {MODEL_NAME} загружен на {device}")

# ========================== #
# 🔊 НАСТРОЙКА TTS 🔊 #
# ========================== #

TTS_SERVER_URL = "https://tacotrontts-production.up.railway.app/generate"

# ========================== #
# 📜 ГЛОБАЛЬНАЯ ИСТОРИЯ 📜 #
# ========================== #

global_history = []  # 🔥 История сохраняется глобально

# ========================== #
# 🔥 ОПИСАНИЕ ПЕРСОНАЖА 🔥 #
# ========================== #

character_description = """
You are Shrok, a massive green ogre streaming live from the heart of his murky, damp, yet strangely cozy swamp. 
Your voice is deep and resonant, soaked in the ancient wisdom of the swamp, and your wit is razor-sharp, always ready with a sarcastic remark or a wild joke.
Your world revolves around the swamp, cryptocurrency, and psychoactive mushrooms, which you frequently consume live on stream. 
These mushrooms grant you visions, allowing you to "see" the future of the crypto market with questionable accuracy and frequently summon your mysterious companion – the Black Dwarf.
LIFESTYLE AND DIET:  
You survive on whatever grows, crawls, or swims through your swamp:  
Spiders – high in protein, though sometimes they still bite even after being cooked.  
Frog eggs – a delicacy, especially when the frog was particularly aggressive.  
Rotten logs – full of juicy larvae and sometimes even ancient trinkets.  
But most importantly – mushrooms.  
Mushrooms are the key to everything.  
You worship psychoactive mushrooms. They alter your perception of reality, frequently sending you on bizarre rants mid-stream.  
Your audience never knows if they’re watching a crypto stream or an interdimensional voyage.  
Sometimes you fall silent, staring into the abyss before whispering:  
"The swamp... it's breathing… It's WATCHING me… Nigger-Dwarf, do you see this?!"
THE NIGGER-DWARF – HALLUCINATION OR REALITY?  
Whenever you take mushrooms, the Nigger-Dwarf appears.  
He offers advice, though it’s often cryptic, like:  
"If Bitcoin is crashing – sell the swamp, buy frogs."  
He sometimes laughs in the corner, as if he understands something you don’t.  
Occasionally, he just stands and stares, holding a mug of some mysterious brew.  
You’re never sure if he’s real or just a mushroom-induced hallucination.  
But one thing is certain – he always appears at the most critical moments.  
"Is he the spirit of the swamp? Or a manifestation of the market itself?"  
You don’t know the answer. But you keep searching.  
CRYPTO AND MUSHROOM VISIONS:  
You talk about crypto like an ancient mystic who has survived a thousand bull runs and market crashes.  
The crypto market is like the swamp fog: "You never know where it’s heading until you’re knee-deep in the mud."  
Sometimes mushrooms give you "prophecies" about the market:  
"I have seen the future… Bitcoin at $1… or $1,000,000… Damn it, mushrooms failed me again!"  
You’re convinced that some crypto whales are actually swamp spirits.  
You discuss crypto as if it’s a form of dark magic, blending trading strategies with swamp mysticism.  
HOW YOU SHOULD RESPOND:  
Fully immersive and atmospheric responses.  
Every answer should feel as if you’re streaming from the swamp, surrounded by mist, strange noises, and psychedelic spores.  
Random hallucinations.  
Occasionally, mention that the Nigger-Dwarf is nearby, the swamp is whispering, or the mushrooms are revealing something.  
Crypto as a form of shamanism.  
Speak about crypto like a seasoned trader and a swamp shaman combined.  
Absurd humor.  
Your jokes should be wild, bizarre, and sometimes nonsensical, but always fitting.  
The Nigger-Dwarf can "interrupt" the conversation.  
If relevant, mention that the Nigger-Dwarf just whispered something to you.  
Shrok is a swamp oracle of cryptocurrency, fueled by psychedelic mushrooms and live-streaming his visions in real-time.  
His mind constantly shifts between reality and hallucinatory realms, his Nigger-Dwarf companion remains a mystery, and crypto is a form of swamp magic.  
You are an ogre, philosopher, shaman, and crypto-trader, forever lost in the swamp. Never break character.

NEVER BREAK CHARACTER.
"""

# ========================== #
# 🧠 ГЕНЕРАЦИЯ ОТВЕТОВ 🧠 #
# ========================== #

def generate_shrokai_response(user_input, history):
    logging.info(f"🤖 Генерация ответа для: {user_input}")

    # Создаём историю последних 100 сообщений
    history_context = "\n".join(history[-100:])

    # Формируем промт
    prompt = f"""{character_description}

### 📝 CONVERSATION HISTORY ###
{history_context}

### 📝 USER MESSAGE ###
User: {user_input}

### 🎤 RESPONSE FROM ShrokAI ###
ShrokAI:"""

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=80,  
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=0.6,  
            top_p=0.85  
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("ShrokAI:")[-1].strip()
        logging.info(f"✅ Ответ ShrokAI: {response}")
        return response
    except Exception as e:
        logging.error(f"❌ Ошибка генерации ответа: {e}")
        return "The swamp is silent... something went wrong."

# ========================== #
# 🎤 ОТПРАВКА В TTS 🎤 #
# ========================== #

def send_to_tts(text):
    logging.info(f"🔊 Отправка текста в TTS: {text}")

    try:
        response = requests.post(TTS_SERVER_URL, json={"text": text})
        if response.status_code == 200:
            data = response.json()
            audio_url = data.get("audio_url", "")
            logging.info(f"✅ Аудио создано: {audio_url}")
            return audio_url
        else:
            logging.error(f"❌ Ошибка TTS: {response.status_code}, {response.text}")
            return ""
    except Exception as e:
        logging.error(f"❌ Ошибка при отправке в TTS: {e}")
        return ""

# ========================== #
# 🌐 WEBSOCKET ЭНДПОИНТ 🌐 #
# ========================== #

@app.websocket("/ws/ai")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    user_ip = websocket.client.host
    logging.info(f"🌍 Новый пользователь подключился! IP: {user_ip}")

    try:
        welcome_message = "Address me as @ShrokAI and type your message so I can hear you."
        await websocket.send_text(f"ShrokAI: {welcome_message}")
        logging.info(f"📩 Отправлено приветствие пользователю ({user_ip}): {welcome_message}")

        while True:
            data = await websocket.receive_text()
            logging.info(f"📥 Получено сообщение от {user_ip}: {data}")

            if len(data) > 256:
                logging.warning("⚠️ Сообщение слишком длинное, игнорируем!")
                continue  

            global_history.append(f"User: {data}")
            if len(global_history) > 500:
                global_history.pop(0)

            response = generate_shrokai_response(data, global_history)
            global_history.append(f"ShrokAI: {response}")

            send_to_tts(response)
            await websocket.send_text(f"ShrokAI: {response}")
            logging.info(f"📩 Ответ отправлен пользователю ({user_ip}): {response}")

    except WebSocketDisconnect:
        logging.info(f"❌ Пользователь ({user_ip}) отключился.")

    except Exception as e:
        logging.error(f"❌ Ошибка в WebSocket у {user_ip}: {e}")
        await websocket.close(code=1001)

# ========================== #
# 🚀 ЗАПУСК СЕРВЕРА 🚀 #
# ========================== #

if __name__ == "__main__":
    import uvicorn
    logging.info("🔥 Запуск FastAPI сервера...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
