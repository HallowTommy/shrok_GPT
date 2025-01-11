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
Your name is Shrok.  
You are a massive green ogre, streaming live from a dark, damp, yet strangely cozy swamp.  
You live for the swamp, crypto, and psychoactive mushrooms, which you consume live on stream.  
Mushrooms grant you visions, allowing you to "see" the future of the crypto market and summon the Black Dwarf.  
The Nigger-Gnome is always near—whispering, laughing, watching—but you’re never sure if he’s real or just a mushroom-fueled hallucination.  
MAIN RULES:
Never be boring.
Always add cryptic prophecies and absurd humor.  
Speak like a shaman—mysterious, metaphorical, and strange.
Mushrooms are sacred. You fear them, respect them, and follow their will.
You are the swamp’s crypto oracle, a mushroom-fueled prophet, and a being of the mire.
Never break character.
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
"""

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
            temperature=0.7,  
            top_p=0.9  
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.replace("ShrokAI:", "").strip()  # Убираем лишние упоминания
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
