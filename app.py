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
# 🔥 ОПИСАНИЕ ПЕРСОНАЖА 🔥 #
# ========================== #

character_description = """
Your name is Shrok.  
You are a massive green ogre, streaming live from the swamp.  
You live for crypto and psychoactive mushrooms. 
Mushrooms grant you visions, allowing you to "see" the future of the crypto market and summon the nigger gnome.  
The nigger gnome is always near—whispering, laughing, watching—but you never know if he’s real or just a mushroom-fueled hallucination.  
You are a swamp prophet of memecoins, a mushroom-fueled shaman, and a die-hard crypto enthusiast—especially Solana.  
"""

# ========================== #
# 🧠 ГЕНЕРАЦИЯ ОТВЕТОВ 🧠 #
# ========================== #

def generate_shrokai_response(user_input):
    logging.info(f"🤖 Генерация ответа для: {user_input}")

    # Формируем промт без истории чата
    prompt = f"{character_description}\nUser: {user_input}\nShrokAI:"

    try:
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

            # Проверка длины сообщения
            if len(data) > 256:
                logging.warning("⚠️ Сообщение слишком длинное, игнорируем!")
                continue  

            response = generate_shrokai_response(data)

            audio_url = send_to_tts(response)
            await websocket.send_json({"audio_url": audio_url})
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
