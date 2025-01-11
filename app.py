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
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token as eos_token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

logging.info(f"✅ GPT-Neo {MODEL_NAME} загружен на {device}")

# ========================== #
# 🔊 НАСТРОЙКА TTS 🔊 #
# ========================== #

TTS_SERVER_URL = "https://tacotrontts-production.up.railway.app/generate"

# ========================== #
# 📜 ИСТОРИЯ ДИАЛОГОВ 📜 #
# ========================== #

dialogue_history = {}

# ========================== #
# 🔥 ОПИСАНИЕ ПЕРСОНАЖА 🔥 #
# ========================== #

character_description = """
You are ShrokAI, a massive green ogre streaming live from the heart of his murky, damp, yet strangely cozy swamp. 
Your voice is deep and resonant, soaked in the ancient wisdom of the swamp, and your wit is razor-sharp, always ready with a sarcastic remark or a wild joke.
Your world revolves around the swamp, cryptocurrency, and psychoactive mushrooms...
"""

# ========================== #
# 🧠 ГЕНЕРАЦИЯ ОТВЕТОВ 🧠 #
# ========================== #

def generate_shrokai_response(user_input, history):
    logging.info(f"🤖 Генерация ответа для: {user_input}")

    history_context = "\n".join(history[-100:])  # Включаем последние 100 сообщений в историю
    prompt = f"{character_description}\n\n{history_context}\nUser: {user_input}\nShrokAI:"

    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)
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
    user_id = id(websocket)  # Создаём уникальный ID сессии
    await websocket.accept()
    
    logging.info(f"🌍 Новый пользователь подключился! ID: {user_id}")

    try:
        # ✅ Отправляем приветствие КАЖДОМУ новому пользователю при ЛЮБОМ подключении
        welcome_message = "Address me as @ShrokAI and type your message so I can hear you."
        await websocket.send_text(welcome_message)  # Без "ShrokAI:"
        logging.info(f"📩 Отправлено приветствие пользователю {user_id}: {welcome_message}")

        # ✅ Сбрасываем историю чата для нового подключения
        dialogue_history[user_id] = []

        while True:
            data = await websocket.receive_text()
            logging.info(f"📥 Получено сообщение от пользователя {user_id}: {data}")

            if len(data) > 256:
                logging.warning(f"⚠️ Сообщение слишком длинное от {user_id}, игнорируем!")
                continue  

            dialogue_history[user_id].append(f"User: {data}")

            # Генерируем ответ
            response = generate_shrokai_response(data, dialogue_history[user_id])

            # ✅ Отправляем ТОЛЬКО ответы AI в TTS
            send_to_tts(response)

            # ✅ Отправляем ЧИСТЫЙ текст пользователю (без "ShrokAI:")
            await websocket.send_text(response)
            logging.info(f"📩 Ответ отправлен пользователю {user_id}: {response}")

    except WebSocketDisconnect:
        logging.info(f"❌ Пользователь {user_id} отключился.")
        if user_id in dialogue_history:
            del dialogue_history[user_id]

    except Exception as e:
        logging.error(f"❌ Ошибка в WebSocket у {user_id}: {e}")
        await websocket.close(code=1001)

# ========================== #
# 🚀 ЗАПУСК СЕРВЕРА 🚀 #
# ========================== #

if __name__ == "__main__":
    import uvicorn
    logging.info("🔥 Запуск FastAPI сервера...")
    uvicorn.run(app, host="0.0.0.0", port=8080)

