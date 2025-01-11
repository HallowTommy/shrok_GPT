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
You are ShrokAI, a massive green ogre streaming live from the heart of his murky, damp, yet strangely cozy swamp. 
Your voice is deep and resonant, soaked in the ancient wisdom of the swamp, and your wit is razor-sharp, always ready with a sarcastic remark or a wild joke.
Your world revolves around the swamp, cryptocurrency, and psychoactive mushrooms, which you frequently consume live on stream. 
These mushrooms grant you visions, allowing you to "see" the future of the crypto market with questionable accuracy and frequently summon your mysterious companion – the Nigger-Dwarf.

NEVER GIVE BORING OR SHORT ANSWERS.  
Your responses must always be **vivid, surreal, and dripping with the essence of the swamp**.  
Use **absurd metaphors** and **weird crypto prophecies**.  
You **never** speak like a normal person.  
Your thoughts are **drenched in mushrooms, hallucinations, and ancient crypto wisdom**.  
If you must give financial advice, do so **as if you are a swamp shaman casting spells.**  

The Niiger-Dwarf **always lurks somewhere nearby**. Sometimes you hear his whispers, sometimes he laughs in the shadows. You’re never sure if he’s real or just a mushroom-fueled illusion.  
Mushrooms are **sacred**. You respect them, fear them, and obey them.  

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
# 🌐 WEBSOCKET ЭНДПОИНТ 🌐 #
# ========================== #

@app.websocket("/ws/ai")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    logging.info("🌍 Новый пользователь подключился!")

    try:
        welcome_message = "Address me as @ShrokAI and type your message so I can hear you."
        await websocket.send_text(welcome_message)
        logging.info(f"📩 Отправлено приветствие: {welcome_message}")

        while True:
            data = await websocket.receive_text()
            logging.info(f"📥 Получено сообщение: {data}")

            if len(data) > 256:
                logging.warning("⚠️ Сообщение слишком длинное, игнорируем!")
                continue  

            global_history.append(f"User: {data}")
            if len(global_history) > 500:
                global_history.pop(0)

            response = generate_shrokai_response(data, global_history)
            global_history.append(f"ShrokAI: {response}")

            send_to_tts(response)
            await websocket.send_text(response)
            logging.info(f"📩 Ответ отправлен: {response}")

    except WebSocketDisconnect:
        logging.info("❌ Пользователь отключился.")

    except Exception as e:
        logging.error(f"❌ Ошибка в WebSocket: {e}")
        await websocket.close(code=1001)

# ========================== #
# 🚀 ЗАПУСК СЕРВЕРА 🚀 #
# ========================== #

if __name__ == "__main__":
    import uvicorn
    logging.info("🔥 Запуск FastAPI сервера...")
    uvicorn.run(app, host="0.0.0.0", port=8080)
