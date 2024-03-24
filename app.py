import asyncio
import logging
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
import sqlite3
import atexit
from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ParseMode
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from md2tgmd import escape
import warnings
warnings.simplefilter('ignore')

# python app.py
# git remote get-url origin # получить url для docker
# api_keys = json.loads(os.getenv('api_key_bard_knows'))
api_keys = json.loads(open('./api_keys.json').read())
logging.basicConfig(filename='./app.log', level=logging.INFO, encoding='utf-8',
                    format='%(asctime)19s %(levelname)s: %(message)s')

class DBConnection:
    """Singleton class for SQLite3 database connection"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(DBConnection, cls).__new__(cls)
            cls._instance.conn = sqlite3.connect('db.sqlite3')
            cls._instance.cursor = cls._instance.conn.cursor()
            atexit.register(cls._instance.close)
        return cls._instance
    
    def __init__(self):
        if not self.check_table():
            self.init_table()

    def execute(self, query):
        self.cursor.execute(query)

    def fetchone(self, query) -> tuple | None:
        self.execute(query)
        return self.cursor.fetchone()
    
    def fetchall(self, query) -> tuple | None:
        self.execute(query)
        return self.cursor.fetchall()
    
    def check_table(self) -> int:
        return self.fetchone("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='users'")[0]
    
    def init_table(self) -> None:
        self.execute('''CREATE TABLE IF NOT EXISTS users
                        (id INT PRIMARY KEY, name TEXT)''')
        self.commit()
    
    def check_user(self, user_id: int) -> tuple | None:
        return self.fetchone(f'SELECT * FROM users WHERE id is {user_id}')

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()


class GeminiAPI:
    """Singleton class for Gemini API"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(GeminiAPI, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        genai.configure(api_key=api_keys["gemini"])
        self.model = genai.GenerativeModel('gemini-pro', safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
        })
        self.buttons = {
                'Цитата': 'Напиши остроумную цитату. Цитата может принадлежать как реально существующей или существовавшей личности, так и вымышленного персонажа',
                'Шутка': 'Выступи в роли профессионального стендап комика и напиши остроумную шутку. Ответом должен быть только текст шутки',
                'Факт': 'Выступи в роли профессионального энциклопедиста и напиши один занимательный факт. Ответом должен быть только текст с фактом',
                'Квиз': '''Выступи в роли профессионального энциклопедиста и напиши три вопроса для занимательного квиза. 
                            Уровень вопросов: Старшая школа. Ответом должен быть только текст с тремя вопросами без ответов'''
            }

    async def prompt(self, text: str) -> str:
        response = self.model.generate_content(text)
        print(response.text)
        return escape(response.text)


db = DBConnection()
gemini = GeminiAPI()
dp = Dispatcher()

builder = ReplyKeyboardBuilder()
for display_text in gemini.buttons:
    builder.button(text=display_text)
builder.adjust(2, 2)


@dp.message(F.content_type.in_({'text'}))
async def echo_handler(message: types.Message | types.KeyboardButtonPollType):
    # input_data = arr_api_data.get(message.text, 'random')
    user_name = db.check_user(message.from_user.id)
    logging.info(f'{user_name[1] if user_name else message.from_user.id}: "{message.text}"')
    text = gemini.buttons.get(message.text, message.text)

    if user_name is None:
        print(f'Новый пользователь: {user_name[0]}')
        await message.reply('Доступ запрещен. Обратитесь к администратору')
        return
    if text == '/start':
        await message.reply(f'Доступ открыт. Добро пожаловать {user_name[1]}!')
        return

    try:
        # output = await handler(input_data)
        await message.reply('Ожидайте...')
        output = await gemini.prompt(text)
        await message.answer(output, reply_markup=builder.as_markup(), parse_mode=ParseMode.MARKDOWN_V2)
    except Exception as e:
        await message.answer(str(e))
        return


async def main() -> None:
    bot = Bot(token=api_keys["telegram"], parse_mode=ParseMode.HTML)
    await dp.start_polling(bot)


if __name__ == "__main__":
    print('Starting')
    asyncio.run(main())
