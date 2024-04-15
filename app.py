import asyncio
import logging
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import cohere
import json
import sqlite3
import atexit
from time import time
from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ParseMode
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from md2tgmd import escape
import warnings
warnings.simplefilter('ignore')

# python app.py
# git remote get-url origin # получить url для docker
# api_keys = json.loads(os.getenv('api_key_bard_knows'))
# self.buttons = {
#         'Цитата': 'Напиши остроумную цитату. Цитата может принадлежать как реально существующей или существовавшей личности, так и вымышленного персонажа',
#         'Шутка': 'Выступи в роли профессионального стендап комика и напиши остроумную шутку. Ответом должен быть только текст шутки',
#         'Факт': 'Выступи в роли профессионального энциклопедиста и напиши один занимательный факт. Ответом должен быть только текст с фактом',
#         'Квиз': '''Выступи в роли профессионального энциклопедиста и напиши три вопроса для занимательного квиза. 
#                     Уровень вопросов: Старшая школа. Ответом должен быть только текст с тремя вопросами без ответов'''
#     }
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
        self.model_name = 'gemini-pro'
        self.model = genai.GenerativeModel(self.model_name, safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
        })
        self.context = []

    async def prompt(self, text: str) -> str:
        self.context.append({'role':'user', 'parts':[text]})
        response = self.model.generate_content(self.context)
        self.context.append({'role':'model', 'parts':[response.text]})
        print(response.text)
        return escape(response.text)


class CohereAPI:
    """Singleton class for Gemini API"""
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(CohereAPI, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        self.co = cohere.Client(api_keys["cohere"])
        self.context = None

    async def prompt(self, text):
        response = self.co.chat(
            model='command-r-plus',
            chat_history=self.context,
            message=text
        )
        self.context = response.chat_history
        print(response.text)
        return escape(response.text)


class Agent:
    ''' Router for agents'''
    def __init__(self):
        self.current = 'gemini'
        self.buttons = {'Добавить в контекст универсальный промпт':'enrich', 
                        'Показать текущий агент и размер контекста':'show',
                        'Изменить агент':'change', 
                        'Очистить контекст':'clear'}
        self.universal_prompt = json.loads(open('./prompts.json').read())
        self.all_prompts = 'https://vaulted-polonium-23c.notion.site/500-Best-ChatGPT-Prompts-63ef8a04a63c476ba306e1ec9a9b91c0'
        self.time_dump = time()
        
    def enrich(self, option='0'):
        if self.current == 'gemini':
            gemini.context.extend([{'role':'user', 'parts':[self.universal_prompt[option][0]]},
                                   {'role':'model', 'parts':[self.universal_prompt[option][1]]}])
        elif self.current == 'cohere':
            cohere_ins.context = [
                {"role": "USER", "message": self.universal_prompt[option][0]},
                {"role": "CHATBOT", "message": self.universal_prompt[option][1]}
            ]

        
    def show(self) -> int:
        if self.current == 'gemini':
            length = 0 if not gemini.context else len(gemini.context)
        elif self.current == 'cohere':
            length = 0 if not cohere_ins.context else len(cohere_ins.context)
        return length

    def change(self):
        if self.current == 'gemini':
            self.current = 'cohere'
        elif self.current == 'cohere':
            self.current = 'gemini'

    def clear(self):
        cohere_ins.context = None
        gemini.context = []

    async def prompt(self, text):
        if self.current == 'gemini':
            output = await gemini.prompt(text)
        elif self.current == 'cohere':
            output = await cohere_ins.prompt(text)
        return output


db = DBConnection()
gemini = GeminiAPI()
cohere_ins = CohereAPI()
dp = Dispatcher()
agent = Agent()

builder = ReplyKeyboardBuilder()
for display_text in agent.buttons:
    builder.button(text=display_text)
builder.adjust(2, 2)


@dp.message(F.content_type.in_({'text'}))
async def echo_handler(message: types.Message | types.KeyboardButtonPollType):
    ## clear context after 15 minutes
    if (time() - agent.time_dump) > 900:
        agent.clear()
    agent.time_dump = time()
    user_name = db.check_user(message.from_user.id)
    logging.info(f'{user_name[1] if user_name else message.from_user.id}: "{message.text}"')
    text = agent.buttons.get(message.text, message.text)

    try:
        if user_name is None:
            print(f'Новый пользователь: {user_name[0]}')
            await message.reply('Доступ запрещен. Обратитесь к администратору')
            return
        if text == '/start':
            await message.reply(f'Доступ открыт. Добро пожаловать {user_name[1]}!')
            return
        if text in agent.buttons.values() or text in '1':
            match text:
                case 'enrich':
                    agent.enrich()
                    output = f'Универсальный контекст добавлен в {agent.current}. [Дополнительные промпты]({agent.all_prompts})'
                case '1':
                    agent.enrich('1')
                    output = 'Роль учителя английского языка начата.'
                case 'show':
                    output = f'Текущий агент: {agent.current}, размер контекста {agent.show()}'
                case 'change':
                    agent.change()
                    output = f'Смена агента на {agent.current}'
                case 'clear':
                    agent.clear()
                    output = 'Контекст диалога отчищен'
            output = escape(output)
            # await message.answer(output, reply_markup=builder.as_markup(), parse_mode=ParseMode.MARKDOWN_V2)
        else:
            await message.reply('Ожидайте...')
            output = await agent.prompt(text)
        await message.answer(output, reply_markup=builder.as_markup(), parse_mode=ParseMode.MARKDOWN_V2)
        return
    except Exception as e:
        await message.answer(str(e))
        return


async def main() -> None:
    bot = Bot(token=api_keys["telegram"], parse_mode=ParseMode.HTML)
    await dp.start_polling(bot)


if __name__ == "__main__":
    print('Starting')
    asyncio.run(main())
