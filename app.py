import asyncio
import logging
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import cohere
from groq import Groq
from openai import OpenAI
import json
import re
import sqlite3
import atexit
from time import time
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command, CommandStart
from aiogram.filters.callback_data import CallbackData
from aiogram.enums import ParseMode
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder
from md2tgmd import escape
# from PIL import Image
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

    def add_user(self, user_id: int, name: str) -> None:
        """
        Insert a new user into the table.
        :param user_id: The user's ID.
        :param name: The user's name.
        """
        query = 'INSERT INTO users (id, name) VALUES (?, ?)'
        self.cursor.execute(query, (user_id, name))
        self.commit()

    def remove_user(self, name: str) -> None:
        """
        Remove a user from the users table.
        :param name: The user's name.
        """
        query = 'DELETE FROM users WHERE name = ?'
        self.cursor.execute(query, (name,))
        self.commit()
    
    def check_user(self, user_id: int) -> tuple | None:
        return self.fetchone(f'SELECT * FROM users WHERE id is {user_id}')

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.close()



class GeminiAPI:
    """Singleton class for Gemini API"""
    # _instance = None

    # def __new__(cls, *args, **kwargs):
    #     if cls._instance is None:
    #         cls._instance = super(GeminiAPI, cls).__new__(cls)
    #     return cls._instance
    
    def __init__(self):
        self.name = 'gemini'
        genai.configure(api_key=api_keys["gemini"])
        self.safety_settings = { 
            'safety_settings':{
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
            }
        }
        self.models = ['gemini-pro', 'gemini-pro-vision']
        self.current_model = self.models[0]
        self.context = []

    def __str__(self):
        return self.name

    async def prompt(self, text: str, image = None) -> str:
        self.context.append({'role':'user', 'parts':[text]})
        if image is None:
            self.model = genai.GenerativeModel(self.models[0], **self.safety_settings)
            response = self.model.generate_content(self.context)
        else:
            self.model = genai.GenerativeModel(self.models[1], **self.safety_settings)
            response = self.model.generate_content([text,image])

        self.context.append({'role':'model', 'parts':[response.text]})

        print(response.text)
        return escape(response.text)



class CohereAPI:
    """Singleton class for Cohere API"""
    # _instance = None

    # def __new__(cls, *args, **kwargs):
    #     if cls._instance is None:
    #         cls._instance = super(CohereAPI, cls).__new__(cls)
    #     return cls._instance
    

    def __init__(self):
        self.name = 'cohere'
        self.co = cohere.Client(api_keys["cohere"])
        self.models = ['command-r-plus','command-r','command','command-light']
        self.current_model = self.models[0]
        self.context = []


    def __str__(self):
        return self.name
    

    async def prompt(self, text, image = None) -> str:
        response = self.co.chat(
            model=self.current_model,
            chat_history=self.context or None,
            message=text
        )
        self.context = response.chat_history
        print(response.text)
        return escape(response.text)



class GroqAPI:
    """Singleton class for Groq API"""
    # _instance = None

    # def __new__(cls, *args, **kwargs):
    #     if cls._instance is None:
    #         cls._instance = super(GroqAPI, cls).__new__(cls)
    #     return cls._instance
    
    def __init__(self):
        self.name = 'groq'
        self.client = Groq(api_key=api_keys["groq"])
        self.models = ['llama3-70b-8192','llama3-8b-8192','mixtral-8x7b-32768','gemma-7b-it'] # https://console.groq.com/docs/models
        self.current_model = self.models[0]
        self.context = []

    def __str__(self):
        return self.name

    async def prompt(self, text, image = None) -> str:
        if image is None:
            body = {'role':'user', 'content': text}
        else:
            body = Agent.make_multimodal_body(text, image)
        self.context.append(body)
        response = self.client.chat.completions.create(
            model=self.current_model,
            messages=self.context,
        )
        self.context.append({'role':'assistant', 'content':response.choices[-1].message.content})
        print(response.choices[-1].message.content)
        return escape(response.choices[-1].message.content)



class NvidiaAPI:
    """Singleton class for Nvidia API"""
    # _instance = None

    # def __new__(cls, *args, **kwargs):
    #     if cls._instance is None:
    #         cls._instance = super(NvidiaAPI, cls).__new__(cls)
    #     return cls._instance
    
    def __init__(self):
        self.name = 'nvidia'
        self.client = OpenAI(api_key=api_keys["nvidia"],
                             base_url = "https://integrate.api.nvidia.com/v1")
        self.models = ['meta/llama3-70b-instruct',
                       'snowflake/arctic',
                       'microsoft/phi-3-mini-128k-instruct',
                       'mistralai/mixtral-8x22b-instruct-v0.1',
                       'meta/llama3-8b-instruct',
                       'google/recurrentgemma-2b',
                       'mistralai/mistral-large'] # https://build.nvidia.com/explore/discover#llama3-70b
        self.current_model = self.models[0]
        self.context = []


    def __str__(self):
        return self.name
    

    async def prompt(self, text, image = None) -> str:
        if image is None:
            body = {'role':'user', 'content': text}
        else:
            body = Agent.make_multimodal_body(text, image)
        self.context.append(body)
        response = self.client.chat.completions.create(
            model=self.current_model,
            messages=self.context,
            temperature=0.5,
            top_p=1,
            max_tokens=1024
        )
        self.context.append({'role':'assistant', 'content':response.choices[-1].message.content})
        print(response.choices[-1].message.content)
        return escape(response.choices[-1].message.content)



class Agent:
    ''' Router for agents'''
    def __init__(self):
        self.api_classes = ['nvidia','cohere'] # [NvidiaAPI, CohereAPI] # GroqAPI, GeminiAPI
        self.current = {'nvidia': NvidiaAPI,'cohere':CohereAPI}.get(self.api_classes[0])() # self.api_classes[0]()
        self.prompts_dict: dict = json.loads(open('./prompts.json','r', encoding="utf-8").read())
        self.time_dump = time()
        self.text = None
        

    def add_context(self) -> str:
        content = self.prompts_dict.get(self.text)
        if content is None:
            return 'Код не найден'

        if isinstance(self.current, GeminiAPI):
            body = {'role':'system', 'parts':[content[0]]}
        elif isinstance(self.current, CohereAPI):
            body = {"role": 'SYSTEM', "message": content[0]}
        elif isinstance(self.current, (GroqAPI,NvidiaAPI)):
            body = {'role':'system', 'content': content[0]}
        
        self.current.context.append(body)
        return f'Контекст {content[-1]} добавлен'

        
    def show_info(self) -> str:
        return f'\n* Текущий бот: {self.current}\n* Модель: {self.current.current_model}\n* Размер контекста: {len(self.current.context)}'
    

    def show_prompts_list(self) -> str:
        all_list = "\n".join([f"{k} - {v[-1]}" for k,v in self.prompts_dict.items() if k != "Дополнительные промпты"])
        return f'{all_list}\n[Дополнительные промпты]({self.prompts_dict["Дополнительные промпты"]})'


    def change_bot(self, bot_name) -> str:
        # lst = self.api_classes
        # current_index = next((i for i, cls in enumerate(lst) if isinstance(self.current, cls)), None)
        # next_index = (current_index + 1) % len(lst)
        # self.current = lst[next_index]()
        self.current = {'nvidia': NvidiaAPI,'cohere':CohereAPI}.get(bot_name)()
        self.clear()
        return f'Смена бота на {self.current}'
    

    def change_model(self, model_name) -> str:
        # lst: list = self.current.models
        # current_index: int = lst.index(self.current.current_model)
        # next_index = (current_index + 1) % len(lst)
        # self.current.current_model = lst[next_index]
        self.current.current_model = model_name
        self.clear()
        output = re.split(r'-|/',model_name, maxsplit=1)[-1]
        return f'Смена модели на {output}'


    def clear(self) -> str:
        self.current.context = []
        return 'Контекст диалога отчищен'
    

    def make_multimodal_body(text, image):
        '''DEPRECATED'''
        return {
            'role':'user', 
            'content':[
                        {"type": "text", "text": text},
                        {"type": "image_url","image_url": {
                                                            "url": f"data:image/jpeg;base64,{image}"
                                                            }
                        }
                    ]
                }


    async def prompt(self, text, image=None):
        output = await self.current.prompt(text, image)
        return output



class UsersMap():
    def __init__(self) -> Agent:
        self._user_ins = {}
    
    def get(self, user_id: int):
        if user_id not in self._user_ins:
            self._user_ins[user_id] = Agent()
        return self._user_ins[user_id]
    

bot = Bot(token=api_keys["telegram"], parse_mode=ParseMode.HTML)
db = DBConnection()
# gemini_ins = GeminiAPI()
# cohere_ins = CohereAPI()
# groq_ins = GroqAPI()
dp = Dispatcher()
users = UsersMap()
buttons = {'Список кодов промптов':'show_prompts_list', 
            'Вывести инфо':'show_info',
            'Сменить бота':'change_bot', 
            'Очистить контекст':'clear',
            'Изменить модель бота':'change_model'}

builder = ReplyKeyboardBuilder()
for display_text in buttons:
    builder.button(text=display_text)
builder.adjust(2,3)

class MyCallback(CallbackData, prefix="my"):
    change_type: str
    name: str


async def check_and_clear(message: types.Message, type_prompt: str) -> Agent | None:
    agent = users.get(message.from_user.id)
    if type_prompt == 'callback':
        return agent
    ## clear context after 15 minutes
    if (time() - agent.time_dump) > 900:
        agent.clear()
    agent.time_dump = time()
    user_name = db.check_user(message.from_user.id)
    type_prompt = {'text':message.text, 'photo': message.caption,'callback': None}.get(type_prompt)
    logging.info(f'{user_name[1] if user_name else message.from_user.id}: "{type_prompt}"')
    agent.text = buttons.get(type_prompt, type_prompt)
    if user_name is None:
        await message.reply(f'Доступ запрещен. Обратитесь к администратору. Ваш id: {message.from_user.id}')
        return None
    
    return agent
    

@dp.message(CommandStart())
async def start_handler(message: types.Message):
    user_name = db.check_user(message.from_user.id)
    logging.info(f'{user_name[1] if user_name else message.from_user.id}: "/start"')
    if user_name:
        text = f'Доступ открыт. Добро пожаловать {message.from_user.full_name}!'
    else:
        text = f'Доступ запрещен. Обратитесь к администратору. Ваш id: {message.from_user.id}'
    await message.reply(text)
    return


@dp.message(Command(commands=["add"]))
async def add_handler(message: types.Message):
    user_name = db.check_user(message.from_user.id)
    if not user_name or user_name[1] != 'ADMIN':
        await message.reply("You don't have admin privileges")
        return
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.reply("Usage: `/add 123456 YourName`")
        return
    # Split the argument into user_id and name
    user_id, name = args[1].split(maxsplit=1)
    try:
        db.add_user(int(user_id), name)
        await message.reply(f"User {name} with ID {user_id} added successfully.")
    except sqlite3.IntegrityError:
        await message.reply("This user ID already exists.")
    except Exception as e:
        await message.reply(f"An error occurred: {e}.")


@dp.message(Command(commands=["remove"]))
async def remove_handler(message: types.Message):
    user_name = db.check_user(message.from_user.id)
    if not user_name or user_name[1] != 'ADMIN':
        await message.reply("You don't have admin privileges")
        return
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.reply("Usage: `/remove TargetName`")
        return
    # Take the argument as the name
    name_to_remove = args[1].strip()
    try:
        if user_name:
            db.remove_user(name_to_remove)
            await message.reply(f"User `{name_to_remove}` removed successfully.")
        else:
            await message.reply(f"User `{name_to_remove}` not found.")
    except Exception as e:
        await message.reply(f"An error occurred: {e}.")


@dp.callback_query(MyCallback.filter(F.change_type.contains('change')))
async def callback_handler(query: types.CallbackQuery, callback_data: MyCallback):
    agent = await check_and_clear(query, 'callback')
    output = getattr(agent, callback_data.change_type)(callback_data.name)
    await query.message.answer(output)
    await query.answer()


@dp.message(F.content_type.in_({'photo'}))
async def photo_handler(message: types.Message | types.KeyboardButtonPollType):
    agent = await check_and_clear(message, 'photo')
    if agent.text is None:
        return
    
    # await message.reply("Изображение получено! Ожидайте......")
    await message.reply("Обработка изображений временно недоступна")
    return
    # tg_photo = await bot.download(message.photo[-1].file_id)
    # output = await agent.prompt(agent.text, Image.open(tg_photo))
    # output = await agent.prompt(agent.text, base64.b64encode(tg_photo.getvalue()).decode('utf-8'))
    # await message.answer(output, reply_markup=builder.as_markup(), parse_mode=ParseMode.MARKDOWN_V2)


@dp.message(F.content_type.in_({'text'}))
async def echo_handler(message: types.Message | types.KeyboardButtonPollType):
    agent = await check_and_clear(message, 'text')
    if agent.text is None:
        return
    try:
        if agent.text in buttons.values() or agent.text.isnumeric():
            if 'change' in agent.text:
                command_type = agent.text.split('_')[-1]
                builder_inline = InlineKeyboardBuilder()
                command_dict = {'bot': [agent.api_classes,'бота'],
                                'model': [agent.current.models,'модель']}
                for value in command_dict.get(command_type)[0]:
                    builder_inline.button(text=re.split(r'-|/',value, maxsplit=1)[-1], 
                                          callback_data=MyCallback(change_type=f'change_{command_type}',
                                                                    name=value).pack())
                await message.answer(f'Выберите {command_dict.get(command_type)[-1]}:', 
                                     reply_markup=builder_inline.as_markup(), parse_mode=ParseMode.MARKDOWN_V2)
                return
            else:
                output = escape(getattr(agent, agent.text, agent.add_context)())
        else:
            await message.reply('Ожидайте...')
            output = await agent.prompt(agent.text)
        await message.answer(output, reply_markup=builder.as_markup(), parse_mode=ParseMode.MARKDOWN_V2)
        return
    except Exception as e:
        logging.info(e)
        await message.answer(str(e))
        return
    


async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    print('Starting')
    asyncio.run(main())
