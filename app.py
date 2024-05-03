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

logging.basicConfig(filename='./app.log', level=logging.INFO, encoding='utf-8',
                    format='%(asctime)19s %(levelname)s: %(message)s')



class CommonData:
    '''Common dataclass with general variables'''
    api_keys: dict = json.loads(open('./api_keys.json').read())
    context_dict: dict = json.loads(open('./prompts.json','r', encoding="utf-8").read())
    template_prompts: dict = {
            'Цитата': 'Напиши остроумную цитату. Цитата может принадлежать как реально существующей или существовавшей личности, так и вымышленного персонажа',
            'Шутка': 'Выступи в роли профессионального стендап комика и напиши остроумную шутку. Ответом должен быть только текст шутки',
            'Факт': 'Выступи в роли профессионального энциклопедиста и напиши один занимательный факт. Ответом должен быть только текст с фактом',
            'Квиз': '''Выступи в роли профессионального энциклопедиста и напиши три вопроса для занимательного квиза. 
                        Уровень вопросов: Старшая школа. Ответом должен быть только текст с тремя вопросами без ответов'''
        }
    buttons: dict = {
            'Добавить контекст':'change_context', 
            'Быстрые команды':'template_prompts',
            'Вывести инфо':'show_info',
            'Сменить бота':'change_bot', 
            'Очистить контекст':'clear',
            'Изменить модель бота':'change_model'
        }
    PARSE_MODE = ParseMode.MARKDOWN_V2
    builder = ReplyKeyboardBuilder()
    for display_text in buttons:
        builder.button(text=display_text)
    builder = builder.adjust(3,3).as_markup()
    # "Дополнительные промпты":"https://vaulted-polonium-23c.notion.site/500-Best-ChatGPT-Prompts-63ef8a04a63c476ba306e1ec9a9b91c0"
    def singleton(cls):
        '''@singleton decorator'''
        instances = {}
        def wrapper(*args, **kwargs):
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
            return instances[cls]
        return wrapper



class CallbackClass(CallbackData, prefix='callback'):
    cb_type: str
    name: str



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
    """Class for Gemini API"""
    def __init__(self):
        self.name = 'gemini'
        genai.configure(api_key=CommonData.api_keys["gemini"])
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
    """Class for Cohere API"""
    def __init__(self):
        self.name = 'cohere'
        self.co = cohere.Client(CommonData.api_keys["cohere"])
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
    """Class for Groq API"""
    def __init__(self):
        self.name = 'groq'
        self.client = Groq(api_key=CommonData.api_keys["groq"])
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
    """Class for Nvidia API"""
    def __init__(self):
        self.name = 'nvidia'
        self.client = OpenAI(api_key=CommonData.api_keys["nvidia"],
                             base_url = "https://integrate.api.nvidia.com/v1")
        self.models = ['meta/llama3-70b-instruct',
                       'meta/llama3-8b-instruct',
                       'mistralai/mixtral-8x22b-instruct-v0.1',
                       'mistralai/mistral-large',
                       'google/recurrentgemma-2b',
                       'google/gemma-7b',
                       'microsoft/phi-3-mini-128k-instruct',
                       'snowflake/arctic',
                       'databricks/dbrx-instruct',
                       ] # https://build.nvidia.com/explore/discover#llama3-70b
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
        self.bot_names = ['nvidia','cohere'] # groq, gemini
        self.current = {'nvidia': NvidiaAPI,'cohere':CohereAPI}.get(self.bot_names[0])()
        self.time_dump = time()
        self.text = None
        

    async def change_context(self, context_name: str) -> str:
        self.clear()
        context = CommonData.context_dict.get(context_name)
        if isinstance(self.current, GeminiAPI):
            body = {'role':'system', 'parts':[context]}
        elif isinstance(self.current, CohereAPI):
            body = {"role": 'SYSTEM', "message": context}
        elif isinstance(self.current, (GroqAPI,NvidiaAPI)):
            body = {'role':'system', 'content': context}

        self.current.context.append(body)
        return f'Контекст {context_name} добавлен'


    async def template_prompts(self, template: str) -> str:
        prompt_text = CommonData.template_prompts.get(template)
        output = await self.prompt(prompt_text)
        return output
    

    def show_info(self) -> str:
        return f'\n* Текущий бот: {self.current}\n* Модель: {self.current.current_model}\n* Размер контекста: {len(self.current.context)}'
    

    def show_prompts_list(self) -> str:
        all_list = "\n".join([f"{k} - {v[-1]}" for k,v in self.prompts_dict.items() if k != "Дополнительные промпты"])
        return f'{all_list}\n[Дополнительные промпты]({self.prompts_dict["Дополнительные промпты"]})'


    async def change_bot(self, bot_name: str) -> str:
        self.current = {'nvidia': NvidiaAPI,'cohere':CohereAPI}.get(bot_name)()
        self.clear()
        return f'Смена бота на {self.current}'
    

    async def change_model(self, model_name: str) -> str:
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


    async def prompt(self, text: str, image=None):
        output = await self.current.prompt(text, image)
        return output



class UsersMap():
    def __init__(self) -> Agent:
        self._user_ins = {}
    
    def get(self, user_id: int):
        if user_id not in self._user_ins:
            self._user_ins[user_id] = Agent()
        return self._user_ins[user_id]
    

bot = Bot(token=CommonData.api_keys["telegram"], parse_mode=ParseMode.HTML)
db = DBConnection()
dp = Dispatcher()
users = UsersMap()


async def check_and_clear(message: types.Message, type_prompt: str) -> Agent | None:
    agent = users.get(message.from_user.id)
    if type_prompt == 'callback':
        return agent
    ## clear context after 15 minutes
    if (time() - agent.time_dump) > 900:
        agent.clear()
    agent.time_dump = time()
    user_name = db.check_user(message.from_user.id)
    type_prompt = {'text':message.text, 'photo': message.caption}.get(type_prompt)
    logging.info(f'{user_name[1] if user_name else message.from_user.id}: "{type_prompt}"')
    agent.text = CommonData.buttons.get(type_prompt, type_prompt)
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
    await message.answer(text, reply_markup=CommonData.builder)
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
    # await message.answer(output, reply_markup=builder, parse_mode=ParseMode.MARKDOWN_V2)


@dp.message(F.content_type.in_({'text'}))
async def echo_handler(message: types.Message | types.KeyboardButtonPollType):
    agent = await check_and_clear(message, 'text')
    if agent.text is None:
        return
    try:
        if agent.text in CommonData.buttons.values():
            if 'change_' in agent.text or 'template_prompts' == agent.text:
                await make_bth_cb(agent, message)
                return
            else:
                output = escape(getattr(agent, agent.text)())
        else:
            await message.reply('Ожидайте...')
            output = await agent.prompt(agent.text)
        await message.answer(output, CommonData.PARSE_MODE, reply_markup=CommonData.builder)
        return
    except Exception as e:
        logging.info(e)
        await message.answer(str(e))
        return


@dp.callback_query(CallbackClass.filter(F.cb_type.contains('change')|F.cb_type.contains('template')))
async def change_callback_handler(query: types.CallbackQuery, callback_data: CallbackClass):
    agent = await check_and_clear(query, 'callback')
    if callback_data.cb_type == 'template_prompts':
        await query.message.reply('Ожидайте...')
    output = await getattr(agent, callback_data.cb_type)(callback_data.name)
    await query.message.answer(output)
    await query.answer()

    
async def make_bth_cb(agent: Agent, message: types.Message) -> None: 
    '''
    Creates callback data with ChangeCallback, 
    generates and sends an inline keyboard as reply markup 
    '''
    builder_inline = InlineKeyboardBuilder()

    command_dict = {'bot': [agent.bot_names,'бота'],
                    'model': [agent.current.models,'модель'],
                    'context':[CommonData.context_dict, 'контекст'],
                    'prompts':[CommonData.template_prompts,'промпт']}
    items = command_dict.get(agent.text.split('_')[-1])
    for value in items[0]:
        data = CallbackClass(cb_type=agent.text, name=value).pack()
        builder_inline.button(text=value, callback_data=data)

    await message.answer(f'Выберите {items[-1]}:', CommonData.PARSE_MODE, 
                         reply_markup=builder_inline.adjust(*[1]*len(items)).as_markup())


async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    print('Starting')
    asyncio.run(main())
