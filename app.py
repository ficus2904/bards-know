import asyncio
import aiohttp
import logging
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import cohere
from groq import Groq
from openai import OpenAI
import json
import sqlite3
import atexit
from time import time
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command, CommandStart
from aiogram.filters.callback_data import CallbackData
from aiogram.enums import ParseMode
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder
from md2tgmd import escape
import base64
from PIL import Image, ImageOps
import io
import warnings
warnings.simplefilter('ignore')

# python app.py

logging.basicConfig(filename='./app.log', level=logging.INFO, encoding='utf-8',
                    format='%(asctime)19s %(levelname)s: %(message)s')


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

    # def execute(self, *args):
    #     self.cursor.execute(*args)

    def fetchone(self, *args) -> tuple | None:
        # self.execute(*args)
        self.cursor.execute(*args)
        return self.cursor.fetchone()
    
    def fetchall(self, *args) -> tuple | None:
        self.cursor.execute(*args)
        return self.cursor.fetchall()
    
    def check_table(self) -> int:
        query = "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='users'"
        return self.fetchone(query)[0]
    
    def init_table(self) -> None:
        # self.execute('''CREATE TABLE IF NOT EXISTS users
        #                 (id INT PRIMARY KEY, name TEXT)''')
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS users
                            (id INT PRIMARY KEY, name TEXT)''')
        self.conn.commit()

    def add_user(self, user_id: int, name: str) -> None:
        """
        Insert a new user into the table.
        :param user_id: The user's ID.
        :param name: The user's name.
        """
        query = 'INSERT INTO users (id, name) VALUES (?, ?)'
        self.cursor.execute(query, (user_id, name))
        self.conn.commit()

    def remove_user(self, name: str) -> None:
        """
        Remove a user from the users table.
        :param name: The user's name.
        """
        query = 'DELETE FROM users WHERE name = ?'
        self.cursor.execute(query, (name,))
        self.conn.commit()
    
    def check_user(self, user_id: int) -> str | None:
        answer = self.fetchone("SELECT name FROM users WHERE id = ? LIMIT 1", (user_id,))
        return answer[0] if answer else None

    # def commit(self):
    #     self.conn.commit()

    def close(self):
        self.conn.close()



class GeminiAPI:
    """Class for Gemini API"""
    name = 'gemini'
    def __init__(self):
        genai.configure(api_key=users.api_keys["gemini"])
        self.settings = { 
            'safety_settings': {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
            },
            'system_instruction': None
        }
        self.models = ['gemini-1.5-pro-latest','gemini-1.5-flash-latest', ] # gemini-pro gemini-pro-vision
        self.current_model = self.models[0]
        self.model = None
        self.context = []
        self.chat = None
        self.reset_chat()

    async def prompt(self, text: str, image = None) -> str:
        # self.context.append({'role':'user', 'parts':[text]})
        # if image is None:
        #     self.model = genai.GenerativeModel(self.models[0], **self.safety_settings)
        #     response = self.model.generate_content(self.context)
        # else:
        #     self.model = genai.GenerativeModel(self.models[1], **self.safety_settings)
        #     response = self.model.generate_content([text, Image.open(image)])


        # response = self.model.generate_content([self.context, Image.open(image) if image else None])
        if image is None:
            response = self.chat.send_message(text)
        else:
            response = self.chat.send_message([text, Image.open(image)])

        # self.context.append({'role':'model', 'parts':[response.text]})

        # print(len(self.chat.history))
        return response.text
    
    def reset_chat(self):
        self.model = genai.GenerativeModel(self.current_model, **self.settings)
        self.context = []
        self.chat = self.model.start_chat(history=self.context)



class CohereAPI:
    """Class for Cohere API"""
    name = 'cohere'
    def __init__(self):
        self.co = cohere.Client(users.api_keys["cohere"])
        self.models = ['command-r-plus','command-r','command','command-light','c4ai-aya-23']
        self.current_model = self.models[0]
        self.context = []
    

    async def prompt(self, text, image = None) -> str:
        response = self.co.chat(
            model=self.current_model,
            chat_history=self.context or None,
            message=text
        )
        self.context = response.chat_history
        # print(response.text)
        return response.text



class GroqAPI:
    """Class for Groq API"""
    name = 'groq'
    def __init__(self):
        self.client = Groq(api_key=users.api_keys["groq"])
        self.models = ['llama3-70b-8192',
                       'llama3-groq-70b-8192-tool-use-preview',
                       'whisper-large-v3'] # https://console.groq.com/docs/models
        self.current_model = self.models[0]
        self.context = []


    async def prompt(self, text, image = None) -> str:
        if image is None:
            body = {'role':'user', 'content': text}
        else:
            body = User.make_multimodal_body(text, image)
        self.context.append(body)
        response = self.client.chat.completions.create(
            model=self.current_model,
            messages=self.context,
        )
        self.context.append({'role':'assistant', 'content':response.choices[-1].message.content})
        # print(response.choices[-1].message.content)
        return response.choices[-1].message.content



class NvidiaAPI:
    """Class for Nvidia API"""
    name = 'nvidia'
    def __init__(self):
        self.client = OpenAI(api_key=users.api_keys["nvidia"],
                             base_url = "https://integrate.api.nvidia.com/v1")
        self.models = [
                        'meta/llama-3.1-405b-instruct',
                        'meta/llama-3.1-8b-instruct',
                        'nvidia/nemotron-4-340b-instruct',
                        'nvidia/llama3-chatqa-1.5-70b',
                        'nvidia/neva-22b',
                        'microsoft/kosmos-2',
                        'adept/fuyu-8b',
                        'microsoft/phi-3-vision-128k-instruct',
                       ] # https://build.nvidia.com/explore/discover
        self.vlm_params = {
                        'microsoft/phi-3-vision-128k-instruct': {
                            "max_tokens": 512,
                            "temperature": 1,
                            "top_p": 0.70,
                            "stream": False
                        },
                        'nvidia/neva-22b': {
                            "max_tokens": 1024,
                            "temperature": 0.20,
                            "top_p": 0.70,
                            "seed": 0,
                            "stream": False
                        },
                        'microsoft/kosmos-2': {
                            "max_tokens": 1024,
                            "temperature": 0.20,
                            "top_p": 0.2
                        },
                        'adept/fuyu-8b': {
                            "max_tokens": 1024,
                            "temperature": 0.20,
                            "top_p": 0.7,
                            "seed":0,
                            "stream": False
                        }
                    }
        self.current_model = self.models[0]
        self.current_vlm_model = self.models[-1]
        self.context = []
    

    async def prompt(self, text, image = None) -> str:
        if image is None and self.current_model not in self.vlm_params:
            body = {'role':'user', 'content': text}
            self.context.append(body)
            response = self.client.chat.completions.create(
                model=self.current_model,
                messages=self.context,
                temperature=0.2,
                top_p=0.7,
                max_tokens=1024
            )
            output = response.choices[-1].message.content
            self.context.append({'role':'assistant', 'content':output})
            # print(output)
            return output
        else:
            self.context.append({"role": "user","content": text})
            model = self.current_model if self.current_model in self.vlm_params else self.current_vlm_model
            if image:
                image_b64 = base64.b64encode(image.getvalue()).decode()
                if len(image_b64) > 180_000:
                    print("Слишком большое изображение, сжимаем...")
                    image_b64 = users.resize_image(image)
                image_b64 = f'. <img src="data:image/jpeg;base64,{image_b64}" />'
            else:
                image_b64 = ''

            body = {"messages": [{"role": "user","content": text + image_b64}]} | self.vlm_params.get(model)
            headers = {"Authorization": f"Bearer {users.api_keys["nvidia"]}",
                        "Accept": "application/json"}
            url = "https://ai.api.nvidia.com/v1/vlm/" + model
            async with aiohttp.ClientSession() as session:
                async with session.post(url=url,headers=headers,json=body) as response:
                    try:
                        response.raise_for_status()
                        output = await response.json()
                    except aiohttp.ClientResponseError as e:
                        logging.error(e)
                        return 'Error exception'
            # response = requests.post(url=url, headers=headers, json=body)
            # output = response.json()
            output = output.get('choices',[{}])[-1].get('message',{}).get('content','')
            self.context.append({'role':'assistant', 'content':output})
            return output
        


class TogetherAPI:
    """Class for Together API"""
    name = 'together'
    def __init__(self):
        self.client = OpenAI(api_key=users.api_keys["together"],
                             base_url="https://api.together.xyz/v1")
        self.models = [
                        'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
                        'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
                        'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
                       ] # https://docs.together.ai/docs/inference-models

        self.current_model = self.models[0]
        self.context = []
    

    async def prompt(self, text, image=None) -> str:
        body = {'role':'user', 'content': text}
        self.context.append(body)
        response = self.client.chat.completions.create(
            model=self.current_model,
            messages=self.context,
            temperature=0.7,
            top_p=0.7,
            max_tokens=1024
        )
        output = response.choices[-1].message.content
        self.context.append({'role':'assistant', 'content':output})
        # print(output)
        return output



class GlifAPI:
    """Class for Glif API"""
    name = 'glif'
    def __init__(self):
        self.api_key = users.api_keys["glif"]
        self.models_with_ids = {
                                "claude 3.5 sonnet":"clxwyy4pf0003jo5w0uddefhd",
                                "GPT4o":"clxx330wj000ipbq9rwh4hmp3",
                                "Llama 3.1 405B":"clyzjs4ht0000iwvdlacfm44y",
                                }
        self.models = list(self.models_with_ids.keys()) #['claude 3.5 sonnet', 'GPT4o','Llama 3.1 405B']
        self.current_model = self.models[0]
        self.context = []


    def form_main_prompt(self) -> str:
        if len(self.context) > 2:
            initial_text = 'Use next json schema as context of our previous dialog: '
            return initial_text + str(self.context[1:])
        else:
            return self.context[-1].get('content')
    

    def form_system_prompt(self) -> str:
        if not self.context:
            default_prompt = users.context_dict.get('Универсальный промпт')
            self.context.append({'role':'system', 'content': default_prompt})
        return self.context[0].get('content')
    

    async def fetch_data(self, main_prompt: str, system_prompt: str) -> str:
        url = "https://simple-api.glif.app"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        body = {"id": self.models_with_ids.get(self.current_model), 
                "inputs": {"main_prompt": main_prompt, 
                           "system_prompt": system_prompt}}
        async with aiohttp.ClientSession() as session:
            async with session.post(url=url,headers=headers,json=body) as response:
                try:
                    response.raise_for_status()
                    answer = await response.json()
                    return answer['output'] or 'Error main'
                except aiohttp.ClientResponseError as e:
                    logging.error(e)
                    return 'Error exception'
    

    async def prompt(self, text, image = None) -> str:
        system_prompt = self.form_system_prompt()
        self.context.append({"role": "user","content": text})
        main_prompt = self.form_main_prompt()
        output = await self.fetch_data(main_prompt, system_prompt)
        self.context.append({'role':'assistant', 'content': output})
        return output



class User:
    '''Specific user interface in chat'''
    def __init__(self):
        self.bots_lst: list = [NvidiaAPI, CohereAPI, GroqAPI, GeminiAPI, TogetherAPI, GlifAPI]
        self.bots: dict = {bot_class.name:bot_class for bot_class in self.bots_lst}
        self.current_bot = self.bots.get(users.DEFAULT_BOT)()
        self.time_dump = time()
        self.text = None
        

    async def change_context(self, context_name: str) -> str:
        self.clear()
        context = users.context_dict.get(context_name)
        if isinstance(self.current_bot, GeminiAPI):
            self.current_bot.settings['system_instruction'] = context
            self.current_bot.reset_chat()
            return f'Контекст {context_name} добавлен'
        
        elif isinstance(self.current_bot, CohereAPI):
            body = {"role": 'SYSTEM', "message": context}
        elif isinstance(self.current_bot, (GroqAPI,NvidiaAPI,TogetherAPI,GlifAPI)):
            body = {'role':'system', 'content': context}

        self.current_bot.context.append(body)
        return f'Контекст {context_name} добавлен'


    async def template_prompts(self, template: str) -> str:
        prompt_text = users.template_prompts.get(template)
        output = await self.prompt(prompt_text)
        return output
    

    def info(self) -> str:
        check_vlm = hasattr(self.current_bot, 'vlm_params')
        is_gemini = self.current_bot.name == 'gemini'
        if is_gemini:
            context = bool(self.current_bot.settings['system_instruction']) + len(self.current_bot.chat.history)
        return '\n'.join(['',
            f'* Текущий бот: {self.current_bot.name}',
            f'* Модель: {self.current_bot.current_model}',
            f'* Модель vlm: {self.current_bot.current_vlm_model}' if check_vlm else '',
            f'* Размер контекста: {len(self.current_bot.context) if not is_gemini else context}'])
    

    def show_prompts_list(self) -> str:
        all_list = "\n".join([f"{k} - {v[-1]}" for k,v in self.prompts_dict.items() if k != "Дополнительные промпты"])
        return f'{all_list}\n[Дополнительные промпты]({self.prompts_dict["Дополнительные промпты"]})'


    async def change_bot(self, bot_name: str) -> str:
        self.current_bot = self.bots.get(bot_name)()
        return f'Смена бота на {self.current_bot.name}'
    

    async def change_model(self, model_name: str) -> str:
        self.current_bot.current_model = model_name
        if hasattr(self.current_bot, 'vlm_params') and model_name in self.current_bot.vlm_params:
            self.current_bot.current_vlm_model = model_name
        self.clear()
        # output = re.split(r'-|/',model_name, maxsplit=1)[-1]
        return f'Смена модели на {users.make_short_name(model_name)}'


    def clear(self) -> str:
        self.current_bot.context.clear()
        if self.current_bot.name == 'gemini':
            self.current_bot.settings['system_instruction'] = None
            self.current_bot.reset_chat()
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
        output = await self.current_bot.prompt(text, image)
        print(output)
        return escape(output)



class UsersMap():
    '''Main storage of user's sessions, common variables and functions'''
    def __init__(self):
        self._user_ins: dict = {}
        self.api_keys: dict = json.loads(open('./api_keys.json').read())
        self.context_dict: dict = json.loads(open('./prompts.json','r', encoding="utf-8").read())
        self.template_prompts: dict = {
                'Цитата': 'Напиши остроумную цитату. Цитата может принадлежать как реально существующей или существовавшей личности, так и вымышленного персонажа',
                'Шутка': 'Выступи в роли профессионального стендап комика и напиши остроумную шутку. Ответом должен быть только текст шутки',
                'Факт': 'Выступи в роли профессионального энциклопедиста и напиши один занимательный факт. Ответом должен быть только текст с фактом',
                'Квиз': '''Выступи в роли профессионального энциклопедиста и напиши три вопроса для занимательного квиза. 
                            Уровень вопросов: Старшая школа. Ответом должен быть только текст с тремя вопросами без ответов'''
            }
        self.buttons: dict = {
                'Добавить контекст':'change_context', 
                'Быстрые команды':'template_prompts',
                'Вывести инфо':'info',
                'Сменить бота':'change_bot', 
                'Очистить контекст':'clear',
                'Изменить модель бота':'change_model'
            }
        self.PARSE_MODE = ParseMode.MARKDOWN_V2
        self.DEFAULT_BOT = 'glif'
        self.builder = self.create_builder()


    def create_builder(self) -> ReplyKeyboardBuilder:
        builder = ReplyKeyboardBuilder()
        for display_text in self.buttons:
            builder.button(text=display_text)
        return builder.adjust(3,3).as_markup()

    
    def get(self, user_id: int) -> User:
        if user_id not in self._user_ins:
            self._user_ins[user_id] = User()
        return self._user_ins[user_id]
    

    def resize_image(self, image: io.BytesIO, max_b64_length=180_000, max_file_size_kb=450):
        max_file_size_bytes = max_file_size_kb * 1024
        img = Image.open(image)
        # Функция для сжатия и конвертации изображения в Base64
        def image_to_base64(img, quality=85):
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            return img_b64, buffer.getvalue()
        
        # Рекурсивная функция для сжатия изображения
        def recursive_compress(img, quality):
            img_b64, img_bytes = image_to_base64(img, quality=quality)
            # Проверить размер изображения
            if len(img_b64) <= max_b64_length and len(img_bytes) <= max_file_size_bytes:
                return img_b64
            # Уменьшить размер изображения
            img = ImageOps.exif_transpose(img)
            img.thumbnail((img.size[0] * 0.9, img.size[1] * 0.9), Image.ADAPTIVE)
            # Уменьшить качество, если размер все еще превышает лимит
            quality = max(10, quality - 5)
            # Рекурсивный вызов для сжатия изображения с новыми параметрами
            return recursive_compress(img, quality)
        
        # Начальное сжатие
        return recursive_compress(img, quality=85)
    

    def make_short_name(self, text: str) -> str:
        if text.startswith('meta'):
            return text.split('/')[1].split('nstruct')[0][:-2]
        else:
            return text.split('/')[1] if '/' in text else text
        

    async def split_text(self, text: str, max_length=4096):
        start = 0
        while start < len(text):
            if len(text) - start <= max_length:
                yield text[start:]
                break
            
            split_index = start + max_length
            for separator in ('\n', ' '):
                last_separator = text.rfind(separator, start, split_index)
                if last_separator != -1:
                    split_index = last_separator
                    break
            
            yield text[start:split_index]
            start = split_index
            while start < len(text) and text[start] in ('\n', ' '):
                start += 1
            
            # Добавляем небольшую задержку, чтобы дать возможность другим задачам выполниться
            await asyncio.sleep(0)


    def set_kwargs(self, text, reply_markup = None) -> dict:
        return {'text': text, 
                'parse_mode':self.PARSE_MODE, 
                'reply_markup': reply_markup or self.builder}


    async def check_and_clear(self, message: types.Message, type_prompt: str) -> User | None:
        user_name = db.check_user(message.from_user.id)
        if user_name is None:
            await message.reply(f'Доступ запрещен. Обратитесь к администратору. Ваш id: {message.from_user.id}')
            return
        user = self.get(message.from_user.id)
        if type_prompt == 'callback':
            return user
        ## clear context after 30 minutes
        if (time() - user.time_dump) > 1800:
            user.clear()
        user.time_dump = time()
        type_prompt = {'text': message.text, 'photo': message.caption}.get(type_prompt, type_prompt)
        logging.info(f'{user_name or message.from_user.id}: "{type_prompt}"')
        user.text = self.buttons.get(type_prompt, type_prompt)
        return user




users = UsersMap()
bot = Bot(token=users.api_keys["telegram"])
db = DBConnection()
dp = Dispatcher()
    

@dp.message(CommandStart())
async def start_handler(message: types.Message):
    user_name = db.check_user(message.from_user.id)
    if user_name:
        text = f'Доступ открыт. Добро пожаловать {message.from_user.full_name}!'
    else:
        text = f'Доступ запрещен. Обратитесь к администратору. Ваш id: {message.from_user.id}'
    await message.answer(text)


@dp.message(Command(commands=["add_prompt"]))
async def add_prompt_handler(message: types.Message):
    user_name = db.check_user(message.from_user.id)
    if user_name != 'ADMIN':
        await message.reply("You don't have admin privileges")
        return
    args = message.text.split('|', maxsplit=2)
    if len(args) != 3 or '|' in args[-1]:
        await message.reply("Usage: `/add_prompt | prompt_name | prompt`")
        return
    # Split the argument into user_id and name
    _, prompt_name, prompt = [arg.strip() for arg in args]
    if users.context_dict.get(prompt_name):
        await message.reply(f"Prompt {prompt_name} already exists")
        return
    try:
        users.context_dict[prompt_name] = prompt
        with open('./prompts.json', 'w', encoding="utf-8") as f:
            json.dump(users.context_dict, f, ensure_ascii=False)
        await message.reply(f"Prompt {prompt_name} added successfully.")
    except Exception as e:
        await message.reply(f"An error occurred: {e}.")


@dp.message(Command(commands=["add"]))
async def add_handler(message: types.Message):
    user_name = db.check_user(message.from_user.id)
    if user_name != 'ADMIN':
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
    if user_name != 'ADMIN':
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


@dp.message(Command(commands=["info","clear"]))
async def short_command_handler(message: types.Message):
    user = await users.check_and_clear(message, message.text.lstrip('/'))
    if user is None:
        return
    kwargs = users.set_kwargs(escape(getattr(user, user.text)()))
    await message.answer(**kwargs)



@dp.message(lambda message: message.text in users.buttons)
async def clear_command(message: types.Message):
    user = await users.check_and_clear(message, 'text')
    if user is None:
        return
    if user.text in {'info','clear'}:
        kwargs = users.set_kwargs(escape(getattr(user, user.text)()))
    else:
        builder_inline = InlineKeyboardBuilder()
        command_dict = {'bot': [user.bots, 'бота'],
                        'model': [user.current_bot.models, 'модель'],
                        'context':[users.context_dict, 'контекст'],
                        'prompts':[users.template_prompts, 'промпт']}
        items = command_dict.get(user.text.split('_')[-1])
        for value in items[0]:
            data = CallbackClass(cb_type=user.text, name=value).pack()
            builder_inline.button(text=users.make_short_name(value), callback_data=data)
        kwargs = users.set_kwargs(f'Выберите {items[-1]}:', 
                                       builder_inline.adjust(*[1]*len(items)).as_markup())
    
    await message.answer(**kwargs)



@dp.message(F.content_type.in_({'photo'}))
async def photo_handler(message: types.Message | types.KeyboardButtonPollType):
    user = await users.check_and_clear(message, 'photo')
    if user is None:
        return
    if user.text is None:
        user.text = 'Следуй системным правилам'
        # return
    
    if user.current_bot.name not in ['nvidia', 'gemini']:
        text_reply = "Переключите бота на nvidia или gemini для обработки изображений"
        await message.reply(text_reply)
        return
    
    text_reply = "Изображение получено! Ожидайте..."
    if user.current_bot.name == 'nvidia' and user.current_bot.current_model not in user.current_bot.vlm_params:
        text_reply = f"Обработка изображения с использованием {user.current_bot.current_vlm_model}..."

    await message.reply(text_reply)
    tg_photo = await bot.download(message.photo[-1].file_id)
    output = await user.prompt(user.text, tg_photo)
    async for part in users.split_text(output):
        await message.answer(**users.set_kwargs(part))
    return


@dp.message(F.content_type.in_({'text'}))
async def echo_handler(message: types.Message | types.KeyboardButtonPollType):
    user = await users.check_and_clear(message, 'text')
    if user is None or user.text is None:
        return
    try:
        await message.reply('Ожидайте...')
        output = await user.prompt(user.text)
        async for part in users.split_text(output):
            await message.answer(**users.set_kwargs(part))
        return
    except Exception as e:
        logging.info(e)
        await message.answer("Error processing message. See logs for details")
        return


@dp.callback_query(CallbackClass.filter(F.cb_type.contains('change')|F.cb_type.contains('template')))
async def change_callback_handler(query: types.CallbackQuery, callback_data: CallbackClass):
    user = await users.check_and_clear(query, 'callback')
    if callback_data.cb_type == 'template_prompts':
        await query.message.reply('Ожидайте...')
    output = await getattr(user, callback_data.cb_type)(callback_data.name)
    # await query.message.answer(output,users.PARSE_MODE)
    await query.message.answer(**users.set_kwargs(output))
    await query.answer()



async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    print('Starting')
    asyncio.run(main())
