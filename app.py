import os
import io
import re
import json
import base64
import asyncio
import aiohttp
import logging
import warnings
import sqlite3
import cohere
import atexit
from mistralai import Mistral
import google.generativeai as genai
from abc import ABC, abstractmethod
from aiolimiter import AsyncLimiter
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from functools import lru_cache
from time import time
from groq import Groq
from openai import OpenAI
from aiogram import Bot, Dispatcher, BaseMiddleware, F
from aiogram.types import TelegramObject, Message, CallbackQuery, KeyboardButtonPollType
from aiogram.filters import Command, CommandStart
from aiogram.filters.callback_data import CallbackData
from aiogram.enums import ParseMode
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder
from md2tgmd import escape
from PIL import Image, ImageOps
from dotenv import load_dotenv
load_dotenv()
warnings.simplefilter('ignore')

# python app.py
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
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



class BaseAPIInterface(ABC):
    @classmethod  
    def __init_subclass__(cls, **kwargs):  
        super().__init_subclass__(**kwargs)  
        cls.api_key = cls.get_api_key(cls.name)
        cls.context = []

    @staticmethod  
    def get_api_key(name: str):  
        return os.getenv(f'{name.upper()}_API_KEY')


    @abstractmethod
    async def prompt(self, *args, **kwargs):
        pass



class GeminiAPI(BaseAPIInterface):
    """Class for Gemini API"""
    name = 'gemini'

    def __init__(self):
        self.settings = { 
            'safety_settings': {
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE, 
            },
            'system_instruction': None
        }
        self.models = ['gemini-1.5-pro-exp-0827',
                       'gemini-1.5-pro-002',
                       'gemini-1.5-flash-002', ] # gemini-1.5-pro-latest
        self.current_model = self.models[0]
        self.client = None
        # self.context = []
        self.chat = None
        self.reset_chat()

    async def prompt(self, text: str, image = None) -> str:
        if image is None:
            response = self.chat.send_message(text)
        else:
            response = self.chat.send_message([Image.open(image), text])
        return response.text
    
    def reset_chat(self):
        self.client = genai.GenerativeModel(self.current_model, **self.settings)
        self.context = []
        self.chat = self.client.start_chat(history=self.context)


    async def get_enhanced_prompt(self, init_prompt: str) -> str:
        # self.settings['system_instruction'] = users.context_dict[''].get('SDXL')
        self.settings['system_instruction'] = users.get_context('SDXL')
        self.reset_chat()
        enhanced_prompt = await self.prompt(init_prompt)
        return enhanced_prompt



class CohereAPI(BaseAPIInterface):
    """Class for Cohere API"""
    name = 'cohere'

    def __init__(self):
        self.client = cohere.Client(self.api_key)
        self.models = ['command-r-plus-08-2024','command-nightly','c4ai-aya-23-35b']
        self.current_model = self.models[0]
        self.context = []
    

    async def prompt(self, text, image = None) -> str:
        response = self.client.chat(
            model=self.current_model,
            chat_history=self.context or None,
            message=text,
            safety_mode='NONE'
        )
        self.context = response.chat_history
        # print(response.text)
        return response.text



class GroqAPI(BaseAPIInterface):
    """Class for Groq API"""
    name = 'groq'

    def __init__(self):
        self.client = Groq(api_key=self.api_key)
        self.models = ['llama-3.2-11b-vision-preview',
                       'llama-3.1-70b-versatile',
                       'llama3-70b-8192',
                       'llama3-groq-70b-8192-tool-use-preview',
                       'llava-v1.5-7b-4096-preview'] # https://console.groq.com/docs/models
        self.current_model = self.models[0]


    async def prompt(self, text: str, image = None) -> str:
        if image:
            self.context.clear()
            User.make_multi_modal_body(text or "What's in this image?", image, self.context)
        else:
            body = {'role':'user', 'content': text}
            self.context.append(body)
        
        kwargs = {'model':self.current_model, # self.models[-1] if image else , 
                  'messages': self.context}
        response = self.client.chat.completions.create(**kwargs).choices[-1].message.content
        self.context.append({'role':'assistant', 'content':response})
        return response



class MistralAPI(BaseAPIInterface):
    """Class for Mistral API"""
    name = 'mistral'

    def __init__(self):
        self.client = Mistral(api_key=self.api_key)
        self.models = ['mistral-small-latest',
                       'mistral-large-latest',
                       'pixtral-12b-2409'] # https://docs.mistral.ai/getting-started/models/
        self.current_model = self.models[0]


    async def prompt(self, text: str, image = None) -> str:
        if image:
            User.make_multi_modal_body(text or "What's in this image?", 
                                        image, self.context, is_mistral=True)
        else:
            body = {'role':'user', 'content': text}
            self.context.append(body)
        
        kwargs = {'model':self.models[-1] if image else self.current_model, 
                  'messages': self.context}
        response = await self.client.chat.complete_async(**kwargs)
        response = response.choices[-1].message.content
        self.context.append({'role':'assistant', 'content':response})
        return response



class NvidiaAPI(BaseAPIInterface):
    """Class for Nvidia API"""
    name = 'nvidia'
    
    def __init__(self):
        self.client = OpenAI(api_key=self.api_key,
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
                        'nvidia/vila',
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
                        },
                        'nvidia/vila': {
                            "max_tokens": 1024,
                            "temperature": 0.20,
                            "top_p": 0.7,
                            "stream": False
                        }
                    }
        self.current_model = self.models[0]
        self.current_vlm_model = self.models[-1]
        # self.context = []
    

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
                image_b64 = f'Hi! What is in this image? <img src="data:image/jpeg;base64,{image_b64}" />'
            else:
                image_b64 = ''

            body = {"messages": [{"role": "user","content": text + image_b64}]} | self.vlm_params.get(model)
            headers = {"Authorization": f"Bearer {self.api_key}",
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
        


class TogetherAPI(BaseAPIInterface):
    """Class for Together API"""
    name = 'together'
    
    def __init__(self):
        self.client = OpenAI(api_key=self.api_key,
                             base_url="https://api.together.xyz/v1")
        self.models = [
                        'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo',
                        'meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo',
                        'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
                       ] # https://docs.together.ai/docs/inference-models

        self.current_model = self.models[0]
        # self.context = []
    

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



class GlifAPI(BaseAPIInterface):
    """Class for Glif API"""
    name = 'glif'

    def __init__(self):
        self.models_with_ids = {
                                "claude 3.5 sonnet":"clxwyy4pf0003jo5w0uddefhd",
                                "GPT4o":"clxx330wj000ipbq9rwh4hmp3",
                                "Llama 3.1 405B":"clyzjs4ht0000iwvdlacfm44y",
                                }
        self.models = list(self.models_with_ids.keys()) #['claude 3.5 sonnet', 'GPT4o','Llama 3.1 405B']
        self.current_model = self.models[0]
        # self.context = []


    def form_main_prompt(self) -> str:
        if len(self.context) > 2:
            initial_text = 'Use next json schema as context of our previous dialog: '
            return initial_text + str(self.context[1:])
        else:
            return self.context[-1].get('content')
    

    def form_system_prompt(self) -> str:
        if not self.context:
            default_prompt = users.context_dict.get('Универсальный')
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
                


    async def fetch_image_glif(self, prompt: str) -> dict:
        url = "https://simple-api.glif.app"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        body = {"id": {
                        True:'clzmbpo6k000u1pb2ar3udjff',
                        False:'clzj1yoqc000i13n0li4mwa2b'
                        }.get(prompt.startswith('-f')), 
                "inputs": {"initial_prompt": prompt.lstrip('-f ')}}
        async with aiohttp.ClientSession() as session:
            async with session.post(url=url,headers=headers,json=body, timeout=90) as response:
                try:
                    response.raise_for_status()
                    answer = await response.json()
                    try:
                        return json.loads(answer['output'])
                    except Exception:
                        match = re.search(r'https://[^"]+\.jpg', answer['output'])
                        return {"photo":match.group(0) if match else None,
                                "caption":answer['output'].split('"caption":"')[-1].rstrip('"}')}
                except Exception as e:
                    match e:
                        case asyncio.TimeoutError():
                            logging.error(error_msg := 'Timeout error')
                        case aiohttp.ClientResponseError():
                            logging.error(error_msg := f'HTTP error {e.status}: {e.message}')
                        case KeyError():
                            logging.error(error_msg := 'No output data')
                        case _:
                            logging.error(error_msg := f'Unexpected error: {str(e)}')
                    return {'error': error_msg}
                

    async def fetch_image_fal(self, prompt: str) -> dict:
        url = "https://fal.run/fal-ai/flux-pro/v1.1" #"https://fal.run/fal-ai/flux-pro/new"
        headers = {"Authorization": f"Key {os.getenv('FAL_API_KEY')}",
                   'Content-Type': 'application/json'}
        body = {"prompt": prompt,
                "image_size": "portrait_4_3", # Possible values: "square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"
                "num_inference_steps": 30,
                "guidance_scale": 3.5,
                "num_images": 1,
                "enable_safety_checker": False,
                "safety_tolerance": "5"}
        async with aiohttp.ClientSession() as session:
            async with session.post(url=url,headers=headers,json=body, timeout=90) as response:
                try:
                    response.raise_for_status()
                    answer = await response.json()
                    try:
                        return answer['images'][0]['url']
                    except Exception:
                        return str(answer)
                except Exception as e:
                    match e:
                        case asyncio.TimeoutError():
                            logging.error(error_msg := 'Timeout error')
                        case aiohttp.ClientResponseError():
                            logging.error(error_msg := f'HTTP error {e.status}: {e.message}')
                        case KeyError():
                            logging.error(error_msg := 'No output data')
                        case _:
                            logging.error(error_msg := f'Unexpected error: {str(e)}')
                    return {'error': error_msg}
                

    async def prompt(self, text, image = None) -> str:
        system_prompt = self.form_system_prompt()
        self.context.append({"role": "user","content": text})
        main_prompt = self.form_main_prompt()
        output = await self.fetch_data(main_prompt, system_prompt)
        self.context.append({'role':'assistant', 'content': output})
        return output



class APIFactory:
    '''A factory pattern for creating bot interfaces'''
    bots_lst: list = [NvidiaAPI, CohereAPI, GroqAPI, GeminiAPI, TogetherAPI, GlifAPI, MistralAPI]
    bots: dict = {bot_class.name:bot_class for bot_class in bots_lst}
    def __init__(self):
        self._instances: dict[str,BaseAPIInterface] = {}


    def get(self, bot_name: str) -> BaseAPIInterface:
        return self._instances.setdefault(bot_name, self.bots[bot_name]())



class RateLimitedQueueManager:
    def __init__(self):
        self.limiters = {name:AsyncLimiter(1, 30) for name in APIFactory.bots}
    
    async def enqueue_request(self, api_name: str, task):
        limiter = self.limiters[api_name]
        async with limiter:
            return await task



class User:
    '''Specific user interface in chat'''
    def __init__(self):
        self.api_factory = APIFactory()
        self.current_bot: BaseAPIInterface = self.api_factory.get(users.DEFAULT_BOT)
        self.time_dump = time()
        self.text = None
        

    async def change_context(self, context_name: str) -> str | dict:
        self.clear()
        if context_name == '◀️':
            return users.context_dict
        
        # context = users.context_dict.get(context_name, 
        #                                  users.get_subcontext(context_name))
        context = users.get_context(context_name)

        if isinstance(context, dict): # subgroup
            context.setdefault('◀️','◀️')
            return context
        # elif context is None: # final set in subgroup
        #     context = users.get_subcontext(context_name)

        if isinstance(self.current_bot, GeminiAPI):
            self.current_bot.settings['system_instruction'] = context
            self.current_bot.reset_chat()
            return f'Контекст {context_name} добавлен'
        
        elif isinstance(self.current_bot, CohereAPI):
            body = {"role": 'SYSTEM', "message": context}
        elif isinstance(self.current_bot, (GroqAPI,NvidiaAPI,TogetherAPI,GlifAPI,MistralAPI)):
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
        self.current_bot = self.api_factory.get(bot_name)
        self.clear()
        return f'Смена бота на {self.current_bot.name}'
    

    async def change_model(self, model_name: str) -> str:
        cur_bot = self.current_bot
        model = next((el for el in cur_bot.models if model_name in el), cur_bot.current_model)
        self.current_bot.current_model = model
        if hasattr(cur_bot, 'vlm_params') and model_name in cur_bot.vlm_params:
            self.current_bot.current_vlm_model = model_name
        self.clear()
        return f'Смена модели на {users.make_short_name(model_name)}'


    def clear(self) -> str:
        self.current_bot.context.clear()
        if self.current_bot.name == 'gemini':
            self.current_bot.settings['system_instruction'] = None
            self.current_bot.reset_chat()
        return 'Контекст диалога отчищен'
    

    def make_multi_modal_body(text, image, context: list, is_mistral = False) -> None:
        image_b64 = base64.b64encode(image.getvalue()).decode()
        if len(image_b64) > 180_000:
            print("Слишком большое изображение, сжимаем...")
            image_b64 = users.resize_image(image)
        part = f"data:image/jpeg;base64,{image_b64}"
        embedded_part = part if is_mistral else {"url": part}
        context.extend([
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {
                    "type": "image_url",
                    "image_url": embedded_part,
                },
            ],
        }
    ])


    async def prompt(self, text: str, image=None) -> str:
        output = await queue_manager.enqueue_request(self.current_bot.name, 
                                                     self.current_bot.prompt(text, image))
        # output = await self.current_bot.prompt(text, image)
        # print(output)
        return escape(output) 



class UsersMap():
    '''Main storage of user's sessions, common variables and functions'''
    def __init__(self):
        self._user_instances: dict[int, User] = {}
        self.context_dict: dict = json.loads(open('./prompts.json','r', encoding="utf-8").read())
        self.template_prompts: dict = {
                'Цитата': 'Напиши остроумную цитату. Цитата может принадлежать как реально существующей или существовавшей личности, так и вымышленного персонажа',
                'Шутка': 'Выступи в роли профессионального стендап комика и напиши остроумную шутку. Ответом должен быть только текст шутки',
                'Факт': 'Выступи в роли профессионального энциклопедиста и напиши один занимательный факт. Ответом должен быть только текст с фактом',
                'Квиз': '''Выступи в роли профессионального энциклопедиста и напиши три вопроса для занимательного квиза. 
                            Уровень вопросов: Старшая школа. Ответом должен быть только текст с тремя вопросами без ответов''',
                'Промпт': '''
                Write 4 interesting and unusual prompts for Stable Diffusion XL in different visual styles. 
                It must consist a sarcastic and ironic plot, showing the absurdity of the situation.
                Wrap each prompt in quotation marks `...`.'''
            }
        self.help = """  
**Help Guide**
Here are the available commands:  
1. **User Management:**  
- Add new user: `/add 123456 UserName` 
- Remove existing user: `/remove UserName`

2. **Agent Context:**
- `-i`: Get context_body info  
- `-a`: Add new context  
- `-r`: Remove existing context

**Usage:**
- `/context [-i | -r] context_name`  
- `/context [-a] context_name | context_body`

3. **Generate Image:**  
- Basic prompt: `/image your_prompt` for prompt enhancer
- Force provided prompt: `/image -f your_prompt`  
"""  
        self.buttons: dict = {
                'Добавить контекст':'change_context', 
                'Быстрые команды':'template_prompts',
                'Вывести инфо':'info',
                'Сменить бота':'change_bot', 
                'Очистить контекст':'clear',
                'Изменить модель бота':'change_model'
            }
        self.PARSE_MODE = ParseMode.MARKDOWN_V2
        self.DEFAULT_BOT: str = 'gemini' #'glif' gemini
        self.builder: ReplyKeyboardBuilder = self.create_builder()


    def create_builder(self) -> ReplyKeyboardBuilder:
        builder = ReplyKeyboardBuilder()
        for display_text in self.buttons:
            builder.button(text=display_text)
        return builder.adjust(3,3).as_markup()

    @lru_cache(maxsize=None)
    def get(self, user_id: int) -> User:
        return self._user_instances.setdefault(user_id, User())
    

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


    def set_kwargs(self, text: str = None, reply_markup = None) -> dict:
        return {'text': text or escape(self.help), 
                'parse_mode':self.PARSE_MODE, 
                'reply_markup': reply_markup or self.builder}


    async def check_and_clear(self, message: Message, type_prompt: str, user_name: str = '') -> User:
        user: User = self.get(message.from_user.id)
        if type_prompt in ['callback']:
            return user
        # if not user_name:
        #     user_name = db.check_user(message.from_user.id)
        if type_prompt in ['image']:
            logging.info(f'{user_name or message.from_user.id}: "{message.text}"')
            return user
        ## clear context after 30 minutes
        if (time() - user.time_dump) > 1800:
            user.clear()
        user.time_dump = time()
        type_prompt = {'text': message.text, 'photo': message.caption}.get(type_prompt, type_prompt)
        if user_name:
            logging.info(f'{user_name}: "{type_prompt}"')
        user.text = self.buttons.get(type_prompt, type_prompt)
        return user


    def get_context(self, key: str, data: dict = None) -> str | dict | None:
        data = data or self.context_dict
        return data.get(key) or next(
            (r for v in data.values() if isinstance(v, dict) and (r := self.get_context(key, v))), None)
    

    def create_inline_kb(self, dict_iter: dict, cb_type: str):
        builder_inline = InlineKeyboardBuilder()
        for value in dict_iter:
            data = CallbackClass(cb_type=cb_type, name=users.make_short_name(value)).pack()
            builder_inline.button(text=users.make_short_name(value), callback_data=data)
        return builder_inline.adjust(*[1]*len(dict_iter)).as_markup()


users = UsersMap()
queue_manager = RateLimitedQueueManager()
bot = Bot(token=os.getenv('TELEGRAM_API_KEY'))
db = DBConnection()
dp = Dispatcher()


class UserFilterMiddleware(BaseMiddleware):
    async def __call__(self, handler: callable, event: TelegramObject, data: dict):
        USER_ID = data['event_from_user'].id
        if user_name:= db.check_user(USER_ID):
            data.setdefault('user_name', user_name)
            await handler(event, data)
        else:
            if isinstance(event, Message):
                await bot.send_message(event.chat.id, 
                f'Доступ запрещен. Обратитесь к администратору. Ваш id: {USER_ID}')


dp.message.middleware(UserFilterMiddleware())
dp.callback_query.middleware(UserFilterMiddleware())


@dp.message(CommandStart())
async def start_handler(message: Message):
    await message.answer(f'Доступ открыт. Добро пожаловать {message.from_user.first_name}!')



@dp.message(Command(commands=["help"]))
async def help_handler(message: Message):
    await message.answer(**users.set_kwargs())

    
@dp.message(Command(commands=["context"]))
async def context_handler(message: Message, user_name: str):
    if user_name != 'ADMIN':
        await message.reply("You don't have admin privileges")
        return
    args = message.text.split(maxsplit=2)
    if (len(args) == 1) or (len(args) != 3 and not args[1].startswith('-')):
        await message.reply("Usage: `/context [-i/-r/-a] prompt_name [| prompt]`")
        return
    
    _, arg, prompt_body = args
    if arg == '-i':
        text = users.get_context(prompt_body) or 'Context name not found'
        await message.reply(**users.set_kwargs(escape(text)))
        return
    
    if arg == '-r' and prompt_body in users.context_dict:
        users.context_dict.pop(prompt_body)
        with open('./prompts.json', 'w', encoding="utf-8") as f:
            json.dump(users.context_dict, f, ensure_ascii=False, indent=4)
        await message.reply(f"Context {prompt_body} removed successfully.")
        return
    
    if arg == '-a' and prompt_body.count('|') == 1:
        prompt_name, prompt = [el.strip() for el in prompt_body.split("|",maxsplit=1)]
        if users.context_dict.get(prompt_name):
            await message.reply(f"Context {prompt_name} already exists")
            return
        try:
            users.context_dict[prompt_name] = prompt
            with open('./prompts.json', 'w', encoding="utf-8") as f:
                json.dump(users.context_dict, f, ensure_ascii=False, indent=4)
            await message.reply(f"Context {prompt_name} added successfully.")
        except Exception as e:
            await message.reply(f"An error occurred: {e}.")
    else:
        await message.reply("Usage: `/context -a prompt_name | prompt`")
        return
    

@dp.message(Command(commands=["add_user"]))
async def add_handler(message: Message, user_name: str):
    # user_name = db.check_user(message.from_user.id)
    if user_name != 'ADMIN':
        await message.reply("You don't have admin privileges")
        return
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.reply("Usage: `/add 123456 UserName`")
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


@dp.message(Command(commands=["remove_user"]))
async def remove_handler(message: Message, user_name: str):
    # user_name = db.check_user(message.from_user.id)
    if user_name != 'ADMIN':
        await message.reply("You don't have admin privileges")
        return
    args = message.text.split(maxsplit=1)
    if len(args) < 2:
        await message.reply("Usage: `/remove UserName`")
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
async def short_command_handler(message: Message, user_name: str):
    user = await users.check_and_clear(message, message.text.lstrip('/'), user_name)
    kwargs = users.set_kwargs(escape(getattr(user, user.text)()))
    await message.answer(**kwargs)



@dp.message(Command(commands=["image"]))
async def image_gen_handler(message: Message, user_name: str):
    user = await users.check_and_clear(message, "image", user_name)
    if user is None:
        return
    args = message.text.split(maxsplit=1)
    if len(args) != 2:
        await message.reply("Usage: `/image prompt` or `/image -f prompt`")
        return
    
    await message.reply('Картинка генерируется...')

    try:
        if args[1].startswith("-f"):
            caption = ''
        else:
            caption = await GeminiAPI().get_enhanced_prompt(args[1].lstrip('-f '))
            # await user.change_context('SDXL')
            # caption = await user.prompt(user.text)

        image_url = await GlifAPI().fetch_image_fal(caption or args[1])
        if 'error' in image_url:
            raise Exception()
        kwargs = {'photo': image_url, 'caption': caption}
        ## max caption length 1024
        kwargs['caption'] = f'`{escape(kwargs['caption'][:1000])}`'
        if kwargs['photo']:
            await message.answer_photo(**kwargs, parse_mode=users.PARSE_MODE)
        else:
            await message.reply(**users.set_kwargs(kwargs['caption']))
    except Exception as e:
        await message.reply(f"Ошибка: {e}")



@dp.message(lambda message: message.text in users.buttons)
async def reply_kb_command(message: Message):
    user = await users.check_and_clear(message, 'text')
    if user.text in {'info','clear'}:
        kwargs = users.set_kwargs(escape(getattr(user, user.text)()))
    else:
        command_dict = {'bot': [user.api_factory.bots, 'бота'],
                        'model': [user.current_bot.models, 'модель'],
                        'context':[users.context_dict, 'контекст'],
                        'prompts':[users.template_prompts, 'промпт']}
        items = command_dict.get(user.text.split('_')[-1])
        builder_inline = users.create_inline_kb(items[0], user.text)
        kwargs = users.set_kwargs(f'Выберите {items[-1]}:',  builder_inline)
    
    await message.answer(**kwargs)



@dp.message(F.content_type.in_({'photo'}))
async def photo_handler(message: Message | KeyboardButtonPollType, user_name: str):
    user = await users.check_and_clear(message, 'photo', user_name)
    if user.text is None:
        user.text = 'the provided image' # Следуй системным правилам
        # return
    
    if user.current_bot.name not in {'gemini', 'nvidia', 'groq', 'mistral'}:
        await user.change_bot('gemini')
        await user.change_context('SDXL')
        await message.reply("Выбран gemini и контекст SDXL")
    # if user.current_bot.name != 'nvidia':
    #     await user.change_bot('nvidia')
    #     await user.change_context('SDXL')
    #     await message.reply("Выбран nvidia")


    
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
async def echo_handler(message: Message | KeyboardButtonPollType, user_name: str):
    user = await users.check_and_clear(message, 'text', user_name)
    if user.text is None or user.text == '/':
        return await message.answer(**users.set_kwargs())
    try:
        await message.reply('Ожидайте...')
        output = await user.prompt(user.text)
        async for part in users.split_text(output):
            await message.answer(**users.set_kwargs(part))
        return
    except Exception as e:
        logging.info(e)
        await message.answer(f"{e}")
        return


@dp.callback_query(CallbackClass.filter(F.cb_type.contains('change')))
async def change_callback_handler(query: CallbackQuery, callback_data: CallbackClass):
    user = await users.check_and_clear(query, 'callback')
    output = await getattr(user, callback_data.cb_type)(callback_data.name)
    is_final_set = isinstance(output, str) and callback_data.name != '◀️'
    reply_markup = None if is_final_set else users.create_inline_kb(output, user.text)
    await query.message.edit_reply_markup(reply_markup=reply_markup)
    if is_final_set:
        await query.message.answer(output)
    await query.answer()


@dp.callback_query(CallbackClass.filter(F.cb_type.contains('template')))
async def template_callback_handler(query: CallbackQuery, callback_data: CallbackClass):
    try:
        user = await users.check_and_clear(query, 'callback')
        await query.message.edit_reply_markup(reply_markup=None)
        await query.message.reply('Ожидайте...')
        output = await user.template_prompts(callback_data.name)
        await query.message.answer(**users.set_kwargs(output))
        await query.answer()
    except Exception as e:
        logging.info(e)
        await query.message.answer("Error processing message. See logs for details")
        return



async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    print('Starting')
    asyncio.run(main())
