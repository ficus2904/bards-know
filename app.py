import os
import io
import re
import json
import atexit
import base64
import sqlite3
import asyncio
import aiohttp
import logging
import warnings
# import cohere
from argparse import ArgumentParser
from mistralai import Mistral
from google import genai
from google.genai.types import GenerateContentConfig, SafetySetting, HarmCategory, Tool, GoogleSearch
from abc import ABC, abstractmethod
from aiolimiter import AsyncLimiter
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
logging.basicConfig(filename='./app.log', level=logging.INFO, encoding='utf-8',
                    format='%(asctime)19s %(levelname)s: %(message)s')
for name in ['aiogram','httpx']: 
    logging.getLogger(name).setLevel(logging.WARNING)


class CallbackClass(CallbackData, prefix='callback'):
    cb_type: str
    name: str



class UserFilterMiddleware(BaseMiddleware):
    async def __call__(self, handler: callable, event: TelegramObject, data: dict):
        USER_ID = data['event_from_user'].id
        if user_name:= users.db.check_user(USER_ID):
            data.setdefault('user_name', user_name)
            try:
                await handler(event, data)
            except Exception as e:
                logging.exception(e)
                await bot.send_message(event.chat.id, f'❌ Error: {e}')
        else:
            if isinstance(event, Message):
                logging.warning(f'Unknown user {USER_ID}')
                await bot.send_message(event.chat.id, 
                f'Доступ запрещен. Обратитесь к администратору. Ваш id: {USER_ID}')



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
        self.safety_settings = [SafetySetting(category=category, threshold="OFF") for category in HarmCategory.__args__[1:]]
        self.models = [
            'gemini-2.0-flash-exp',
            'gemini-exp-1206',
            'learnlm-1.5-pro-experimental',
            'gemini-1.5-pro-exp-0827',
            'gemini-1.5-pro-002',
            ]
        self.current_model = self.models[0]
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.chat = None
        self.reset_chat()
        # self.imagen = genai.ImageGenerationModel("imagen-3.0-generate-001")

    async def prompt(self, text: str, image = None) -> str:
        try:
            if image is None:
                response = self.chat.send_message(text)
            else:
                response = self.chat.send_message([Image.open(image), text])
            return response.text
        except Exception as e:
            return f'error: {e}'
    
    
    def reset_chat(self, context: str = None, clear_system: bool = False, enable_search: bool = False):
        if context or clear_system or self.chat is None:
            config = GenerateContentConfig(
                    system_instruction=context,
                    safety_settings=self.safety_settings, 
                    tools=[Tool(google_search=GoogleSearch())] if enable_search else None
                    )
            self.chat = self.client.chats.create(model=self.current_model, config=config)


    def length(self) -> int: 
        return int(self.chat._config.system_instruction is not None) + len(self.chat._curated_history)

    async def gen_image(self, prompt, image_size):
        result = self.imagen.generate_images(
            prompt=prompt,
            number_of_images=1,
            safety_filter_level="block_only_high",
            person_generation="allow_adult",
            aspect_ratio=image_size or "3:4",
            # negative_prompt="Outside",
        )
        return result.images


    async def get_enhanced_prompt(self, init_prompt: str) -> str:
        '''DEPRECATED'''
        # self.settings['system_instruction'] = users.context_dict[''].get('SDXL')
        self.settings['system_instruction'] = users.get_context('SDXL')
        self.reset_chat()
        enhanced_prompt = await self.prompt(init_prompt)
        return enhanced_prompt



# class CohereAPI(BaseAPIInterface):
#     """Class for Cohere API"""
#     name = 'cohere'

#     def __init__(self):
#         self.client = openai.Client(self.api_key)
#         self.models = ['command-r-plus-08-2024','command-nightly','c4ai-aya-23-35b']
#         self.current_model = self.models[0]
#         self.context = []
    

#     async def prompt(self, text, image = None) -> str:
#         response = self.client.chat(
#             model=self.current_model,
#             chat_history=self.context or None,
#             message=text,
#             safety_mode='NONE'
#         )
#         self.context = response.chat_history
#         # print(response.text)
#         return response.text



class GroqAPI(BaseAPIInterface):
    """Class for Groq API"""
    name = 'groq'

    def __init__(self):
        self.client = Groq(api_key=self.api_key)
        self.models = [
            'llama-3.3-70b-versatile',
            'llama-3.3-70b-specdec',
            'llava-v1.5-7b-4096-preview',
            ] # https://console.groq.com/docs/models
        self.current_model = self.models[0]


    async def prompt(self, text: str, image = None) -> str:
        if image:
            self.context.clear()
            await User.make_multi_modal_body(text or "What's in this image?", image, self.context)
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
        self.models = [
            'mistral-large-2411',
            'pixtral-large-2411',
            'mistral-small-latest',
            'pixtral-12b-2409',
            ] # https://docs.mistral.ai/getting-started/models/
        self.current_model = self.models[0]


    async def prompt(self, text: str, image = None) -> str:
        if image:
            await User.make_multi_modal_body(text or "What's in this image?", 
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
            default_prompt = users.get_context('♾️ Универсальный')
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
                

    async def gen_image(self, prompt: str) -> dict:
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
                

    async def prompt(self, text, image = None) -> str:
        system_prompt = self.form_system_prompt()
        self.context.append({"role": "user","content": text})
        main_prompt = self.form_main_prompt()
        output = await self.fetch_data(main_prompt, system_prompt)
        self.context.append({'role':'assistant', 'content': output})
        return output



class FalAPI(BaseAPIInterface):
    """Class for Fal API"""
    name = 'fal'
    
    def __init__(self):
        self.models = ["v1.1-ultra","v1.1",]
        self.current_model = self.models[0]
        self.image_size = '9:16'
        self.raw = False


    def change_model(self, model):
        self.current_model = {
            "ultra": "v1.1-ultra",
            "1.1": "v1.1",
        }.get(model, "v1.1-ultra")
        if model == 'raw':
            self.raw = True
        elif model in {'no_raw','wo_raw'}:
            self.raw = False


    async def prompt(self, *args, **kwargs):
        pass


    def get_info(self) -> str:
        return (f'\n📏 Ratio: {self.image_size}\n'
                f'🤖 Model: {self.current_model} {int(self.raw)}')


    def change_image_size_old(self, image_size: str) -> str:
        self.image_size = {
            "9:16":"portrait_16_9", 
            "3:4":"portrait_4_3",
            "1:1":"square_hd", 
            "4:3":"landscape_4_3", 
            "16:9":"landscape_16_9",
        }.get(image_size, 'portrait_4_3')
        return self.image_size


    def change_image_size(self, image_size: str) -> str:
        if image_size in {"9:21","9:16","3:4","1:1","4:3","16:9","21:9"}:
            self.image_size = image_size
        else:
            self.image_size = "9:16"
        return self.image_size
    

    def get_kwargs(self, image_size: str, model: str) -> dict:
        if model:
            self.change_model(model)

        if self.current_model == 'v1.1':
            kwargs = {
                "image_size": self.change_image_size_old(image_size),
            }
        elif self.current_model == 'v1.1-ultra':
            kwargs = {
                "aspect_ratio": self.change_image_size(image_size),
                "raw": self.raw,
            }
        return kwargs


    async def gen_image(self, prompt: str, 
                        image_size: str | None = None, 
                        model: str | None = None) -> str:

        kwargs = self.get_kwargs(image_size, model)

        if prompt == '':
            return self.get_info()
        
        url = "https://fal.run/fal-ai/flux-pro/" + self.current_model
        headers = {"Authorization": f"Key {self.api_key}",
                   'Content-Type': 'application/json'}
        body = {
                "prompt": prompt,
                "num_images": 1,
                "enable_safety_checker": False,
                "safety_tolerance": "5",
                } | kwargs
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
                    return f'❌: {error_msg}'



class APIFactory:
    '''A factory pattern for creating bot interfaces'''
    bots_lst: list = [NvidiaAPI, GroqAPI, GeminiAPI, TogetherAPI, GlifAPI, MistralAPI] # CohereAPI
    bots: dict = {bot_class.name:bot_class for bot_class in bots_lst}
    image_bots_lst: list = [FalAPI]
    image_bots: dict = {bot_class.name:bot_class for bot_class in image_bots_lst}
    
    def __init__(self):
        self._instances: dict[str,BaseAPIInterface] = {}


    def get(self, bot_name: str) -> BaseAPIInterface:
        return self._instances.setdefault(bot_name, self.bots[bot_name]())



class RateLimitedQueueManager:
    def __init__(self):
        self.all_bots = APIFactory.bots | APIFactory.image_bots
        self.limiters = {name:AsyncLimiter(1, 30) for name in self.all_bots}
    
    async def enqueue_request(self, api_name: str, task):
        limiter = self.limiters[api_name]
        async with limiter:
            return await task



class ImageGenArgParser:
    def __init__(self):
        self.parser = ArgumentParser(description='Generate images with custom parameters')
        self.parser.add_argument('prompt', nargs='*', help='Image generation prompt')
        self.parser.add_argument('--ar', dest='aspect_ratio', help='Aspect ratio (e.g., 9:16)')
        self.parser.add_argument('--m', dest='model' ,help='Model selection') # type=int, choices=[1, 2]

    def get_args(self, args_str: str) -> tuple:
        try:
            prompt, flags = args_str.split('--', 1) if '--' in args_str else (args_str, '')
            args = self.parser.parse_args((f"--{flags}" if flags else '').split())
            return prompt.strip(), args.aspect_ratio, args.model
        except Exception as e:
            raise ValueError(f"Invalid arguments: {str(e)}")

    def get_usage(self) -> str:
        return ("Usage examples:\n"
                "• Basic: `/i your prompt here`\n"
                "• With aspect ratio: `/i your prompt here --ar 9:16`\n"
                "• With model selection: `/i your prompt here --m 1.1`\n"
                "• Combined: `/i your prompt here --ar 9:16 --m ultra`\n"
                "• Raw mode in ultra: `/i prompt --m raw`\n"
                "• Unable Raw mode in ultra: `/i --m no_raw OR wo_raw`")
    


class ConfigArgParser:
    def __init__(self):
        self.parser = ArgumentParser(description='Change configuration options')
        self.parser.add_argument('--es', dest='enable_search', help='Turn search in gemini',type=int, choices=[0, 1])
        # self.parser.add_argument('--m', dest='model' ,help='Model selection') # type=int, choices=[0, 1]

    def get_args(self, args_str: str) -> dict:
        try:
            args = self.parser.parse_args(args_str.split())
            return {k:v for k,v in (vars(args).items()) if v is not None}
        except Exception as e:
            raise ValueError(f"Invalid arguments: {str(e)}")

    def get_usage(self) -> str:
        return ("Usage examples:\n"
                "• Search on in gemini: `/conf --es 1`\n"
                "• Search off search in gemini: `/conf --es 0`\n"
                "• Another setting: `/conf --some some`\n")



class User:
    '''Specific user interface in chat'''
    def __init__(self):
        self.api_factory = APIFactory()
        self.current_bot: BaseAPIInterface = self.api_factory.get(users.DEFAULT_BOT)
        self.current_image_bot = FalAPI()
        self.time_dump = time()
        self.text = None
        

    async def change_context(self, context_name: str) -> str | dict:
        await self.clear()
        if context_name == '◀️':
            return users.context_dict
        
        context = users.get_context(context_name)

        if isinstance(context, dict): # subgroup
            context.setdefault('◀️','◀️')
            return context
        
        output_text = f'Контекст {context_name} добавлен'
        
        if context_name in users.context_dict['🖼️ Image_desc']:
            output_text += self.current_image_bot.get_info()

        if isinstance(self.current_bot, GeminiAPI):
            self.current_bot.reset_chat(context=context)
            return output_text
        
        # if isinstance(self.current_bot, CohereAPI):
        #     body = {"role": 'SYSTEM', "message": context}

        else:
            body = {'role':'system', 'content': context}

        self.current_bot.context.append(body)
        return output_text


    async def template_prompts(self, template: str) -> str:
        prompt_text = users.template_prompts.get(template)
        output = await self.prompt(prompt_text)
        return output
    

    async def info(self) -> str:
        check_vlm = hasattr(self.current_bot, 'vlm_params')
        is_gemini = self.current_bot.name == 'gemini'
        return '\n'.join(['',
            f'* Текущий бот: {self.current_bot.name}',
            f'* Модель: {self.current_bot.current_model}',
            f'* Модель vlm: {self.current_bot.current_vlm_model}' if check_vlm else '',
            f'* Размер контекста: {len(self.current_bot.context) if not is_gemini else self.current_bot.length()}'])
    

    async def change_config(self, kwargs: dict) -> str:
        output = ''
        if self.current_bot.name == 'gemini':
            if 'enable_search' in kwargs:
                self.current_bot.reset_chat(clear_system = True, 
                                            enable_search = kwargs['enable_search'])
                status = 'включен ✅' if kwargs['enable_search'] else 'выключен ❌'
                output += f'Поиск в gemini {status}\n' 
        return output.strip()



    async def change_bot(self, bot_name: str) -> str:
        self.current_bot = self.api_factory.get(bot_name)
        await self.clear()
        return f'🤖 Смена бота на {self.current_bot.name}'
    

    async def change_model(self, model_name: str) -> str:
        cur_bot = self.current_bot
        model = next((el for el in cur_bot.models if model_name in el), cur_bot.current_model)
        self.current_bot.current_model = model
        if hasattr(cur_bot, 'vlm_params') and model_name in cur_bot.vlm_params:
            self.current_bot.current_vlm_model = model_name
        await self.clear()
        return f'🔄 Смена модели на {users.make_short_name(model_name)}'


    async def clear(self) -> str:
        if self.current_bot.name == 'gemini':
            clear_system = self.current_bot.length() in {1,0}
            self.current_bot.reset_chat(clear_system=clear_system)
        else:
            ct = self.current_bot.context
            clear_system = True #len(ct) <= 1
            self.current_bot.context = [] if clear_system else ct[:1]
        status = ' полностью' if clear_system else ' кроме системного'
        return f'🧹 Диалог очищен{status}'
    

    async def make_multi_modal_body(text, image, context: list, is_mistral = False) -> None:
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
        output = await users.queue_manager.enqueue_request(self.current_bot.name, 
                                            self.current_bot.prompt(text, image))
        return escape(output)
    

    async def gen_image(self, *args, **kwargs) -> str:
        output = await users.queue_manager.enqueue_request(self.current_image_bot.name,
                                    self.current_image_bot.gen_image(*args, **kwargs))
        return output



class UsersMap():
    '''Main storage of user's sessions, common variables and functions'''
    def __init__(self):
        self.db = DBConnection()
        self.queue_manager = RateLimitedQueueManager()
        self._user_instances: dict[int, User] = {}
        self.context_dict: dict = json.loads(open('./prompts.json','r', encoding="utf-8").read())
        self.template_prompts: dict = {
                'Цитата': 'Напиши остроумную цитату. Цитата может принадлежать как реально существующей или существовавшей личности, так и вымышленного персонажа',
                'Шутка': 'Выступи в роли профессионального стендап комика и напиши остроумную шутку. Ответом должен быть только текст шутки',
                'Факт': 'Выступи в роли профессионального энциклопедиста и напиши один занимательный факт. Ответом должен быть только текст с фактом',
                'Квиз': '''Выступи в роли профессионального энциклопедиста и напиши три вопроса для занимательного квиза. 
                            Уровень вопросов: Старшая школа. Ответом должен быть только текст с тремя вопросами без ответов''',
                'QuizPlease': '''Выступи в роли профессионального ведущего quiz - вечеринок. Напиши 5 вопросов по теме кино и сериалы. 
                                Вопросы должны быть минимум продвинутого уровня, рассчитанные на искушённых киноманов.''',
                'Промпт': '''
                Write 4 interesting and unusual prompts for Stable Diffusion in different visual styles. 
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
- Equal commands: `/image` or `/i`
- Default size with prompt: `/image your_prompt` with 9:16 default size
- Target size with prompt: `/image your_prompt --ar 9:16` 
- Only change size: `/i --ar 9:16`
- Acceptable ratio size: 9:16, 3:4, 1:1, 4:3, 16:9
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
        self.image_arg_parser = ImageGenArgParser()
        self.config_arg_parser = ConfigArgParser()


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
        trigger = 'Closing Prompt'
        if (trigger_index := text.find(trigger, 2500)) != -1:  
            text = f'`{text[trigger_index + len(trigger):].strip(':\n"*_')}`'

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
        elif type_prompt in ['image']:
            logging.info(f'{user_name or message.from_user.id}: "{message.text}"')
            return user
        ## clear dialog context after 1 hour
        if (time() - user.time_dump) > 3600:
            user.clear()
        user.time_dump = time()
        type_prompt = {'text': message.text, 'photo': message.caption}.get(type_prompt, type_prompt)
        if user_name:
            logging.info(f'{user_name}: "{type_prompt}"')
        user.text = self.buttons.get(type_prompt, type_prompt)
        return user


    def get_context(self, key: str, data: dict = None) -> str | dict | None:
        '''Get target context in multilevel dict structure'''
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
bot = Bot(token=os.getenv('TELEGRAM_API_KEY'))
dp = Dispatcher()
dp.message.middleware(UserFilterMiddleware())
dp.callback_query.middleware(UserFilterMiddleware())


@dp.message(CommandStart())
async def start_handler(message: Message):
    output = f'Доступ открыт.\n'\
            f'Добро пожаловать {message.from_user.first_name}!'\
            '\nОтправьте /help для дополнительной информации'
    await message.answer(output)


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
        users.db.add_user(int(user_id), name)
        await message.reply(f"User {name} with ID {user_id} added successfully.")
    except sqlite3.IntegrityError:
        await message.reply("This user ID already exists.")
    except Exception as e:
        await message.reply(f"An error occurred: {e}.")


@dp.message(Command(commands=["remove_user"]))
async def remove_handler(message: Message, user_name: str):
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
            users.db.remove_user(name_to_remove)
            await message.reply(f"User `{name_to_remove}` removed successfully.")
        else:
            await message.reply(f"User `{name_to_remove}` not found.")
    except Exception as e:
        await message.reply(f"An error occurred: {e}.")


@dp.message(Command(commands=["info","clear"]))
async def short_command_handler(message: Message, user_name: str):
    user = await users.check_and_clear(message, message.text.lstrip('/'), user_name)
    kwargs = await users.set_kwargs(escape(getattr(user, user.text)()))
    await message.answer(**kwargs)


@dp.message(Command(commands=["conf","config"]))
async def config_handler(message: Message, user_name: str):
    user: User = users.get(message.from_user.id)
    if user_name != 'ADMIN':
        await message.reply("You don't have admin privileges")
        return
    
    args = message.text.split(maxsplit=1)

    if len(args) != 2:
        await message.reply(escape(
            users.config_arg_parser.get_usage()), parse_mode=users.PARSE_MODE)
        return
    
    output = await user.change_config(users.config_arg_parser.get_args(args[1]))
    await message.reply(output)


@dp.message(Command(commands=["i","I","image"]))
async def image_gen_handler(message: Message, user_name: str):
    user = await users.check_and_clear(message, "image", user_name)
    args = message.text.split(maxsplit=1)
    if len(args) != 2:
        await message.reply(escape(
            users.image_arg_parser.get_usage() + user.current_image_bot.get_info()
            ), parse_mode=users.PARSE_MODE)
        return
    
    await message.reply('Картинка генерируется ⏳')

    image_url = await user.gen_image(*users.image_arg_parser.get_args(args[1]))
    if image_url.startswith(('\n📏','❌')):
        await message.reply(image_url)
    else:
        await message.answer_photo(photo=image_url)



@dp.message(Command(commands=["imagen"]))
async def imagen_handler(message: Message, user_name: str):
    user = await users.check_and_clear(message, "image", user_name)
    args = message.text.split(maxsplit=1)
    if len(args) != 2:
        text = "Usage: `/i prompt` or `/i prompt --ar 9:16`" \
                "\nFor changing size: `/i --ar 9:16`"
        await message.reply(escape(text), parse_mode=users.PARSE_MODE)
        return
    
    await message.reply('Картинка генерируется ⏳')

    try:
        prompt, image_size = args[1], None
        if '--ar ' in prompt:
            prompt, image_size = prompt.split('--ar ')
        images = await user.current_bot.gen_image(prompt, image_size)

        for image in images:
            await message.answer_photo(photo=image)

    except Exception as e:
        await message.reply(f"❌ Ошибка: {e}")



@dp.message(lambda message: message.text in users.buttons)
async def reply_kb_command(message: Message):
    user = await users.check_and_clear(message, 'text')
    if user.text in {'info','clear'}:
        output = await getattr(user, user.text)()
        kwargs = users.set_kwargs(escape(output))
    else:
        command_dict = {'bot': [user.api_factory.bots, 'бота'],
                        'model': [user.current_bot.models, 'модель'],
                        'context':[users.context_dict, 'контекст'],
                        'prompts':[users.template_prompts, 'промпт']}
        items = command_dict.get(user.text.split('_')[-1])
        builder_inline = users.create_inline_kb(items[0], user.text)
        kwargs = users.set_kwargs(f'🤔 Выберите {items[-1]}:',  builder_inline)
    
    await message.answer(**kwargs)



@dp.message(F.content_type.in_({'photo'}))
async def photo_handler(message: Message | KeyboardButtonPollType, user_name: str):
    user = await users.check_and_clear(message, 'photo', user_name)
    if user.text is None:
        user.text = 'the provided image' # Следуй системным правилам
    
    if user.current_bot.name not in {'gemini', 'nvidia', 'groq', 'mistral'}:
        await user.change_bot('gemini')
        await users.get_context('♾️ Универсальный')
        await message.reply("Выбран gemini и контекст ♾️ Универсальный")

    text_reply = "Изображение получено! Ожидайте ⏳"
    if user.current_bot.name == 'nvidia' and user.current_bot.current_model not in user.current_bot.vlm_params:
        text_reply = f"Обработка изображения с использованием {user.current_bot.current_vlm_model} ⏳"
    await message.reply(text_reply)
    tg_photo = await bot.download(message.photo[-1].file_id)
    output = await user.prompt(user.text, tg_photo)
    async for part in users.split_text(output):
        await message.answer(**users.set_kwargs(part))


@dp.message(F.content_type.in_({'text'}))
async def echo_handler(message: Message | KeyboardButtonPollType, user_name: str):
    user = await users.check_and_clear(message, 'text', user_name)
    if user.text is None or user.text == '/':
        return await message.answer(**users.set_kwargs())

    await message.reply('Ожидайте ⏳')
    output = await user.prompt(user.text)
    async for part in users.split_text(output):
        await message.answer(**users.set_kwargs(part))
        

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
        await query.message.reply('Ожидайте ⏳')
        await query.answer()
        output = await user.template_prompts(callback_data.name)
        await query.message.answer(**users.set_kwargs(output))
    except Exception as e:
        logging.exception(e)
        await query.message.answer("Error processing message. See logs for details")
        return



async def main() -> None:
    await dp.start_polling(bot)


if __name__ == "__main__":
    print('Start polling')
    logging.warning('Start polling')
    asyncio.run(main())
