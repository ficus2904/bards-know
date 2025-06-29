import os
import io
import re
import wave
import json
import httpx
import atexit
import base64
import sqlite3
import asyncio
import aiohttp
import warnings
from loguru import logger
from contextlib import suppress
from argparse import ArgumentParser
from mistralai import Mistral
from google.genai import Client as GeminiClient
from google.genai import errors as GeminiError
from google.genai import types
from abc import ABC, abstractmethod
from aiolimiter import AsyncLimiter
from functools import lru_cache
from time import time
from groq import AsyncGroq
from openai import OpenAI
from aiogram import (
    Bot, 
    Dispatcher, 
    BaseMiddleware,
    exceptions,
    F)
from aiogram.types import (
    TelegramObject, 
    Message, 
    CallbackQuery,
    BotCommand
    # ReplyKeyboardMarkup,
    # InlineKeyboardMarkup,
    )
from aiogram.types import BufferedInputFile as BIF
from aiogram.utils.markdown import text
from aiogram.utils.formatting import ExpandableBlockQuote
from aiogram.filters import Command, CommandStart
from aiogram.filters.callback_data import CallbackData
from aiogram.enums import ParseMode
from aiogram.utils.chat_action import ChatActionSender
from aiogram.utils.keyboard import ReplyKeyboardBuilder, InlineKeyboardBuilder
from md2tgmd import escape
from PIL import Image, ImageOps
from dotenv import load_dotenv
load_dotenv(override=True)
warnings.simplefilter('ignore')

# uv run app.py

logger.add(sink='./app.log', 
           format='{time:YYYY-MM-DD HH:mm:ss} {level} {message}', 
           level='INFO',
           backtrace=True,
           rotation='1 MB',
           retention="7 days"
           )
                      

class CallbackClass(CallbackData, prefix='callback'):
    cb_type: str
    name: str

class MenuCallbacks(CallbackData, prefix='menu'):
    btn_text: str
    btn_target: str
    btn_act: str

class UserFilterMiddleware(BaseMiddleware):
    """
    UserFilterMiddleware is a middleware class that checks if a user is registered in the database before allowing them to proceed with the handler.

    Methods:
        __call__(handler: callable, event: TelegramObject, data: dict):
            Asynchronously checks if the user is registered in the database.
            If the user is registered, it adds the user's name to the data dictionary and calls the handler.
            If the user is not registered, it sends a warning message to the user and logs the event.

    Args:
        handler (callable): The handler function to be called if the user is registered.
        event (TelegramObject): The event object containing information about the Telegram event.
        data (dict): A dictionary containing event data, including the user information.

    Raises:
        Exception: If an error occurs while calling the handler, it logs the exception and sends an error message to the user.
    """
    async def __call__(self, 
                        handler: callable, # # type: ignore
                        event: TelegramObject | CallbackQuery, 
                        data: dict):
        USER_ID = data['event_from_user'].id
        if user_name:= users.db.check_user(USER_ID):
            data.setdefault('user_name', user_name)
            try:
                await handler(event, data)
            except Exception as e:
                logger.exception(e)
                if isinstance(event, Message):
                    await bot.send_message(
                        event.chat.id, **users.set_kwargs(f'❌ Error: {e}'[:200])
                        ) # type: ignore
        else:
            if isinstance(event, Message):
                logger.warning(f'Unknown user {USER_ID}')
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

    @staticmethod  
    def get_models(bot_menu: dict) -> list[str]:
        return [m["select"] for m in bot_menu["buttons"] if "select" in m]


class BOTS:
    """LLM bot interfaces"""
    class GeminiAPI(BaseAPIInterface):
        """Class for Gemini API"""
        name = 'gemini'
        safety_settings = [types.SafetySetting(category=category, 
                                threshold="BLOCK_NONE") for category # type: ignore
                                in types.HarmCategory._member_names_[1:]]

        def __init__(self, menu: dict):
            # self.models = [
            #     'gemini-2.5-flash',
            #     'gemini-2.0-flash-preview-image-generation',
            #     'gemini-2.5-flash-lite-preview-06-17',
            #     ]
            self.models = self.get_models(menu[self.name])
            self.current = self.models[0]
            self.chat = None
            self.proxy_status: bool = True
            self.search_status: bool = False
            self.client: GeminiClient = None
            self.reset_chat(with_proxy=self.proxy_status)


        def create_client(self, with_proxy: bool) -> None:
            http_options = {'api_version':'v1beta'}
            if with_proxy:
                if socks := os.getenv('SOCKS') or os.getenv('LOCAL_SOCKS'):
                    http_options = types.HttpOptions(
                        async_client_args={'proxy': socks},
                        **http_options)
                else:
                    http_options = types.HttpOptions(
                        base_url=os.getenv('WORKER'),
                        headers={'X-Custom-Auth': os.getenv('AUTH_SECRET'),
                                'EXTERNAL-URL': 'https://generativelanguage.googleapis.com'},
                        **http_options)
            self.proxy_status = with_proxy
            self.client = GeminiClient(api_key=self.api_key, http_options=http_options) # type: ignore

            
        async def prompt(self, 
                        text: str | None = None, 
                        data: list | None = None, 
                        attempts: int = 0) -> str | dict | None:
            try:
                content = [
                    *[types.Part.from_bytes(**subdata) # type: ignore
                    for subdata in data], text] if data else text
                response = await self.chat.send_message(content)
                if 'image' in self.current:
                    try:
                        for part in response.candidates[0].content.parts:
                            if part.inline_data is not None:
                                return {
                                    'photo': BIF(part.inline_data.data, "image.png"),
                                    'caption': part.text if part.text is not None else None,
                                    # 'reply_markup': users.builder,
                                    }
                            elif part.text is not None:
                                return part.text
                    except Exception:
                        return str(response.candidates[0].finish_reason)

                else:
                    return response.text
                
            except GeminiError.APIError as e:
                match e.code:
                    case code if 500 <= code < 600:
                        if attempts < 3:
                            await asyncio.sleep(5)
                            logger.warning(f'Gemini attempt: {attempts}')
                            return await self.prompt(text, data, attempts+1)

                return f'Gemini error {e.code}: {e}'
                    
                
            except Exception as e:
                logger.exception(e)
                return f'Exception in Gemini: {e}'
        
        
        def reset_chat(self, 
                       context: str | None = None, 
                       with_proxy: bool | None = None):
            if isinstance(with_proxy, bool):
                self.create_client(with_proxy)
            self.context = [{'role':'system', 'content': context}]
            config = types.GenerateContentConfig(
                system_instruction=context, 
                safety_settings=self.safety_settings,
                thinking_config=types.ThinkingConfig(thinking_budget=-1),
                )
            self.chat = self.client.aio.chats.create(model=self.current, config=config)
            if 'image' in self.current:
                self.chat._config.thinking_config = None
                self.chat._config.response_modalities = ['Text', 'Image']


        async def change_chat_config(self, clear: bool | None = None, 
                                    search: int | None = None, 
                                    new_model: str | None = None, 
                                    proxy: int | None = None) -> str | None:
            if self.chat._model != self.current:
                return self.reset_chat()
            
            if new_model:
                if new_model == 'list':
                    response = await self.client.aio.models.list(config={'query_base': True})
                    return "\n".join([model.name.split('/')[1] for model in response 
                            if 'generateContent' in model.supported_actions])
                else:
                    self.models.append(new_model)
                    return f'В gemini добавлена модель {new_model}'

            if clear:
                if self.chat._curated_history and self.chat._config.system_instruction:
                    self.chat._curated_history.clear()
                    return 'кроме системного'
                else:
                    self.chat._curated_history.clear()
                    self.chat._config.system_instruction = None
                    return 'полностью'

            if search is not None:
                self.search_status = bool(search)
                self.chat._config.tools = [types.Tool(google_search=types.GoogleSearch(),
                                                url_context = types.UrlContext())] if search else None
                return 'Поиск в gemini включен ✅' if search else 'Поиск в gemini выключен ❌'
            
            if isinstance(proxy, int):
                self.reset_chat(with_proxy=bool(proxy))
                return f'Прокси {'включен ✅' if proxy else 'выключен ❌'}\n'
            

        def length(self) -> int: 
            return int(self.chat._config.system_instruction is not None) + len(self.chat._curated_history)


        async def gen_image(self, prompt, image_size: str = '9:16', model: str | None = None):
            response = self.client.models.generate_images(
                model = f'imagen-3.0-generate-00{model or 2}',
                prompt = prompt,
                config = types.GenerateImagesConfig(
                    number_of_images=1,
                    include_rai_reason=True,
                    output_mime_type='image/jpeg',
                    safety_filter_level="BLOCK_LOW_AND_ABOVE", # type: ignore
                    # BLOCK_LOW_AND_ABOVE BLOCK_ONLY_HIGH
                    person_generation="ALLOW_ADULT", # type: ignore
                    # ALLOW_ADULT ALLOW_ALL
                    output_compression_quality=95,
                    aspect_ratio=image_size
                )
            )
            output = response.generated_images[0]
            return output.image or output.rai_filtered_reason
        

        async def tts(self, text: str, attempts: int = 0) -> BIF | None:
            try:
                response = await self.client.aio.models.generate_content(
                    model="gemini-2.5-flash-preview-tts",
                    contents=text,
                    config=types.GenerateContentConfig(
                        safety_settings=self.safety_settings,
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name='Kore'
                                    )
                                )
                        ),
                    )
                )
                output_bytes: bytes = response.candidates[0].content.parts[0].inline_data.data
                wav_buffer = io.BytesIO()
                with wave.open(wav_buffer, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    wf.writeframes(output_bytes)
                    return BIF(wav_buffer.getvalue(), filename='tts.wav')
            except GeminiError.APIError as e:
                match e.code:
                    case code if 500 <= code < 600:
                        if attempts < 3:
                            await asyncio.sleep(5)
                            logger.warning(f'Gemini attempt: {attempts}')
                            return await self.tts(text, attempts+1)

                raise Exception(f'Gemini error {e.code}: {e}')


        def dialogue_api_router(self, cmd: str) -> None:
            '''Remove last question and answer from the chat history'''
            system_content: str | None = self.context[0].get('content') if self.context else None
            system_instruction: str | None = {
                'dlg_last': system_content,
                'dlg_clear': system_content,
                'dlg_wipe': None,
            }.get(cmd)
            self.context = [{'role':'system', 'content': system_instruction}]
            config = types.GenerateContentConfig(
                system_instruction=system_instruction, 
                safety_settings=self.safety_settings,
                thinking_config=types.ThinkingConfig(thinking_budget=-1),
                )
            self.chat = self.client.aio.chats.create(
                model=self.current, 
                config=config, 
                history=self.chat.get_history()[:-2] if cmd == 'dlg_last' else None
                )
            if 'image' in self.current:
                self.chat._config.thinking_config = None
                self.chat._config.response_modalities = ['Text', 'Image']
            # return {'Удален последний ответ и вопрос из контекста Gemini'}


    class GroqAPI(BaseAPIInterface):
        """Class for Groq API"""
        name = 'groq'

        def __init__(self, menu: dict):
            # self.models = [
            #     'llama-4-scout-17b-16e-instruct',
            #     'llama-4-maverick-17b-128e-instruct',
            #     'deepseek-r1-distill-llama-70b',
            #     'llama-3.2-90b-vision-preview',
            #     ] # https://console.groq.com/docs/models
            self.models = self.get_models(menu[self.name])
            self.current = self.models[0]
            self.proxy_status: bool = False
            self.client: AsyncGroq = None
            self.create_client(self.proxy_status)


        def create_client(self, with_proxy: bool) -> None:
            '''Create a Groq client with or without proxy'''
            if with_proxy:
                if socks := os.getenv('SOCKS'):
                    kwargs = {'http_client': httpx.AsyncClient(proxy=socks)}
                else:
                    kwargs = {
                            'base_url': os.getenv('WORKER'),
                            'default_headers':{
                                'X-Custom-Auth': os.getenv('AUTH_SECRET'),
                                'EXTERNAL-URL': 'https://api.groq.com',}
                            }
            else:
                kwargs = {}
            self.proxy_status = with_proxy
            self.client = AsyncGroq(api_key=self.api_key,**kwargs)

        async def prompt(self, text: str, image = None) -> str:
            if image:
                self.context.clear()
                await User.make_multi_modal_body(text or "What's in this image?", image, self.context)
            else:
                body = {'role':'user', 'content': text}
                self.context.append(body)
            
            kwargs = {'model':('meta-llama/' if '4' in self.current else '') + self.current,
                    'messages': self.context}
            try:
                response = await self.client.chat.completions.create(**kwargs)
                data = response.choices[-1].message.content
                self.context.append({'role':'assistant', 'content': data})
                return data
            except Exception as e:
                return f'{e}'



    class MistralAPI(BaseAPIInterface):
        """Class for Mistral API"""
        name = 'mistral'

        def __init__(self, menu: dict):
            self.client = Mistral(api_key=self.api_key)
            # self.models = [
            #     'mistral-large-latest',
            #     'mistral-medium-latest',
            #     'mistral-small-latest',
            #     'pixtral-large-latest',
            #     ] # https://docs.mistral.ai/getting-started/models/
            self.models = self.get_models(menu[self.name])
            self.current = self.models[0]


        async def prompt(self, text: str, image = None) -> str:
            if image:
                await User.make_multi_modal_body(text or "What's in this image?", 
                                            image, self.context, is_mistral=True)
            else:
                body = {'role':'user', 'content': text}
                self.context.append(body)
            
            kwargs = {'model':self.models[-1] if image else self.current, 
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
                            'nvidia/vila': {
                                "max_tokens": 1024,
                                "temperature": 0.20,
                                "top_p": 0.7,
                                "stream": False
                            }
                        }
            self.current = self.models[0]
            self.current_vlm_model = self.models[-1]
            # self.context = []
        

        async def prompt(self, text, image = None) -> str:
            if image is None and self.current not in self.vlm_params:
                body = {'role':'user', 'content': text}
                self.context.append(body)
                response = self.client.chat.completions.create(
                    model=self.current,
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
                model = self.current if self.current in self.vlm_params else self.current_vlm_model
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
                            logger.error(e)
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
                'deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free',
                'Qwen/Qwen2-72B-Instruct',
                ] # https://docs.together.ai/docs/inference-models

            self.current = self.models[0]


        async def prompt(self, text, image=None) -> str:
            body = {'role':'user', 'content': text}
            self.context.append(body)
            response = self.client.chat.completions.create(
                model=self.current,
                messages=self.context,
                temperature=0.7,
                top_p=0.7,
                max_tokens=1024
            )
            output = response.choices[-1].message.content
            self.context.append({'role':'assistant', 'content':output})
            return output
        


    class OpenRouterAPI(BaseAPIInterface):
        """Class for OpenRouter API"""
        name = 'open_router'
        
        def __init__(self):
            self.client = OpenAI(api_key=self.api_key,
                                base_url="https://openrouter.ai/api/v1")
            self.models = [
                'deepseek/deepseek-chat-v3-0324',
                'qwen/qwen2.5-vl-32b-instruct',
                'meta-llama/llama-4-maverick',
                'meta-llama/llama-4-scout'
                ] # https://openrouter.ai/models

            self.current = self.models[0]
        

        async def prompt(self, text, image = None) -> str:
            body = {'role':'user', 'content': text}
            self.context.append(body)
            response = self.client.chat.completions.create(
                model=self.current +':free',
                messages=self.context,
            )
            output = response.choices[-1].message.content
            self.context.append({'role':'assistant', 'content':output})
            return output



    class GlifAPI(BaseAPIInterface):
        """Class for Glif API"""
        name = 'glif'

        def __init__(self, menu: dict):
            self.url = "https://simple-api.glif.app"
            self.headers: dict[str,str] = {"Authorization": f"Bearer {self.api_key}"}
            self.models_with_ids = {
                "Claude 4 sonnet":"clxwyy4pf0003jo5w0uddefhd",
                "Claude 4 opus":"clyzjs4ht0000iwvdlacfm44y",
                "OpenAI o3 mini":"clxx330wj000ipbq9rwh4hmp3",
                }
            self.models = list(self.models_with_ids.keys())
            self.current = self.models[0]


        def form_main_prompt(self) -> str:
            if len(self.context) > 2:
                return f'Use next json schema as context of our previous dialog: {self.context[1:]}'
            else:
                return self.context[-1].get('content')
        

        def form_system_prompt(self) -> str:
            if not self.context:
                default_prompt = users.get_context('♾️ Универсальный')
                self.context.append({'role':'system', 'content': default_prompt})
            return self.context[0].get('content')
        

        async def fetch_data(self, main_prompt: str, system_prompt: str) -> str:
            body: dict[str,str] = {
                "id": self.models_with_ids.get(self.current), 
                "inputs": {"main_prompt": main_prompt, "system_prompt": system_prompt}
                }
            async with aiohttp.ClientSession() as session:
                async with session.post(url=self.url, headers=self.headers, json=body) as response:
                    try:
                        response.raise_for_status()
                        answer = await response.json()
                        return answer['output'] or 'Error main'
                    except aiohttp.ClientResponseError as e:
                        logger.error(e)
                        return 'Error exception'
                    

        async def gen_image(self, prompt: str) -> dict:
            '''DEPRECATED'''
            body: dict[str,str] = {
                "id": {True:'clzmbpo6k000u1pb2ar3udjff',
                    False:'clzj1yoqc000i13n0li4mwa2b'}.get(prompt.startswith('-f')), 
                "inputs": {"initial_prompt": prompt.lstrip('-f ')}
                }
            async with aiohttp.ClientSession() as session:
                async with session.post(url=self.url,headers=self.headers,json=body, timeout=90) as response:
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
                                logger.error(error_msg := 'Timeout error')
                            case aiohttp.ClientResponseError():
                                logger.error(error_msg := f'HTTP error {e.status}: {e.message}')
                            case KeyError():
                                logger.error(error_msg := 'No output data')
                            case _:
                                logger.error(error_msg := f'Unexpected error: {str(e)}')
                        return {'error': error_msg}
                    

        async def prompt(self, text, image = None) -> str:
            system_prompt = self.form_system_prompt()
            self.context.append({"role": "user","content": text})
            main_prompt = self.form_main_prompt()
            output = await self.fetch_data(main_prompt, system_prompt)
            self.context.append({'role':'assistant', 'content': output})
            return output
        

        async def tts(self, text: str) -> str | None:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            body = {"id": 'cm6xrl05a0000s1qx4277g92y', "inputs": [text]}
            async with httpx.AsyncClient(timeout=httpx.Timeout(240)) as client:
                response = await client.post(url=self.url, headers=headers, json=body)
                try:
                    response.raise_for_status()
                    answer: dict = response.json()
                    return answer.get('output')
                except httpx.HTTPStatusError as e:
                    logger.error(f"{e.response.status_code} {e.response.text}")
                    return None


class FalAPI(BaseAPIInterface):
    """Class for Fal API"""
    name = 'FalAI'
    
    def __init__(self, menu: dict):
        # self.models = ["v1.1-ultra","v1.1",]
        self.models = self.get_models(menu[self.name])
        self.current = self.models[0]
        self.image_size = '9:16'
        self.raw = False


    def change_model(self, model):
        self.current = {
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
                f'🤖 Model: {self.current} {int(self.raw)}')


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

        if self.current == 'v1.1':
            kwargs = {
                "image_size": self.change_image_size_old(image_size),
            }
        elif self.current == 'v1.1-ultra':
            kwargs = {
                "aspect_ratio": self.change_image_size(image_size),
                "raw": self.raw,
            }
        return kwargs


    async def gen_image(self, prompt: str | None = None, 
                        image_size: str | None = None, 
                        model: str | None = None) -> str:

        kwargs = self.get_kwargs(image_size, model)

        if not prompt:
            return self.get_info()
        
        url = "https://fal.run/fal-ai/" + self.current
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
                            logger.error(error_msg := 'Timeout error')
                        case aiohttp.ClientResponseError():
                            logger.error(error_msg := f'HTTP error {e.status}: {e.message}')
                        case KeyError():
                            logger.error(error_msg := 'No output data')
                        case _:
                            logger.error(error_msg := f'Unexpected error: {str(e)}')
                    return f'❌: {error_msg}'



class APIFactory:
    '''A factory pattern for creating bot interfaces'''
    # bots_lst: list = [NvidiaAPI, GroqAPI, GeminiAPI, TogetherAPI, GlifAPI, MistralAPI, OpenRouterAPI]
    bots: dict = {v.name:v for k,v in BOTS.__dict__.items() if not k.startswith('__')}
    # bots: dict = {bot_class.name:bot_class for bot_class in bots_lst}
    image_bots_lst: list = [FalAPI]
    image_bots: dict = {bot_class.name:bot_class for bot_class in image_bots_lst}
    
    def __init__(self):
        self._instances: dict[str,BaseAPIInterface] = {}


    def get(self, menu: dict, bot_name: str) -> BaseAPIInterface:
        return self._instances.setdefault(bot_name, self.bots[bot_name](menu))


class RateLimitedQueueManager:
    """
    Manages a queue of API requests with rate limiting.
    Attributes:
        all_bots (set): A set of all bot names from APIFactory.
        limiters (dict): A dictionary mapping bot names to their respective AsyncLimiter instances.
    Methods:
        enqueue_request(api_name: str, task):
            Enqueues an API request for the specified bot, ensuring it adheres to the rate limit.
    """
    def __init__(self):
        self.all_bots = APIFactory.bots | APIFactory.image_bots
        self.limiters = {name:AsyncLimiter(5, 30) for name in self.all_bots}
    
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

    def get_args(self, args_str: str) -> tuple[str,str,str] | tuple[str,None,None]:
        try:
            prompt, flags = args_str.split('--', 1) if '--' in args_str else (args_str, '')
            args = self.parser.parse_args((f"--{flags}" if flags else '').split())
            return prompt.strip(), args.aspect_ratio, args.model
        except SystemExit:
            return prompt, None, None

    def get_usage(self) -> str:
        return text(
                "🖼️ Basic: `/i your prompt here`",
                "📐 With aspect ratio: `/i your prompt here --ar 9:16`",
                "⚙️ With model selection: `/i your prompt here --m 1.1`",
                "✨ Combined: `/i your prompt here --ar 9:16 --m ultra`",
                "🎬 Raw mode in ultra: `/i prompt --m raw`", 
                "🚫 Unable Raw mode in ultra: `/i --m no_raw OR wo_raw`",
                sep='\n')
    

class ConfigArgParser:
    """
    A class to parse and handle configuration arguments for the application.
    ----
    get_args(args_str: str) -> dict:
        Parses the provided argument string and returns a dictionary of arguments and their values.
        Parameters:
            args_str (str): A string of arguments to be parsed.
        Returns:
            dict: A dictionary containing the parsed arguments and their values.
        Raises:
            ValueError: If the arguments are invalid.
    get_usage() -> str:
        Provides usage examples for the argument parser.
        Returns:
            str: A string containing usage examples.
    """
    def __init__(self):
        self.parser = ArgumentParser(description='Change configuration options')
        self.parser.add_argument('--es', dest='search', help='Turn search in gemini',type=int, choices=[0, 1])
        self.parser.add_argument('--nm', dest='new_model', help='Add new model in gemini',type=str)
        self.parser.add_argument('--rr', dest='proxy', help='Turn proxy globally',type=int, choices=[0, 1])
        # self.parser.add_argument('--m', dest='model' ,help='Model selection') # type=int, choices=[0, 1]

    def get_args(self, args_str: str) -> dict:
        try:
            args = self.parser.parse_args(args_str.split())
            return {k:v for k,v in (vars(args).items()) if v is not None}
        except SystemExit:
            return {'SystemExit': "❌ Invalid arguments"}

    def get_usage(self) -> str:
        return text(
                "🔎 Search on in gemini: `/conf --es 1`",  
                "🚫 Search off in gemini: `/conf --es 0`",  
                "🤖 Gemini's models: `/conf --nm list`",  
                "➕ Add model to gemini: `/conf --nm str`",  
                "🔄 Turn proxy: `/conf --rr 1`", sep='\n'
                )



class User:
    '''Specific user interface in chat'''
    def __init__(self):
        self.api_factory = APIFactory()
        self.current_bot: BaseAPIInterface = self.api_factory.get(users.menu, users.DEFAULT_BOT)
        self.current_pic = FalAPI(users.menu)
        self.time_dump = time()
        self.text: str = None
        self.last_msg: dict = None # for deleting messages
        self.media_group_buffer: dict = None ## for media_group_handler
        self.nav_type: str = 'bot'
        

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
            output_text += self.current_pic.get_info()

        if isinstance(self.current_bot, BOTS.GeminiAPI):
            self.current_bot.reset_chat(context=context)
            return output_text

        else:
            body = {'role':'system', 'content': context}

        self.current_bot.context.append(body)
        return output_text


    async def template_prompts(self, template: str) -> str:
        if template.isdigit():
            for num, prompt_text in enumerate(users.template_prompts.values(), start=1):
                if num == int(template):
                    break
        else:
            prompt_text = users.template_prompts.get(template)
        output = await self.prompt(prompt_text)
        return output
    

    async def info(self, delete_prev: bool = False) -> tuple:
        is_gemini = self.current_bot.name == 'gemini'
        output = text(
            f'🤖 Текущий бот: {self.current_bot.name}',
            f'🧩 Модель: {self.current_bot.current}',
            f'📚 Размер контекста: {len(self.current_bot.context) 
                                    if not is_gemini else self.current_bot.length()}',
            sep='\n')
        if delete_prev:
            await bot.delete_message(**self.last_msg) # type: ignore
        return output, self.make_conf_btns()
    

    def make_conf_btns(self):
        emoji_stat = {True:'✅',False:'❌'}
        output = []
        if hasattr(self.current_bot,'proxy_status'):
            output += [f'{emoji_stat[self.current_bot.proxy_status]} proxy']
        if hasattr(self.current_bot,'search_status'):
            output += [f'{emoji_stat[self.current_bot.search_status]} search']
        return users.create_inline_kb(output, 'conf')
    

    async def change_config(self, kwargs: dict) -> str:
        output = ''
        # if (proxy := kwargs.get('turn_proxy')) is not None:
        #     output += users.turn_proxy(proxy)
        if self.current_bot.name == 'gemini':
            output += f'{await self.current_bot.change_chat_config(**kwargs)}\n'
        if self.current_bot.name == 'groq':
            self.current_bot.create_client(kwargs['proxy'])
            output += f'Прокси {'включен ✅' if kwargs['proxy'] else 'выключен ❌'}\n'
        if error := kwargs.get('SystemExit'):
            return error + '\n' + users.config_arg_parser.get_usage()

        return output.strip().strip('None')


    async def change_bot(self, bot_name: str) -> str:
        self.current_bot = self.api_factory.get(bot_name)
        await self.clear()
        return f'🤖 Смена бота на {self.current_bot.name}'
    

    async def change_model(self, model_name: str) -> str:
        cur_bot = self.current_bot
        model = next((el for el in cur_bot.models if model_name in el), cur_bot.current)
        self.current_bot.current = model
        if hasattr(cur_bot, 'vlm_params') and model_name in cur_bot.vlm_params:
            self.current_bot.current_vlm_model = model_name
        await self.clear()
        return f'🔄 Смена модели на {users.make_short_name(model_name)}'


    def change_model_new(self, btn_type: str, bot: str, model: str) -> None:
        cbt = f'current_{btn_type}'
        if getattr(self, cbt).name != bot:
            setattr(self, cbt, getattr(self,  f'{btn_type}_dct')[bot](self.menu))
        if model:
            getattr(self, cbt).current = model


    def change_state(self, state: str) -> None:
        cbt = getattr(self, f'current_{self.nav_type}')
        if hasattr(cbt, state):
            attr: bool = getattr(cbt, state)
            if 'proxy' in state:
                cbt.create_client(not attr)
            else:
                setattr(cbt, state, not attr)


    def dialogue_router(self, cmd: str) -> str:
        """Router for command actions."""
        cbt = getattr(self, f'current_{self.nav_type}')
        if hasattr(cbt, 'dialogue_api_router'):
            getattr(cbt, 'dialogue_api_router')(cmd)
            return {'dlg_last': 'Удален последний ответ и вопрос из контекста',
                    'dlg_clear': 'Очистка контекста',
                    'dlg_wipe': 'Очистка контекста и системной инструкции'}[cmd]
        else:
            return f'❌ Команда {cmd} отсутствует в {cbt.name}'
            


    async def clear(self, delete_prev: bool = False) -> tuple:
        if self.current_bot.name == 'gemini':
            status = await self.current_bot.change_chat_config(clear=True)
        else:
            ct = self.current_bot.context
            if (len(ct) not in {0,1}) and (ct[0].get('role') == 'system'):
                self.current_bot.context = ct[:1]
                status = 'кроме системного'
            else:
                self.current_bot.context.clear()
                status = 'полностью'
        if delete_prev:
            await bot.delete_message(**self.last_msg) # type: ignore
        return f'🧹 Диалог очищен {status}', None
    

    async def make_multi_modal_body(text, 
                                    image, 
                                    context: list, 
                                    is_mistral = False) -> None:
        image_b64 = base64.b64encode(image.get('data')).decode()
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


    async def prompt(self, *args) -> str:
        output = await users.queue_manager.enqueue_request(self.current_bot.name, 
                                            self.current_bot.prompt(*args))
        return output


    async def gen_image(self, *args, **kwargs) -> str:
        output = await users.queue_manager.enqueue_request(self.current_pic.name,
                                    self.current_pic.gen_image(*args, **kwargs))
        return output


    async def delete_menu_cmd(self) -> None:
        """Deletes the /menu message in the chat"""
        if self.last_msg:
            await bot.delete_message(**self.last_msg) # type: ignore
            self.last_msg = {}



class UsersMap():
    '''Main storage of user's sessions, common variables and functions'''
    def __init__(self):
        self.load_json = lambda file: json.loads(open(f'./{file}.json', 'r', encoding="utf-8").read())
        self.menu: dict[str, dict] = self.load_json('settings')
        self.state_btns: set = set(filter(None, map(lambda btn: btn.get('state'), 
                        self.menu['switch_bot']['buttons'] + self.menu['switch_pic']['buttons'])))
        self.db = DBConnection()
        self.queue_manager = RateLimitedQueueManager()
        self._user_instances: dict[int, User] = {}
        self.context_dict: dict = self.load_json('prompts')
        # self.context_dict: dict = json.loads(open('./prompts.json','r', encoding="utf-8").read())
        self.template_prompts: dict = {
                '💬 Цитата': 'Напиши остроумную цитату. Цитата может принадлежать как реально существующей или существовавшей личности, так и вымышленного персонажа',
                '🤣 Шутка': self.context_dict.get("🤡 Юмор",{}).get("🍻 Братюня",'') + '\nВыступи в роли профессионального стендап комика и напиши остроумную шутку. Ответом должен быть только текст шутки',
                '💡 Факт': 'Выступи в роли профессионального энциклопедиста и напиши один занимательный факт. Ответом должен быть только текст с фактом',
                '🤔 Квиз': '''Выступи в роли профессионального энциклопедиста и напиши три вопроса для занимательного квиза. 
                            Уровень вопросов: Старшая школа. Ответом должен быть только текст с тремя вопросами без ответов''',
                '🤓 QuizPlease': '''Выступи в роли профессионального ведущего quiz - вечеринок. Напиши 5 вопросов по теме кино и сериалы. 
                                Вопросы должны быть минимум продвинутого уровня, рассчитанные на искушённых киноманов.''',
                '📝 Промпт': ('Write 4 interesting and unusual prompts in different visual styles. '
                            'First, think through the main idea of the picture and then realize the visual storytelling that will be revealed by that one prompt. '
                            'It must consist a sarcastic, ironic and brutal plot with black humor, showing the situation. '
                            'Wrap each prompt in quotation marks `...`.'),
                '⚖️ Правда': self.context_dict.get("🤡 Юмор",{}).get("🍻 Братюня",'') + (
                            '\nНапиши непопулярное мнение на твое усмотрение на основе научных данных.'
                            'Желательно такое, чтобы мир прям наизнанку и пиши развернутый аргументированный ответ')
            }
        self.help = self.create_help()
        self.buttons: dict = {
                'Меню':'menu', 
                'Добавить контекст':'change_context', 
                'Быстрые команды':'template_prompts',
                # 'Вывести инфо':'info',
                # 'Сменить бота':'change_bot', 
                # 'Очистить контекст':'clear',
                # 'Изменить модель бота':'change_model'
            }
        self.simple_cmds: set = {'clear', 'info'}
        self.PARSE_MODE = ParseMode.MARKDOWN_V2
        self.DEFAULT_BOT: str = 'gemini' #'glif' gemini mistral
        self.proxy_settings = os.environ.get('HTTPS_PROXY')
        # self.builder: ReplyKeyboardBuilder = self.create_builder()
        self.image_arg_parser = ImageGenArgParser()
        self.config_arg_parser = ConfigArgParser()


    # def create_builder(self) -> ReplyKeyboardMarkup | InlineKeyboardMarkup:
    #     builder = ReplyKeyboardBuilder()
    #     for display_text in self.buttons:
    #         builder.button(text=display_text)
    #     return builder.adjust(1).as_markup()


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
            img.thumbnail((img.size[0] * 0.9, img.size[1] * 0.9), Image.ADAPTIVE) # type: ignore
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
        

    async def split_text_old(self, text: str, max_length=4096):
        trigger = 'Closing Prompt'
        if (trigger_index := text.find(trigger, 2500)) != -1:  
            text = f'`{text[trigger_index + len(trigger):].strip(':\n"*_ ')}`'

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


    async def split_text(self, text: str, max_length: int = 4090):
        """
        Разбивает текст на фрагменты, учитывая markdown-блоки, так чтобы блоки не делились.
        """
        trigger = 'Closing Prompt'
        if (trigger_index := text.find(trigger, 2500)) != -1:  
            text = f'`{text[trigger_index + len(trigger):].strip(':\n"*_ ')}`'

        start = 0
        markers = ["```", "`", "**", "__", "*", "_", "~"]

        while start < len(text):
            # Если оставшийся текст короче max_length, берём весь
            if len(text) - start <= max_length:
                chunk = text[start:]
            else:
                # Ищем оптимальную точку разбиения (например, последний перенос строки или пробел)
                split_index = self.find_split_index(text, start, max_length)
                chunk = text[start:split_index]

            # Проверка баланса для каждого markdown-маркера
            for marker in markers:
                # Если количество вхождений маркера нечётное – блок не закрыт
                if chunk.count(marker) % 2 != 0:
                    # Пытаемся найти закрывающий маркер в оставшейся части текста
                    end = min(start + max_length, len(text))
                    closing_index = text.find(marker, start + len(chunk), end)
                    if closing_index != -1:
                        # Расширяем фрагмент, включая закрывающий маркер
                        chunk = text[start:closing_index + len(marker)]
                    else:
                        # Если закрывающий маркер не найден, можно добавить его искусственно
                        chunk += marker
            
            # Если фрагмент превышает max_length, обрезаем его
            if len(chunk) > max_length:
                chunk = chunk[:max_length]

            yield chunk
            # Переход к следующему фрагменту – учитываем, что мы могли превысить max_length
            start += len(chunk)


    def find_split_index(self, text: str, start: int, max_length: int) -> int:
        """
        Ищет индекс для разбиения текста, ориентируясь на последний перенос строки или пробел.
        """
        split_index = start + max_length
        for separator in ('\n', ' '):
            last_separator = text.rfind(separator, start, split_index)
            if last_separator != -1:
                split_index = last_separator
                break
        return split_index


    def set_kwargs(self, 
                   text: str | None = None, 
                   reply_markup: ReplyKeyboardBuilder | None = None, 
                   parse_mode: ParseMode | None = None) -> dict:
        return {'text': text or self.help, 
                'parse_mode': parse_mode or self.PARSE_MODE, 
                'reply_markup': reply_markup,# or self.builder,
                }


    async def send_split_response(self, message: Message, output: str):
        async for part in users.split_text(output):
            try:
                await message.answer(**users.set_kwargs(escape(part))) # type: ignore
            except exceptions.TelegramBadRequest:
                await message.answer(**users.set_kwargs(part, parse_mode=ParseMode.HTML)) # type: ignore


    def create_help(self) -> str:
        help_items_simple = [
            text('1. 🧑‍💼 User Management:',
                 '🔹 Add new user: /add 123456 UserName',
                 '🔹 Remove existing user: /remove UserName',
                  sep='\n'),
            text('2. 🗂️ Agent Context:',
                 '🔹 -i: Get context_body info',
                 '🔹 -a: Add new context',
                 '🔹 -r: Remove existing context',
                 'Usage:',
                 '🔹 /context [-i | -r] [context_name | c OR current]',
                 '🔹 /context [-a] context_name | context_body', 
                 sep='\n'),
            text('3. 🖼️ Generate Image:',
                 '🔹 Equal commands: /image or /i',
                 '🔹 Default size with prompt: /image your_prompt with 9:16 default size',
                 '🔹 Target size with prompt: /image your_prompt --ar 9:16',
                 '🔹 Only change size: /i --ar 9:16',
                 '🔹 Acceptable ratio size: 9:16, 3:4, 1:1, 4:3, 16:9', 
                 sep='\n'),
            text('4. ⚙️ Change config',
                 '🔹 /conf: Get conf cases', 
                 sep='\n'),
        ]
        return ExpandableBlockQuote(text(*help_items_simple, sep='\n')).as_markdown()


    async def check_and_clear(self, 
                              message: Message | CallbackQuery, 
                              type_prompt: str, 
                              user_name: str = '') -> User:
        user: User = self.get(message.from_user.id)  # type: ignore
        if type_prompt in {'callback','tts'}:
            return user
        elif type_prompt in ['gen_image']:
            logger.info(f'{user_name or message.from_user.id}: "{message.text}"') # type: ignore
            return user
        ## clear dialog context after 1 hour
        if (time() - user.time_dump) > 3600:
            user.clear()
        user.time_dump = time()
        if type_prompt == 'text':
            user.text = self.buttons.get(message.text, message.text) # type: ignore
            type_prompt = message.text # type: ignore
        else:
            user.text = message.caption or f"the provided {type_prompt}." # type: ignore
            type_prompt = (lambda x: f'{x}: {message.caption or "no desc"}')(type_prompt) # type: ignore
        user.text = user.text.lstrip('/')
        if user_name:
            # logger.info(f'{user_name}: "{type_prompt if len(type_prompt) < 100 else 'too long prompt'}"')
            logger.info(f'{user_name}: {type_prompt[:100]}...')
         
        return user


    def get_context(self, key: str, data: dict | None = None) -> str | dict | None:
        '''Get target context in multilevel dict structure'''
        data = data or self.context_dict
        return data.get(key) or next(
            (r for v in data.values() if isinstance(v, dict) and (r := self.get_context(key, v))), None)
    

    def create_inline_kb(self, dict_iter: dict | list, cb_type: str):
        builder_inline = InlineKeyboardBuilder()
        for value in dict_iter:
            cb_btn_name = users.make_short_name(value)
            data = CallbackClass(cb_type=cb_type, name=cb_btn_name).pack()
            builder_inline.button(text=cb_btn_name, callback_data=data)
        return builder_inline.adjust(*[2]*(len(dict_iter)//2)).as_markup()


    def get_current_context(self, user_id: int) -> str:
        user: User = self.get(user_id)
        ct = user.current_bot.context
        if len(ct) and ct[0].get('role') == 'system':
            return ct[0].get('content')
        return 'No current context'
    

    def create_menu_kb(self, user: User, target: str, btn_act: str | None = None) -> dict:
        """Creates keyboard markup with buttons for menu navigation
        
        Args:
            target: Target menu section
            btn_act: Current button action
            
        Returns:
            tuple: Headline text and keyboard markup
        """
        builder = InlineKeyboardBuilder()
        
        if target in {'bot', 'pic','switch_bot', 'switch_pic'}:
            user.nav_type = target.replace('switch_','')
        target_menu: dict = self.menu[target]
        cb = getattr(user, f'current_{user.nav_type}')
        if user.nav_type in {'bot', 'pic'} and target not in {'main', 'switch', 'utils', 'cmd'}:
            headline: str = f'Текущая модель:\n🤖 {cb.name}\n'\
                            f'{'🧩' if user.nav_type == 'bot' else '🎨'} {cb.current}'
        else:
            headline: str = target_menu['text']

        btns_list: list[dict] = target_menu["buttons"]
        for btn in btns_list:
            state, select = btn.get('state'), btn.get('select')
            if state in self.state_btns and not hasattr(cb, state):
                continue
            builder.button(
                text=self._add_emoji_prefix(cb, btn_act, state, select) + btn['text'], 
                callback_data=MenuCallbacks(
                    btn_text=btn['text'], 
                    btn_target=btn.get('target', target),
                    btn_act=state or select or 'go',
                    ).pack())
        columns = 2 if len(target_menu["buttons"]) > 5 else 1
        return {'text': headline,'reply_markup': builder.adjust(columns).as_markup()}
    

    def _add_emoji_prefix(self, cb,
            btn_act: str | None, 
            state: str | None, 
            select: str | None ) -> str:
        """Adds appropriate emoji prefix to button text
        
        Args:
            btn_act: Current button action
            state: Button state for toggle buttons
            select: Button selection for model selection
            
        Returns:
            str: Emoji prefix or empty string
        """
        # Handle state toggle buttons
        if state and hasattr(cb, state):
            return '✅ ' if getattr(cb, state) else '❌ '
        # Handle model selection buttons
        if select:
            return '✅ ' if select in {cb.current, btn_act} else ''
        return ''



users = UsersMap()
bot = Bot(token=os.environ['TELEGRAM_API_KEY'])
dp = Dispatcher()
dp.message.middleware(UserFilterMiddleware())
dp.callback_query.middleware(UserFilterMiddleware())


class Handlers:
    @dp.message(CommandStart())
    async def start_handler(message: Message):
        output = ('Доступ открыт.\nДобро пожаловать '
            f'{message.from_user.first_name}!\n' # type: ignore
            'Отправьте /help для дополнительной информации')
        await message.answer(output)


    @dp.message(Command(commands=["menu"]))
    async def cmd_settings(message: Message):
        """Entry point for settings via /menu command."""
        user: User = users.get(message.from_user.id) # type: ignore
        user.last_msg = {'chat_id': message.chat.id, 
                         'message_id': message.message_id}
        await message.answer(**users.create_menu_kb(user, "main")) # type: ignore
        await user.delete_menu_cmd()


    @dp.message(Command(commands=["context"]))
    async def context_handler(message: Message, user_name: str):
        if user_name != 'ADMIN':
            await message.reply("You don't have admin privileges")
            return
        args = message.text.split(maxsplit=2) # type: ignore
        if (len(args) == 1) or (len(args) != 3 and not args[1].startswith('-')):
            await message.reply("Usage: `/context [-i/-r/-a] prompt_name [| prompt]`")
            return
        
        _, arg, prompt_body = args
        if arg == '-i':
            if prompt_body in ['c', 'current']:
                text = users.get_current_context(message.from_user.id) # type: ignore
            else:
                text = users.get_context(prompt_body) or 'Context name not found'
            text = f'```plaintext\n{text}\n```'
            await message.reply(**users.set_kwargs(escape(text))) # type: ignore
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
        

    @dp.message(Command(commands=["add_user","remove_user"]))
    async def add_remove_user_handler(message: Message, user_name: str):
        """
        Handles the addition and removal of users based on the command received.

        Parameters
        ----------
        message : types.Message
            The message object containing the command and arguments.
        
        user_name : str
            The user name if user in base

        Returns
        -------
        None
            The function sends a reply message to the user and does not return any value.

        Raises
        ------
        Exception
            
        sqlite3.IntegrityError
            If the user ID already exists in the database.
        Exception
            If the user does not have admin privileges or if the command usage is incorrect. 
            OR For any other unexpected errors.

        Notes
        -----
        - The function checks if the user has admin privileges by verifying the username.
        - It splits the message text to extract the command and arguments.
        - If the command is `/add_user`, it expects a user ID and a username, adds the user to the database, and sends a success message.
        - If the command is `/remove_user`, it expects a username, checks if the user exists, removes the user from the database, and sends a success message.
        - If the user does not exist, it sends a warning message.
        - The function handles various exceptions and provides appropriate feedback to the user.

        Examples
        --------
        >>> # Example of adding a user
        >>> message_text = "/add_user 123456 JohnDoe"
        >>> await add_remove_user_handler(message)
        ✅ User JohnDoe with ID 123456 added successfully.

        >>> # Example of removing a user
        >>> message_text = "/remove_user JohnDoe"
        >>> await add_remove_user_handler(message)
        ✅ User `JohnDoe` removed successfully.

        >>> # Example of invalid usage
        >>> message_text = "/add_user"
        >>> await add_remove_user_handler(message)
        ❌ An error occurred: Usage: /add_user 123456 UserName
        """
        try:
            if user_name != 'ADMIN':
                raise Exception("You don't have admin privileges")
            
            is_add_command = message.text.startswith('/add') # type: ignore
            args = message.text.split(maxsplit=1) # type: ignore
            if len(args) < 2:
                raise Exception(f"Usage: `/add {'123456 ' if is_add_command else ''}UserName`")
        
            if is_add_command:
                user_id, name = args[1].split(maxsplit=1)
                users.db.add_user(int(user_id), name)
                output = f"✅ User {name} with ID {user_id} added successfully."
            else:
                name_to_remove = args[1].strip()
                if user_name:
                    users.db.remove_user(name_to_remove)
                    output = f"✅ User `{name_to_remove}` removed successfully."
                else:
                    output = f"⚠️ User `{name_to_remove}` not found."

        except sqlite3.IntegrityError:
            output = "⚠️ This user ID already exists."
        except Exception as e:
            output = f"❌ An error occurred: {e}."
        finally:
            await message.reply(output)


    @dp.message(Command(commands=["info","clear"]))
    async def short_command_handler(message: Message, user_name: str):
        await Handlers.reply_kb_command(message)


    @dp.message(Command(commands=["change_context"]))
    async def change_context_command_handler(message: Message, user_name: str):
        await Handlers.reply_kb_command(message)


    @dp.message(Command(commands=["conf"]))
    async def config_handler(message: Message, user_name: str):
        user: User = users.get(message.from_user.id) # type: ignore
        if user_name != 'ADMIN':
            await message.reply("You don't have admin privileges")
            return
        
        args = message.text.split(maxsplit=1) # type: ignore

        if len(args) != 2:
            await message.reply(escape(
                users.config_arg_parser.get_usage()), parse_mode=users.PARSE_MODE)
            return
        
        output = await user.change_config(users.config_arg_parser.get_args(args[1]))
        await message.reply(output)


    @dp.message(Command(commands=["i","I","image"]))
    async def image_gen_handler(message: Message, user_name: str):
        user = await users.check_and_clear(message, "gen_image", user_name)
        args = message.text.split(maxsplit=1) # type: ignore
        if len(args) != 2:
            await message.reply(escape(
                users.image_arg_parser.get_usage() + user.current_pic.get_info()
                ), parse_mode=users.PARSE_MODE)
            return
        
        await message.reply('Картинка генерируется ⏳')
        async with ChatActionSender.upload_photo(chat_id=message.chat.id, bot=bot):
            image_url = await user.gen_image(*users.image_arg_parser.get_args(args[1]))
        if image_url.startswith(('\n📏','❌')):
            await message.reply(image_url)
        else:
            await message.answer_photo(photo=image_url)


    @dp.message(Command(commands=["imagen"]))
    async def imagen_handler(message: Message, user_name: str):
        user = await users.check_and_clear(message, "gen_image", user_name)
        args = message.text.split(maxsplit=1) # type: ignore
        if len(args) != 2 or (is_gemini := user.current_bot.name != 'gemini'):
            text = 'Переключите на gemini' if is_gemini else "Usage: `/imagen prompt --ar 9:16 --m 2`"
            await message.reply(escape(text), parse_mode=users.PARSE_MODE)
            return

        await message.reply('Картинка генерируется ⏳')
        try:
            parse_args = users.image_arg_parser.get_args(args[1])
            async with ChatActionSender.upload_photo(chat_id=message.chat.id, bot=bot):
                output = await user.current_bot.gen_image(*parse_args)
            if isinstance(output, str):
                await message.reply(f"❌ RAI: {output}")
            else:
                await message.answer_photo(photo=output)

        except Exception as e:
            await message.reply(f"❌ Ошибка: {e}")


    @dp.message(Command(commands=["tts"]))
    async def generate_audio_story(message: Message, user_name: str):
        user = await users.check_and_clear(message, 'tts', user_name)
        parts = message.text.split(maxsplit=1) # type: ignore
        text = parts[1] if len(parts) == 2 else None
        if (not text) or (user.current_bot.name == 'gemini'):
            await message.reply("Отсутствует текст/переключите модель на gemini")
        else:
            async with ChatActionSender.record_voice(chat_id=message.chat.id, bot=bot):
                link = await user.current_bot.tts(text)
                if link:
                    await message.answer_voice(link)




    @dp.message(F.text.in_(users.buttons) | F.text.casefold().in_(users.simple_cmds))
    async def reply_kb_command(message: Message):
        user = await users.check_and_clear(message, 'text')
        user.last_msg = {'chat_id': message.chat.id, 
                         'message_id': message.message_id,}
        if (user.text.casefold() in ('menu', 'меню')):
            kwargs: dict = users.create_menu_kb(user, "main")
        elif (simple_cmd := user.text.casefold()) in users.simple_cmds:
            output, builder_inline = await getattr(user, simple_cmd)(True)
            kwargs: dict = users.set_kwargs(escape(output), builder_inline)
        else:
            command_dict: dict[str, list] = {
                'bot': (user.api_factory.bots, 'бота'),
                'model': (user.current_bot.models, 'модель'),
                'context':(users.context_dict, 'контекст'),
                'prompts':(users.template_prompts, 'промпт')
                }
            items: tuple[dict, str] = command_dict[user.text.split('_')[-1]]
            builder_inline = users.create_inline_kb(items[0], user.text)
            kwargs: dict = users.set_kwargs(f'🤔 Выберите {items[-1]}:',  builder_inline)
        await message.answer(**kwargs) # type: ignore


    @dp.message(F.media_group_id)
    async def media_group_handler(message: Message, user_name: str):
        data_info = getattr(message, message.content_type, None)
        mime_type = getattr(data_info, 'mime_type', 'image/jpeg')
        data_type = mime_type.split('/')[0]
        user = await users.check_and_clear(message, data_type, user_name)
        if data_type == 'image':
            data_info = data_info[-1] # type: ignore
        data = await bot.download(data_info.file_id) # type: ignore
        current_dict = {'data': data.getvalue(), 'mime_type': mime_type}

        if user.media_group_buffer is None:
            user.media_group_buffer = current_dict
            return
        else:
            tg_photo1 = user.media_group_buffer
            tg_photo2 = current_dict
            user.media_group_buffer = None

        async with ChatActionSender.typing(chat_id=message.chat.id, bot=bot):
            output = await user.prompt(user.text, [tg_photo1, tg_photo2])
            if isinstance(output, str):
                await users.send_split_response(message, output)
            else:
                await message.answer_photo(**output) # type: ignore


    @dp.message(F.content_type.in_({'photo'}))
    async def photo_handler(message: Message, user_name: str):
        user = await users.check_and_clear(message, 'image', user_name)
        if user.current_bot.name not in {'gemini'}:
            await user.change_bot('gemini')
            # await users.get_context('♾️ Универсальный')
            await message.reply("Выбран gemini")

        async with ChatActionSender.typing(chat_id=message.chat.id, bot=bot):
            tg_photo = await bot.download(message.photo[-1].file_id)  # type: ignore
            output = await user.prompt(user.text, [{'data': tg_photo.getvalue(), 
                                                    'mime_type': 'image/jpeg'}])
            if isinstance(output, str):
                await users.send_split_response(message, output)
            else:
                await message.answer_photo(**output) # type: ignore


    @dp.message(F.content_type.in_({'voice','video_note','video','document'}))
    async def data_handler(message: Message, user_name: str):
        data_info = getattr(message, message.content_type, None)
        mime_type = getattr(data_info, 'mime_type', None)
        data_type = mime_type.split('/')[0] # type: ignore
        user = await users.check_and_clear(message, data_type, user_name)
        if user.current_bot.name not in {'gemini'}:
            await user.change_bot('gemini')

        await message.reply(f"{data_type.capitalize()} получено! Ожидайте ⏳")
        async with ChatActionSender.typing(chat_id=message.chat.id, bot=bot):
            data = await bot.download(data_info.file_id) # type: ignore
            output = await user.prompt(user.text, [{'data': data.getvalue(), 'mime_type': mime_type}])
            if isinstance(output, str):
                await users.send_split_response(message, output)
            else:
                await message.answer_photo(**output) # type: ignore



    @dp.message(F.text.startswith('/') | F.text.casefold().startswith('help'))
    async def unknown_handler(message: Message, user_name: str):
        await bot.delete_message(message.chat.id, message.message_id)
        await message.answer(**users.set_kwargs()) # type: ignore


    @dp.message(F.content_type.in_({'text'}))
    async def text_handler(message: Message, user_name: str):
        user = await users.check_and_clear(message, 'text', user_name)
        async with ChatActionSender.typing(chat_id=message.chat.id, bot=bot):
            output = await user.prompt(user.text)
            if isinstance(output, str):
                await users.send_split_response(message, output)
            else:
                await message.answer_photo(**output) # type: ignore
        


class Callbacks:
    @dp.callback_query(CallbackClass.filter(F.cb_type.contains('change')))
    async def change_callback_handler(query: CallbackQuery, callback_data: CallbackClass):
        user = await users.check_and_clear(query, 'callback')
        output = await getattr(user, callback_data.cb_type)(callback_data.name)
        is_final_set = isinstance(output, str) and callback_data.name != '◀️'
        reply_markup = None if is_final_set else users.create_inline_kb(output, user.text)
        await query.message.edit_reply_markup(reply_markup=reply_markup) # type: ignore
        if is_final_set:
            await query.message.edit_text(output) # type: ignore
            # await bot.delete_message(query.message.chat.id, user.last_msg['message_id'])
            # await bot.delete_message(**user.last_msg) # type: ignore
            await user.delete_menu_cmd()


    @dp.callback_query(CallbackClass.filter(F.cb_type.contains('conf')))
    async def conf_callback_handler(query: CallbackQuery, callback_data: CallbackClass):
        user = await users.check_and_clear(query, 'callback')
        var_name = f'{callback_data.name.split()[-1]}'
        kwargs: dict = {var_name: not getattr(user.current_bot, f'{var_name}_status')}
        await user.change_config(kwargs)
        await query.message.edit_reply_markup(reply_markup=user.make_conf_btns()) # type: ignore



    @dp.callback_query(CallbackClass.filter(F.cb_type.contains('template')))
    async def template_callback_handler(query: CallbackQuery, callback_data: CallbackClass):
        user = await users.check_and_clear(query, 'callback')
        await query.message.edit_text(f'{callback_data.name} 👇') # type: ignore
        async with ChatActionSender.typing(chat_id=query.message.chat.id, bot=bot):  # type: ignore
            output = await user.template_prompts(callback_data.name)
        await users.send_split_response(query.message, output) # type: ignore
        # await bot.delete_message(query.message.chat.id, user.last_msg['message_id'])
        await bot.delete_message(**user.last_msg) # type: ignore



    @dp.callback_query(MenuCallbacks.filter(F.btn_act == "go"))
    async def menu_callback_go(query: CallbackQuery, callback_data: MenuCallbacks):
        user = await users.check_and_clear(query, 'callback')
        cb = callback_data
        if cb.btn_target == 'exit':
            await query.message.delete() # type: ignore
            return
        elif cb.btn_target == 'change_context':
            cur_bot = user.current_bot
            kwargs: dict = {
                'text': f'Текущая модель:\n🤖 {cur_bot.name}\n🧩 {cur_bot.current}',
                'reply_markup':users.create_inline_kb(users.context_dict, 'change_context')
            }
        elif cb.btn_target.startswith('cmd_'):
            kwargs: dict = {'text': await user.template_prompts(cb.btn_target.split('_')[-1])}
        elif cb.btn_target.startswith('dlg_'):
            kwargs: dict = {'text': user.dialogue_router(cb.btn_target)}
        else:
            kwargs: dict = users.create_menu_kb(user, cb.btn_target)
        await query.message.edit_text(**kwargs) # type: ignore
        await query.answer()
        

    @dp.callback_query(MenuCallbacks.filter(F.btn_act != "go"))
    async def menu_callback_state_select(query: CallbackQuery, callback_data: MenuCallbacks):
        user = await users.check_and_clear(query, 'callback')
        cb = callback_data
        if cb.btn_act in users.state_btns:
            user.change_state(cb.btn_act)
        else:
            user.change_model_new(
                user.nav_type, 
                cb.btn_target.replace('_pic',''), 
                cb.btn_act
                )
        with suppress(Exception):
            await query.message.edit_text(  # type: ignore
                **users.create_menu_kb(user, cb.btn_target, cb.btn_act)
                )
        await query.answer()


async def main() -> None:
    await bot.set_my_commands([
        BotCommand(command="/menu", description="🍽️ Меню"),
        BotCommand(command="/change_context", description="✍️ Добавить контекст"),
        BotCommand(command="/clear", description="🧹 Очистить диалог"),
        BotCommand(command="/info", description="📚 Вывести инфо"),
    ])
    await dp.start_polling(bot)


if __name__ == "__main__":
    logger.info('Start polling')
    asyncio.run(main())