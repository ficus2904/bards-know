import os
import io
import math
import wave
import json
import httpx
import atexit
import base64
import sqlite3
import asyncio
import aiohttp
import warnings
from datetime import datetime, timezone
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
from openai import OpenAI, AsyncOpenAI
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
    )
from aiogram.types import BufferedInputFile as BIF
from aiogram.utils.markdown import text
from aiogram.utils.formatting import ExpandableBlockQuote, as_numbered_list
from aiogram.filters import Command, CommandStart, CommandObject
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

class MenuCallbacks(CallbackData, prefix='m'):
    target: str
    act: str

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
        if username:= users.db.check_tg_id(USER_ID):
            data.setdefault('username', username)
            try:
                await handler(event, data)
            except Exception as e:
                logger.exception(e)
                if isinstance(event, Message):
                    await bot.send_message(event.chat.id, f'❌ Error: {e}'[:200])
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

    def add_user(self, username: str, tg_id: str) -> None:
        """
        Insert a new user into the table.
        :param username: The user's name.
        :param tg_id: The user's ID.
        """
        query = 'INSERT INTO users (id, name) VALUES (?, ?)'
        self.cursor.execute(query, (tg_id, username))
        self.conn.commit()

    def remove_user(self, username: str) -> None:
        """
        Remove a user from the users table.
        :param username: The user's name.
        """
        query = 'DELETE FROM users WHERE name = ?'
        self.cursor.execute(query, (username,))
        self.conn.commit()
    
    def check_tg_id(self, user_id: int) -> str | None:
        answer = self.fetchone("SELECT name FROM users WHERE id = ? LIMIT 1", (user_id,))
        return answer[0] if answer else None
    
    def check_username(self, username: str) -> str | None:
        answer = self.fetchone("SELECT name FROM users WHERE name = ? LIMIT 1", (username,))
        return answer[0] if answer else None
      
    def get_list(self) -> str | None:
        return self.fetchall("SELECT * FROM users")

    def close(self):
        self.conn.close()


class BaseAPIInterface(ABC):
    @classmethod  
    def __init_subclass__(cls, **kwargs):  
        super().__init_subclass__(**kwargs)  
        cls.api_key = cls.get_api_key(cls.name)

    @property
    def context(self) -> list:
        if not hasattr(self, '_context'):
            self._context = []
        return self._context
    
    @context.setter
    def context(self, value):
        self._context = value

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
        safety_settings = [types.SafetySetting(
            category=category, 
            threshold="BLOCK_NONE"
            ) for category in types.HarmCategory._member_names_[1:-5]]

        def __init__(self, menu: dict):
            self.models = self.get_models(menu[self.name])
            self.current = self.models[0]
            self.chat = None
            self.states: dict[str,str] = {
                "proxy": True,
                "search": False,
                "code": False,
                # "image_gen_reset": True,
            }
            self.client: GeminiClient = None
            self.reset_chat(with_proxy=self.states['proxy'])


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
            self.states['proxy'] = with_proxy
            self.client = GeminiClient(api_key=self.api_key, http_options=http_options)

            
        async def prompt(self, 
                        text: str | None = None, 
                        data: list | None = None, 
                        attempts: int = 0) -> str | dict | None:
            try:
                content= [
                    *[types.Part.from_bytes(**subdata) # type: ignore
                    for subdata in data], text] if data else text
                response = await self.chat.send_message(content)
                if 'image' in self.current:
                    try:
                        output: dict = {}
                        for part in response.candidates[0].content.parts:
                            if part.inline_data is not None:
                                output['photo'] = BIF(part.inline_data.data, "image.png")
                            elif part.text is not None:
                                output['caption'] = part.text

                        return (output if 'photo' in output  else response.text
                                ) or response.candidates[0].finish_reason
                    
                    except Exception:
                        return str(response.candidates[0].finish_reason)
                    finally:
                        if self.states.get('image_gen_reset'):
                            self.dialogue_api_router('clear')
                else:
                    if response.text:
                        return response.text
                    else:
                        raise GeminiError.APIError(598, response_json={})
                
            except GeminiError.APIError as e:
                match e.code:
                    case code if 500 <= code < 600:
                        if attempts < 3:
                            await asyncio.sleep(10)
                            logger.warning(f'Gemini attempt: {attempts}')
                            return await self.prompt(text, data, attempts+1)

                return f'Gemini error {e.code}: {e}'
                    
                
            except Exception as e:
                logger.exception(e)
                return f'Exception in Gemini: {e}'
        
        
        def reset_chat(self, 
                       context: str | None = None, 
                       with_proxy: bool | None = None,
                       history: str | None = None):
            
            if isinstance(with_proxy, bool):
                self.create_client(with_proxy)
            self.context = [{'role':'system', 'content': context}]
            config = types.GenerateContentConfig(
                system_instruction=context, 
                safety_settings=self.safety_settings,
                thinking_config=types.ThinkingConfig(thinking_budget=-1),
                )
            self.chat = self.client.aio.chats.create(
                model=self.current, 
                config=config,
                history=history,
                )
            if 'image' not in self.current:
                self.chat._config.tools = [
                    types.Tool(
                        google_search=types.GoogleSearch() if self.states['search'] else None,
                        url_context = types.UrlContext() if self.states['search'] else None,
                        code_execution=types.ToolCodeExecution if self.states['code'] else None,
                        )
                    ]
            else:
                self.chat._config.tools = None
                self.chat._config.thinking_config = None
                self.chat._config.response_modalities = ['Text','Image']


        async def get_list(self) -> str:
            response = await self.client.aio.models.list(config={'query_base': True})
            lst = [model.name.split('/')[1] for model in response 
                    if 'generateContent' in model.supported_actions]
            return "\n".join(lst)


        def length(self) -> int: 
            return int(self.chat._config.system_instruction is not None) + len(self.chat._curated_history)


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


        def dialogue_api_router(self, cmd: str | None = None) -> str:
            '''Remove last question and answer from the chat history'''
            if cmd is None:
                cmd: str = 'clear' if (self.chat._curated_history 
                                and self.chat._config.system_instruction
                                ) else 'wipe'
            system_content: str | None = self.context[0].get('content') if self.context else None
            system_instruction: str = {
                'last': system_content,
                'clear': system_content,
                'wipe': None,}[cmd]
            self.reset_chat(
                context=system_instruction,
                history=self.chat.get_history()[:-2] if cmd == 'last' else None
                )
            return 'кроме системного' if cmd == 'clear' else 'полностью'



    class ZenmuxAPI(BaseAPIInterface):
        """Class for Zenmux API"""
        name = 'zenmux'

        def __init__(self, menu: dict):
            self.base_url = "https://zenmux.ai/api/v1"
            self.models = self.get_models(menu[self.name])
            self.current = self.models[0]
            self.chat = None
            self.states: dict[str,str] = {
                "proxy": True,
                "search": True,
                "code": False,
                # "image_gen_reset": True,
            }
            self.client: AsyncOpenAI = None
            self.create_client(self.states['proxy'])


        def create_client(self, with_proxy: bool) -> None:
            '''Create a Groq client with or without proxy'''
            if with_proxy:
                if socks := os.getenv('SOCKS'):
                    kwargs = {
                        'http_client': httpx.AsyncClient(proxy=socks),
                        'base_url': self.base_url,
                        }
                else:
                    kwargs = {
                            'base_url': os.getenv('WORKER'),
                            'default_headers':{
                                'X-Custom-Auth': os.getenv('AUTH_SECRET'),
                                'EXTERNAL-URL': self.base_url,}
                            }
            else:
                kwargs = {'base_url': self.base_url}
            self.states['proxy'] = with_proxy
            self.client = AsyncOpenAI(api_key=self.api_key,**kwargs)
        

        async def prompt(self, text, image: list[dict] = None) -> str | dict:
            if image:
                User.make_multi_modal_body(image, text, self.context)
            else:
                body = {'role':'user', 'content': text}
                self.context.append(body)
            response = await self.client.chat.completions.create(
                model=self.current,
                messages=self.context,
                modalities=["image", "text"] if 'image' in self.current else ["text"],
            )
            output = response.choices[-1].message
            self.context.append({'role':'assistant', 'content':output.content})
            if hasattr(output, 'images'):
                return User.encode_multi_modal_body(output)
            else:
                return output.content



    class GroqAPI(BaseAPIInterface):
        """Class for Groq API"""
        name = 'groq'

        def __init__(self, menu: dict):
            self.models = self.get_models(menu[self.name])
            self.current = self.models[0]
            self.states: dict[str,str] = {
                "proxy": True,
                "search": True,
                "code": False,
            }
            self.client: AsyncGroq = None
            self.create_client(self.states['proxy'])


        def create_client(self, with_proxy: bool) -> None:
            '''Create a Groq client with or without proxy'''
            if with_proxy:
                if socks := os.getenv('SOCKS') or os.getenv('LOCAL_SOCKS'):
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
            self.states['proxy'] = with_proxy
            self.client = AsyncGroq(api_key=self.api_key,**kwargs)


        async def prompt(self, text: str, image = None) -> str:
            if image:
                self.context.clear()
                User.make_multi_modal_body(image, text, self.context)
            else:
                body = {'role':'user', 'content': text}
                self.context.append(body)
            
            kwargs: dict = self.get_kwargs()
            try:
                response = await self.client.chat.completions.create(
                    model=self.current, 
                    messages=self.context,
                    **kwargs)
                data = response.choices[-1].message.content
                self.context.append({'role':'assistant', 'content': data})
                return data
            except Exception as e:
                return f'{e}'
            

        def get_kwargs(self) -> dict:
            match self.current:
                case 'qwen/qwen3-32b':
                    kwargs = {
                        'reasoning_format': 'hidden',
                        'reasoning_effort': 'default',
                        'temperature':0.6, 
                        'top_p':0.95, 
                    }
                case s if 'openai' in s:
                    kwargs = {
                        'temperature':1,
                        'max_completion_tokens':4096,
                        'top_p':1,
                        'tool_choice':"auto",
                        "reasoning_effort": "low",
                        "include_reasoning": True,
                        'tools': [x for x in [
                            {"type": "browser_search"} if self.states['search'] else None,
                            {"type": "code_interpreter" if self.states['code'] else None},
                            ] if x]
                    }
                case _:
                    kwargs = {}
            return kwargs
        


    class MistralAPI(BaseAPIInterface):
        """Class for Mistral API"""
        name = 'mistral'
        # https://docs.mistral.ai/getting-started/models/

        def __init__(self, menu: dict):
            self.client = Mistral(api_key=self.api_key)
            self.models = self.get_models(menu[self.name])
            self.current = self.models[0]


        async def prompt(self, text: str, image = None) -> str:
            if image:
                User.make_multi_modal_body(image, text, self.context)
            else:
                body = {'role':'user', 'content': text}
                self.context.append(body)
            
            kwargs = {'model':self.models[-1] if image else self.current, 
                    'messages': self.context}
            response = await self.client.chat.complete_async(**kwargs)
            response = response.choices[-1].message.content
            self.context.append({'role':'assistant', 'content':response})
            return response



    class OpenRouterAPI(BaseAPIInterface):
        """Class for OpenRouter API"""
        name = 'open_router'
        # https://openrouter.ai/models
        
        def __init__(self, menu: dict):
            self.models = self.get_models(menu[self.name])
            self.current = self.models[0]
            self.base_url = "https://openrouter.ai/api/v1"
            self.states: dict[str,str] = {
                "proxy": True,
                # "search": True,
                # "code": False,
                # "image_gen_reset": True,
            }
            # self.proxy_status: bool = True
            # self.image_gen_reset_status: bool = True
            self.client: AsyncOpenAI = None
            self.create_client(self.states['proxy'])


        def create_client(self, with_proxy: bool) -> None:
            '''Create a Groq client with or without proxy'''
            if with_proxy:
                if socks := os.getenv('SOCKS'):
                    kwargs = {
                        'http_client': httpx.AsyncClient(proxy=socks),
                        'base_url': self.base_url,
                        }
                else:
                    kwargs = {
                            'base_url': os.getenv('WORKER'),
                            'default_headers':{
                                'X-Custom-Auth': os.getenv('AUTH_SECRET'),
                                'EXTERNAL-URL': self.base_url,}
                            }
            else:
                kwargs = {'base_url': self.base_url}
            self.states['proxy'] = with_proxy
            self.client = AsyncOpenAI(api_key=self.api_key,**kwargs)
        

        async def prompt(self, text, image: list[dict] = None) -> str | dict:
            if image:
                User.make_multi_modal_body(image, text, self.context)
            else:
                body = {'role':'user', 'content': text}
                self.context.append(body)
            response = await self.client.chat.completions.create(
                model=self.current,
                messages=self.context,
                modalities=["image", "text"] if 'image' in self.current else ["text"],
            )
            output = response.choices[-1].message
            # if self.states['image_gen_reset']:
            #     self.context.clear()
            # else:
            self.context.append({'role':'assistant', 'content':output.content})
            if hasattr(output, 'images'):
                return User.encode_multi_modal_body(output)
            else:
                return output.content



class PIC_BOTS:
    """Picture generation bot interfaces"""

    class FalAPI(BaseAPIInterface):
        """Class for Fal API"""
        name = 'FalAI'
        
        def __init__(self, menu: dict):
            self.headers: dict[str,str] = {
                "Authorization": f"Key {self.api_key}",
                'Content-Type': 'application/json',
                }
            self.models = self.get_models(menu[self.name])
            self.current = self.models[0]
            self.image_size = '9:16'
            self.raw = False


        async def prompt(self, *args, **kwargs):
            pass


        def get_info(self) -> str:
            return (f'\n📏 Ratio: {self.image_size}\n'
                    f'🤖 Model: {self.current}')


        def to_aspect_ratio(self) -> str:
            '''DEPRECATED'''
            return {
                "portrait_16_9":"9:16", 
                "portrait_4_3":"3:4",
                "square_hd":"1:1", 
                "landscape_4_3":"4:3", 
                "landscape_16_9":"16:9",
            }.get(self.image_size, '4:3')


        def get_kwargs(self) -> dict[str,str]:
            match self.current:
                case 'flux-pro/v1.1-ultra':
                    kwargs = {
                        "aspect_ratio": self.image_size,
                        "raw": self.raw,
                    }
                case s if 'imagen' in s:
                    kwargs = {
                        "aspect_ratio": self.image_size,
                    }
                case s if 'banana' in s:
                    kwargs = {
                        "aspect_ratio": self.image_size,
                        "output_format": 'webp',
                        "resolution": "4K",
                    }
                case s if 'seedream' in s:
                    kwargs = {
                        "image_size": s,
                    }
                case _:
                    kwargs = {}
            return kwargs


        async def gen_image(self, prompt: str) -> dict | str:
            '''Method to generate an image using the Fal API'''
            kwargs = self.get_kwargs()
            body: dict[str,str] = {
                    "prompt": prompt,
                    "num_images": 1,
                    "enable_safety_checker": False,
                    "safety_tolerance": "5",
                    } | kwargs
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=f"https://fal.run/fal-ai/{self.current}",
                    headers=self.headers,
                    json=body, 
                    timeout=90,
                    ) as response:
                    try:
                        response.raise_for_status()
                        answer = await response.json()
                        try:
                            return {'photo': answer['images'][0]['url']}
                        except Exception:
                            return {'photo': str(answer)}
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
    bots: dict = {v.name:v for k,v in BOTS.__dict__.items() if not k.startswith('__')}
    image_bots: dict = {v.name:v for k,v in PIC_BOTS.__dict__.items() if not k.startswith('__')}
    
    def __init__(self):
        self._instances: dict[str,BaseAPIInterface] = {}


    def get(self, bot_type: str, bot_name: str) -> BaseAPIInterface:
        dct = self.bots if bot_type == 'bot' else self.image_bots
        return self._instances.setdefault(bot_name, dct[bot_name](users.menu))


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
        self.limiters = {name:AsyncLimiter(5, 60) for name in self.all_bots}
    
    async def enqueue_request(self, api_name: str, task):
        limiter = self.limiters[api_name]
        async with limiter:
            return await task


    

class UsersArgParser:
    """A wrapper for ArgumentParser to handle user management commands.

    This class encapsulates the argparse configuration for parsing command-line
    arguments related to user operations like adding, removing, and listing users.
    It provides a clean interface to parse command strings and retrieve usage
    information, suitable for use in applications like chat bots.

    Attributes:
        parser (ArgumentParser): The main ArgumentParser instance.
        subparser: The special action object for creating subparsers.
        parser_add (ArgumentParser): The subparser for the 'add' action.
        parser_remove (ArgumentParser): The subparser for the 'remove' action.
        parser_list (ArgumentParser): The subparser for the 'list' action.
    """
    def __init__(self):
        self.parser = ArgumentParser(description="User's management", exit_on_error=False)
        # главный парсер
        self.subparser = self.parser.add_subparsers(
            dest='action', 
            required=True, 
            help='Действие'
            )
        # Парсер для команды "add"
        self.parser_add = self.subparser.add_parser('add', help='Добавить нового пользователя')
        self.parser_add.add_argument('username', type=str.upper, help='Имя нового пользователя')
        self.parser_add.add_argument('tg_id', type=str, help='ID нового пользователя')

        # Парсер для команды "remove"
        self.parser_remove = self.subparser.add_parser('remove', help='Удалить пользователя')
        self.parser_remove.add_argument('username', type=str.upper, help='Имя пользователя')

        # Парсер для команды "list"
        self.parser_list = self.subparser.add_parser('list', help='Список пользователей')


    def get_args(self, args_str: str) -> dict:
        try:
            args = self.parser.parse_args(args_str.split())
            return {k:v for k,v in (vars(args).items()) if v is not None}
        except SystemExit:
            return {'SystemExit': "❌ Invalid arguments"}

    def get_usage(self) -> str:
        return text("➕ Add: `/user add username ID`",  
                    "➖ Remove: `/user remove username`",
                    "📃 List: `/user list`", sep='\n')



class User:
    '''Specific user interface in chat'''
    def __init__(self):
        self.DEFAULT_BOT: str = 'gemini'
        self.DEFAULT_PIC: str = 'glif_pic'
        self.api_factory = APIFactory()
        self.current_bot: BaseAPIInterface = self.api_factory.get('bot',self.DEFAULT_BOT)
        self.current_pic: BaseAPIInterface = self.api_factory.get('pic',self.DEFAULT_PIC)
        self.time_dump = time()
        self.text: str = None
        self.last_msg: dict = None # for deleting messages
        self.media_group_buffer: dict = None ## for media_group_handler
        self.nav_type: str = 'bot'
        

    async def change_context(self, context_name: str) -> str | dict:
        await self.clear()
        if context_name == '◀️':
            return users.context_dict
        
        context= users.get_context(context_name)

        if isinstance(context, dict): # subgroup
            context.setdefault('◀️','◀️')
            return context
        
        output_text = f'Контекст {context_name} добавлен'
        
        if context_name in users.context_dict['🖼️ Image_desc'] and hasattr(self.current_pic,'get_info'):
            output_text += self.current_pic.get_info()

        if isinstance(self.current_bot, (BOTS.GeminiAPI,BOTS.ZenmuxAPI)):
            self.current_bot.reset_chat(context=context)
            return output_text

        else:
            body = {'role':'system', 'content': context}

        self.current_bot.context.append(body)
        return output_text


    async def template_prompts(self, template: str) -> str:
        if template.isdigit():
            for num, prompt_text in enumerate(users.template_prompts.values(),1):
                if num == int(template):
                    break
        else:
            prompt_text = users.template_prompts.get(template)
        output = await self.prompt(prompt_text)
        return escape(output)
    

    async def info(self, delete_prev: bool = False) -> tuple:
        is_gemini = self.current_bot.name in {'gemini'}
        output = text(
            f'🤖 Текущий бот: {self.current_bot.name}',
            f'🧩 Модель: {self.current_bot.current}',
            f'📚 Размер контекста: {len(self.current_bot.context) 
                                    if not is_gemini else self.current_bot.length()}',
            sep='\n')
        if delete_prev:
            await bot.delete_message(**self.last_msg) # type: ignore
        return output, None #self.make_conf_btns()
    
    
    async def change_bot(self, bot_name: str) -> str:
        '''DEPRECATED'''
        self.current_bot = self.api_factory.get('bot',bot_name)
        await self.clear()
        return f'🤖 Смена бота на {self.current_bot.name}'
    

    async def change_model(self, btn_type: str, bot: str, model: str) -> None:
        cbt = f'current_{btn_type}'
        if getattr(self, cbt).name != bot:
            setattr(self, cbt, self.api_factory.get(btn_type, bot))
        if model:
            # getattr(self, cbt).current = model
            getattr(self, cbt).current = getattr(self, cbt).models[int(model)]
        if btn_type == 'bot':
            await self.clear(cmd='wipe')


    def change_state(self, state: str) -> None:
        cbt = getattr(self, f'current_{self.nav_type}')
        if hasattr(cbt, 'states') and (state in cbt.states):
            attr: bool = not cbt.states[state]
            match state:
                case 'proxy':
                    cbt.create_client(attr)
                case _:
                    cbt.states[state] = attr
                    cbt.reset_chat() if hasattr(cbt, 'reset_chat') else None


    def dialogue_router(self, cmd: str) -> str:
        """Router for command actions."""
        cbt = getattr(self, f'current_{self.nav_type}')
        if hasattr(cbt, 'dialogue_api_router'):
            getattr(cbt, 'dialogue_api_router')(cmd.removeprefix('dlg_'))
            return {'dlg_last': 'Удален последний ответ и вопрос из контекста',
                    'dlg_clear': 'Очистка контекста',
                    'dlg_wipe': 'Очистка контекста и системной инструкции'}[cmd]
        else:
            return f'❌ Команда {cmd} отсутствует в {cbt.name}'


    async def utils_router(self, cmd: str, user_id: int) -> str:
        """Router for utils actions."""
        match cmd:
            case 'utils_gemini_list':
                output = await self.current_bot.get_list()
            case 'utils_modify_models':
                output: str = '❌ Используйте формат: `/modify_models gemini Ultra gemini-2.5-ultra`'\
                            'Или `/modify_models gemini remove short_name` для удаления модели'
            case 'utils_context':
                output: str = users.get_current_context(user_id) or '❌ Нет текущего контекста'
                output: str = f'```plaintext\n{output}\n```' if not output.startswith('❌') else output
        return escape(output)
        

    async def clear(self, delete_prev: bool = False, cmd: str | None = None) -> tuple:
        if self.current_bot.name in {'gemini'}:
            status: str = self.current_bot.dialogue_api_router(cmd)
        else:
            ct = self.current_bot.context
            if (len(ct) not in {0,1}) and (ct[0].get('role') == 'system'):
                self.current_bot.context = ct[:1]
                status: str = 'кроме системного'
            else:
                self.current_bot.context.clear()
                status: str = 'полностью'
        if delete_prev:
            await bot.delete_message(**self.last_msg) # type: ignore
        return f'🧹 Диалог очищен {status or ''}', None
    

    def make_multi_modal_body(images: list[dict] | dict, 
                              text: str | None = None, 
                              context: list = None) -> str:
        data = images[0] if isinstance(images, list) else images
        image_str: str = base64.b64encode(data.get('data')).decode()
        if len(image_str) > 180_000:
            print("Слишком большое изображение, сжимаем...")
            image_str = users.resize_image(image_str)
        if context:
            part = f"data:image/jpeg;base64,{image_str}"
            context.extend([
            {
                "role": "user",
                "content": [
                        {"type": "text", "text": text or "Describe this image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": part},
                        },
                    ],
                }
            ])
        return image_str
        

    def encode_multi_modal_body(output) -> dict:
        if isinstance(output, dict):
            ## reve
            used, rem = output.get('credits_used'), output.get('credits_remaining')
            logger.info(caption :=f"x - {used} = {rem}")
            output_dct: dict[str,str] = {'caption': caption, 'photo': output.get('image')}
        else:
            ## open_router
            output_dct: dict[str,str] = {'caption': output.content.strip()}
            for part in output.images:
                if img_data := part.get('image_url'):
                    output_dct['photo'] = img_data.get('url','')

        if raw_data := output_dct.get('photo'):
            if raw_data.startswith('data:'):
                header, b64data = raw_data.split(",", 1)
            else:
                b64data = raw_data
            data = io.BytesIO(base64.b64decode(b64data))
            data.seek(0)
            output_dct['photo'] = BIF(data.getvalue(), "image.png")
        return output_dct


    async def prompt(self, *args) -> str:
        output = await users.queue_manager.enqueue_request(self.current_bot.name, 
                                            self.current_bot.prompt(*args))
        return output


    async def gen_image(self, *args, **kwargs) -> str:
        output = await users.queue_manager.enqueue_request(self.current_pic.name,
                                    self.current_pic.gen_image(*args, **kwargs))
        return output


    async def delete_last_cmd(self) -> None:
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
        self.template_prompts: dict = {
                '💬 Цитата': 'Напиши остроумную цитату. Цитата может принадлежать как реально существующей или существовавшей личности, так и вымышленного персонажа',
                '🤣 Шутка': self.context_dict.get("🤡 Юмор",{}).get("🍻 Братюня",'') + '\nВыступи в роли профессионального стендап комика и напиши остроумную шутку. Ответом должен быть только текст шутки',
                '💡 Факт': 'Выступи в роли профессионального энциклопедиста и напиши один занимательный факт. Ответом должен быть только текст с фактом',
                '🤔 Квиз': '''Выступи в роли профессионального энциклопедиста и напиши три вопроса для занимательного квиза. 
                            Уровень вопросов: Старшая школа. Ответом должен быть только текст с тремя вопросами без ответов''',
                '🤓 QuizPlease': '''Выступи в роли профессионального ведущего quiz - вечеринок. Напиши 5 вопросов по теме кино и сериалы. 
                                Вопросы должны быть минимум продвинутого уровня, рассчитанные на искушённых киноманов.''',
                '📝 Промпт': ''''Write 3 interesting and unusual prompts in different visual styles.
                            First, think through the main idea of the picture and then realize the visual storytelling that will be revealed by that one prompt.
                            It must consist a sarcastic, ironic and brutal plot with black humor, showing the situation.
                            Wrap each prompt in plaintext block. Max tokens 500.''',
                '⚖️ Правда': self.context_dict.get("🤡 Юмор",{}).get("🍻 Братюня",'') + (
                            '\nНапиши непопулярное мнение на твое усмотрение на основе научных данных.'
                            'Желательно такое, чтобы мир прям наизнанку и пиши развернутый аргументированный ответ')
            }
        self.help = self.create_help()
        self.buttons: dict = {
                'Меню':'menu', 
                'Добавить контекст':'change_context', 
                'Быстрые команды':'template_prompts',
            }
        self.simple_cmds: set = {'clear', 'info'}
        self.PARSE_MODE = ParseMode.MARKDOWN_V2
        self.all_abs: dict[str, str] = {'bot': '🧩', 'pic': '🖼️'}
        # self.image_arg_parser = ImageGenArgParser()



    @lru_cache(maxsize=None)
    def get(self, user_id: int) -> User:
        return self._user_instances.setdefault(user_id, User())
    

    def resize_image(self, image: str, max_b64_length=180_000, max_file_size_kb=450) -> str:
        max_file_size_bytes = max_file_size_kb * 1024
        img = Image.open(io.BytesIO(base64.b64decode(image)))
        # Функция для сжатия и конвертации изображения в Base64
        def image_to_base64(img, quality=85) -> tuple[str, bytes]:
            buffer = io.BytesIO()
            img.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img_b64 = base64.b64encode(buffer.getvalue()).decode()
            return img_b64, buffer.getvalue()
        
        # Рекурсивная функция для сжатия изображения
        def recursive_compress(img, quality) -> str | tuple[str, bytes]:
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
            text('1. 🧑‍💼 User Management (/user or /users):',
                 '🔹 Add new user: /user add USERNAME TG_ID',
                 '🔹 Remove existing user: /user remove USERNAME',
                 '🔹 List all users: /user list', 
                  sep='\n'),
            text('2. 🗂️ Context:',
                 '🔹 -i: Get context_body info',
                 '🔹 -a: Add new context',
                 '🔹 -r: Remove existing context',
                 'Usage:',
                 '🔹 /context [-i | -r] [context_name | c OR current]',
                 '🔹 /context [-a] context_name | context_body', 
                 sep='\n'),
            text('3. 🖼️ Generate Image:',
                 '🔹 Equal commands: /image or /i or /I',
                 '🔹 Usage: /image your_prompt',
                 '🔹 Acceptable ratio size: 9:16, 3:4, 1:1, 4:3, 16:9', 
                 sep='\n'),
        ]
        return ExpandableBlockQuote(text(*help_items_simple, sep='\n')).as_markdown()


    async def check_and_clear(self, 
                              message: Message | CallbackQuery, 
                              type_prompt: str, 
                              username: str = '') -> User:
        user: User = self.get(message.from_user.id)  # type: ignore
        if type_prompt in {'callback','tts'}:
            return user
        elif type_prompt in ['gen_image']:
            logger.info(f'{username or message.from_user.id}: "{message.text[:100]}"') # type: ignore
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
        if username:
            logger.info(f'{username}: {type_prompt[:100]}...')
         
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


    def get_current_context(self, user) -> str | None:
        ct = user.current_bot.context
        if len(ct) and ct[0].get('role') == 'system':
            return ct[0].get('content')
        # return 'No current context'
    

    def create_menu_kb(self, user: User, target: str, btn_act: str | None = None) -> dict:
        """Creates keyboard markup with buttons for menu navigation
        
        Args:
            target: Target menu section
            btn_act: Current button action
            
        Returns:
            tuple: Headline text and keyboard markup
        """
        builder = InlineKeyboardBuilder()
        
        if target in self.all_abs or target.startswith('switch_'):
            user.nav_type = target.replace('switch_','')
        target_menu: dict = self.menu[target]
        cb = getattr(user, f'current_{user.nav_type}')
        if user.nav_type in self.all_abs and target not in {'main', 'switch', 'utils', 'cmd'}:
            headline: str = f'Текущая модель:\n🤖 {cb.name}\n'\
                            f'{self.all_abs.get(user.nav_type)} {cb.current}'\
                            f'{'\n📐' + cb.image_size if user.nav_type == 'pic' else ''}'
        else:
            headline: str = target_menu['text']

        btns_list: list[dict] = target_menu["buttons"]
        for num, btn in enumerate(btns_list):
            state, select = btn.get('state'), str(num) if btn.get('select') else None
            # if state in self.state_btns and not hasattr(cb, state):
            if (state in self.state_btns) and (state not in getattr(cb, "states", ())):
                continue
            builder.button(
                text=self._add_emoji_prefix(cb, btn_act, state, btn.get('select')) + btn['text'], 
                callback_data=MenuCallbacks(
                    target=btn.get('target', target),
                    act=state or select or 'go',
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
        if state and hasattr(cb, 'states'):
            return '✅ ' if cb.states[state] else '❌ '
        # Handle model selection buttons
        if select:
            if select.startswith('ratio'):
                # Handle image_size selection buttons
                return '✅ ' if select.removeprefix('ratio_') in {cb.image_size, btn_act} else ''
            
            return '✅ ' if select in {cb.current, btn_act} else ''
        return ''


    def edit_json_settings(self) -> None:
        with open('./settings.json', 'w', encoding="utf-8") as f:
            json.dump(self.menu, f, ensure_ascii=False, indent=4)


    def modify_models(self, bot: str, nm_name: str, new_model: str) -> str:
        """Modify models for bot"""
        if nm_name == 'remove':
            try:
                self.menu[bot]["buttons"].remove(
                    next((m for m in self.menu[bot]["buttons"] if m["text"] == new_model), None)
                    )
                output = f'✂️ Модель {new_model} удалена из {bot}'
            except ValueError:
                return f'❌ Модель {new_model} не найдена в {bot}'

        else:
            self.menu[bot]["buttons"] = [
                *self.menu[bot]["buttons"][:-1],
                {'text': nm_name, 'select': new_model},
                self.menu[bot]["buttons"][-1]]
            output = f'✅ В {bot} добавлена модель {nm_name}'
        self.edit_json_settings()
        self._user_instances.clear()  # Clear user instances to refresh menu
        return output


    async def remove_kb_for_users(self):
        from aiogram.types import ReplyKeyboardRemove
        for tg_id, username in self.db.get_list():
            await bot.send_message(
                chat_id=tg_id,
                text="Технические сообщение, старая клавиатура убрана",
                reply_markup=ReplyKeyboardRemove()
            )
            logger.info(f"OK: {username}")
            await asyncio.sleep(0.3)


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
        await user.delete_last_cmd()


    @dp.message(Command(commands=["context"]))
    async def context_handler(message: Message, username: str, command: CommandObject):
        '''Handles the context management commands for the bot.'''
        if username != 'ADMIN':
            return await message.reply("You don't have admin privileges")

        cur_cont: str | None = users.get_current_context(users.get(message.from_user.id)) # type: ignore
        if not command.args and cur_cont:
            arg, prompt_body = ('-i', 'c')
        elif command.args:
            arg, prompt_body = command.args.split(maxsplit=1)
        else:
            arg, prompt_body = '', ''

        if arg == '-i':
            if prompt_body in ['c', 'current']:
                text: str = cur_cont or 'No current context'
            else:
                text: str = str(users.get_context(prompt_body)) or 'Context name not found'
            text = f'```plaintext\n{text}\n```'
            await message.reply(**users.set_kwargs(escape(text))) # type: ignore
        
        elif arg == '-r' and prompt_body in users.context_dict:
            users.context_dict.pop(prompt_body)
            with open('./prompts.json', 'w', encoding="utf-8") as f:
                json.dump(users.context_dict, f, ensure_ascii=False, indent=4)
            await message.reply(f"Context {prompt_body} removed successfully.")
        
        elif arg == '-a' and prompt_body.count('|') == 1:
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
            text = escape("Usage: `/context [-i/-r/-a] prompt_name [| prompt]`")
            await message.reply(**users.set_kwargs(text)) # type: ignore
            return
        
        await bot.delete_message(message.chat.id, message.message_id)
        

    @dp.message(Command(commands=["user","users"]))
    async def user_management_handler(message: Message, username: str, command: CommandObject):
        """
        Handles the addition / removal of users based on the command received.
        """
        if username != 'ADMIN':
            return await message.reply("You don't have admin privileges")
        
        parser = UsersArgParser()
        if not command.args:
            output = parser.get_usage()
        else:
            dict_args = parser.get_args(command.args)
            name = dict_args.get('username')
            match dict_args['action']:
                case 'add':
                    if users.db.check_username(name):
                        output = f"❌ User {name} already exists."
                    elif users.db.check_tg_id(dict_args['tg_id']):
                        output = f"❌ This TG ID {dict_args['tg_id']} already exists."
                    else:
                        users.db.add_user(name, dict_args['tg_id'])
                        output = f"✅ User {name} added."
                case 'remove':
                    if users.db.check_username(name):
                        users.db.remove_user(name)
                        output = f"✅ {name} removed."
                    else:
                        output = f"❌ {name} not found."
                case 'list':
                    lst = users.db.get_list()
                    output = as_numbered_list(*[f'{v[1]}: {v[0]}' for v in lst]).as_html()
                    
        await message.reply(output)


    @dp.message(Command(commands=["info","clear","change_context"]))
    async def short_command_handler(message: Message):
        await Handlers.reply_kb_command(message)


    @dp.message(Command(commands=["modify_models"]))
    async def modify_models_handler(message: Message, username: str, command: CommandObject):
        if username != 'ADMIN':
            await message.reply("You don't have admin privileges")
            return
        
        args: list = message.text.split(maxsplit=3) if command.args else [] # type: ignore
        logger.info(args)
        if len(args) != 4:
            output: str = '❌ Используйте формат:\n`/modify_models gemini Ultra gemini-2.5-ultra`'\
                    '\nИли\n`/modify_models gemini remove short_name` для удаления модели'
            return await message.reply(output, parse_mode=ParseMode.MARKDOWN_V2)
            
        await message.reply(users.modify_models(*args[1:])) # type: ignore


    @dp.message(Command(commands=["image", "i","I"]))
    async def image_gen_handler(message: Message, username: str, command: CommandObject):
        user = await users.check_and_clear(message, "gen_image", username)
        if command.args is None:
            return await message.reply(
                'Введите промпт для генерации картинки, например: /i your_prompt'
                )
            
        async with ChatActionSender.upload_photo(chat_id=message.chat.id, bot=bot):
            image_info: dict | str = await user.gen_image(command.args)
        if isinstance(image_info, str) and image_info.startswith('❌'):
            await message.answer(image_info)
        else:
            await message.answer_photo(**image_info) # type: ignore


    @dp.message(Command(commands=["tts"]))
    async def generate_audio_story(message: Message, username: str, command: CommandObject):
        user = await users.check_and_clear(message, 'tts', username)
        if command.args is None:
            return await message.reply("Отсутствует текст")

        async with ChatActionSender.record_voice(chat_id=message.chat.id, bot=bot):
            link = await user.current_bot.tts(command.args)
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
    async def media_group_handler(message: Message, username: str):
        data_info = getattr(message, message.content_type, None)
        mime_type = getattr(data_info, 'mime_type', 'image/jpeg')
        data_type = mime_type.split('/')[0]
        user = await users.check_and_clear(message, data_type, username)
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
    async def photo_handler(message: Message, username: str):
        user = await users.check_and_clear(message, 'image', username)
        if (user.current_bot.name not in {'gemini', 'open_router', 'zenmux'}):
            await user.change_model('bot','gemini')
            await message.reply(f"Выбран {user.current_bot.name}")

        async with ChatActionSender.typing(chat_id=message.chat.id, bot=bot):
            tg_photo = await bot.download(message.photo[-1].file_id)  # type: ignore
            output = await user.prompt(user.text, [{'data': tg_photo.getvalue(), 
                                                    'mime_type': 'image/jpeg'}])
            if isinstance(output, str):
                await users.send_split_response(message, output)
            elif isinstance(output, dict):
                if (caption := output.get('caption')) and len(caption) < 1024:
                    await message.answer_photo(**output) # type: ignore
                else:
                    await message.answer_photo(photo=output.get('photo'))
                    await users.send_split_response(message, caption)


    @dp.message(F.content_type.in_({'voice','video_note','video','document'}))
    async def data_handler(message: Message, username: str):
        data_info = getattr(message, message.content_type, None)
        mime_type = getattr(data_info, 'mime_type', None)
        data_type = mime_type.split('/')[0] # type: ignore
        user = await users.check_and_clear(message, data_type, username)
        if user.current_bot.name not in {'gemini'}:
            await user.change_model('bot','gemini')

        # await message.reply(f"{data_type.capitalize()} получено! Ожидайте ⏳")
        async with ChatActionSender.typing(chat_id=message.chat.id, bot=bot):
            data = await bot.download(data_info.file_id) # type: ignore
            output = await user.prompt(user.text, [{'data': data.getvalue(), 'mime_type': mime_type}])
            if isinstance(output, str):
                await users.send_split_response(message, output)
            else:
                await message.answer_photo(**output) # type: ignore



    @dp.message(F.text.startswith('/') | F.text.casefold().startswith('help'))
    async def unknown_handler(message: Message):
        await bot.delete_message(message.chat.id, message.message_id)
        await message.answer(**users.set_kwargs()) # type: ignore


    @dp.message(F.content_type.in_({'text'}))
    async def text_handler(message: Message, username: str):
        user = await users.check_and_clear(message, 'text', username)
        async with ChatActionSender.typing(chat_id=message.chat.id, bot=bot):
            output = await user.prompt(user.text)
            if isinstance(output, str):
                await users.send_split_response(message, output)
            elif isinstance(output, dict):
                # await message.answer_photo(**output)
                if (caption := output.get('caption')) and len(caption) < 1024:
                    await message.answer_photo(**output) # type: ignore
                else:
                    await message.answer_photo(photo=output.get('photo'))
                    await users.send_split_response(message, caption)
            else:
                await message.answer(str(output))



class Callbacks:

    @dp.callback_query(CallbackClass.filter(F.cb_type.contains('change')))
    async def change_callback_handler(query: CallbackQuery, callback_data: CallbackClass):
        user = await users.check_and_clear(query, 'callback')
        output = await getattr(user, callback_data.cb_type)(callback_data.name)
        # is_final_set = isinstance(output, str) and callback_data.name != '◀️'
        if isinstance(output, str) and callback_data.name not in {'◀️','🏠'}:
            await query.message.edit_text(output) # type: ignore
            await user.delete_last_cmd()
        else:
            if callback_data.name == '🏠':
                kwargs = users.create_menu_kb(user, "main")
                await query.message.edit_text(**kwargs) # type: ignore
            else:
                reply_markup = users.create_inline_kb(output, 'change_context')
                await query.message.edit_reply_markup(reply_markup=reply_markup) # type: ignore


    @dp.callback_query(MenuCallbacks.filter(F.act == "go"))
    async def menu_callback_go(query: CallbackQuery, callback_data: MenuCallbacks):
        user = await users.check_and_clear(query, 'callback')
        target = callback_data.target
        if target == 'exit':
            await query.message.delete() # type: ignore
            return
        elif target == 'context':
            cur_bot = user.current_bot
            kwargs: dict = {
                'text': f'Текущая модель:\n🤖 {cur_bot.name}\n🧩 {cur_bot.current}',
                'reply_markup':users.create_inline_kb(users.context_dict, 'change_context')
            }
        elif target.startswith('cmd_'):
            await query.answer()
            async with ChatActionSender.typing(chat_id=query.message.chat.id, bot=bot): # type: ignore
                output: str = await user.template_prompts(target.removeprefix('cmd_'))
            kwargs: dict = users.set_kwargs(output)

        elif target.startswith('dlg_'):
            kwargs: dict = {'text': user.dialogue_router(target)}
        elif target.startswith('utils_'):
            kwargs: dict = users.set_kwargs(await user.utils_router(target, user))  # type: ignore
        else:
            kwargs: dict = users.create_menu_kb(user, target)
        await query.message.edit_text(**kwargs) # type: ignore
        await query.answer()
        

    @dp.callback_query(MenuCallbacks.filter(F.act != "go"))
    async def menu_callback_state_select(query: CallbackQuery, callback_data: MenuCallbacks):
        user = await users.check_and_clear(query, 'callback')
        target, act = callback_data.target, callback_data.act
        if act in users.state_btns:
            user.change_state(act)
        elif target == 'ratio':
            ratio = BaseAPIInterface.get_models(users.menu['ratio'])[int(act)].removeprefix('ratio_')
            user.current_pic.image_size = ratio
            if hasattr(user.current_bot, 'image_size'):
                user.current_bot.image_size = ratio
        else:
            await user.change_model(user.nav_type, target, act)
        with suppress(Exception):
            await query.message.edit_text(  # type: ignore
                **users.create_menu_kb(user, target, act)
                )
        await query.answer()



async def main() -> None:
    await bot.set_my_commands([
        BotCommand(command="/menu", description="🏠 Меню"),
        BotCommand(command="/change_context", description="✍️ Добавить контекст"),
        BotCommand(command="/clear", description="🧹 Очистить диалог"),
        BotCommand(command="/info", description="📚 Вывести инфо"),
    ])
    # await users.remove_kb_for_users()
    await dp.start_polling(bot)



if __name__ == "__main__":
    try:
        logger.info('🚀 Start polling')
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("⚠️  Stop polling")