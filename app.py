import asyncio
import random
from datetime import datetime

import aiohttp
import json
from aiogram import Bot, Dispatcher, F, types
from aiogram.enums import ParseMode
from aiogram.utils.keyboard import ReplyKeyboardBuilder
from translate import Translator

with open("api_keys.json", "r") as f:
    api_keys = json.load(f)

date_raw = datetime.now()

dp = Dispatcher()

arr_api_data = {
    'Случайно': 'random',
    'Цитата': 'quotes',
    'Шутка': 'jokes',
    'Факт': 'facts',
    'Событие в этот день': 'historicalevents',
    'Квиз': 'trivia'
}

builder = ReplyKeyboardBuilder()
for display_text in arr_api_data:
    builder.button(text=display_text)
builder.adjust(3, 2)


async def handler(text: str) -> str:
    if text == 'random' or text not in arr_api_data.values():
        end_tab_text = random.choice(list(arr_api_data.values())[1:])
        is_random = True
    else:
        end_tab_text = text
        is_random = False


    async def get_api_info(end_tab: str, params: dict = {}) -> str:
        url = f"https://api.api-ninjas.com/v1/{end_tab}"
        headers = {'X-Api-Key': api_keys["api_key_riddle"]}
        
        async with aiohttp.ClientSession(headers=headers) as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                if isinstance(data, list):
                    data = random.choice(data)

                if end_tab == 'quotes':
                    info = f'{data[end_tab[:-1]]} © {data["author"]} ▶ Категория {data["category"]}'
                elif end_tab == 'trivia':
                    info = f'{data["question"]} ▶ {data["answer"]}'
                elif end_tab == 'historicalevents':
                    info = f'On this day in {data["year"]}, {data["event"]}'
                else:
                    info = data[end_tab[:-1]]
                return info

    print(end_tab_text)
    params = {'month': date_raw.month,
              'day': date_raw.day} if end_tab_text == 'historicalevents' else {}
    info = await get_api_info(end_tab_text, params)

    translator = Translator(to_lang="ru")
    translation = translator.translate(info)

    return f"{end_tab_text} ▶ {translation}" if is_random else translation


@dp.message(F.content_type.in_({'text'}))
async def echo_handler(message: types.Message | types.KeyboardButtonPollType):
    input_data = arr_api_data.get(message.text, 'random')
    if len(input_data) == 0:
        await message.reply("Сначала выберите тему ниже")
        return

    try:
        output = await handler(input_data)
        await message.answer(output, reply_markup=builder.as_markup())
    except Exception as e:
        await message.answer(str(e))
        return


async def main() -> None:
    bot = Bot(token=api_keys["api_key_bot"], parse_mode=ParseMode.HTML)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
