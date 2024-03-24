# README.md

## Bard Knows Telegram Bot

This Python Telegram bot can provide users with quotes, jokes, facts, historical events, and trivia questions.

## How to use

1. Start the bot by running the following command:

Create a virtual environment and install the dependencies with the following command:

```bash
pip install -r requirements.txt
```
```python
python app.py
```

2. For getting access to the bot, it must first add telegram user ID to sqlite3 database, which was created after running bot (telegram user ID write in logs)
```
>>> /start
"Доступ запрещен. Обратитесь к администратору"
```

2. Type the name of the category you want to receive information from. For example, to receive a random quote, type "Цитата".

3. The bot will send you the requested information.

## Examples

```
>>> Цитата
"Жизнь слишком коротка, чтобы тратить ее на маленькие мысли." Марк Аврелий

>>> Шутка
Что общего у слона и балерины? Они оба на ногах стоят.

>>> Факт
Самая большая страна в мире по площади - Россия.

>>> Квиз
Какой самый большой океан на Земле?
```

## Set api keys

Place the api_keys.json file in the project. 
Schema:
```json
{
    "api_key_bot": "your-telegram-token",
    "api_key_riddle": "your-api-token-from: api-ninjas.com", 
    "gemini": "your-gemini-api-key"
}
```

## Support

If you have any questions or suggestions, please feel free to contact me.