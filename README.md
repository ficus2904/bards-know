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

2. Type the name of the category you want to receive information from. For example, to receive a random quote, type "Цитата". To receive a trivia question, type "Квиз".

3. The bot will send you the requested information.

## Examples

```
>>> /start
>>> Цитата
"Жизнь слишком коротка, чтобы тратить ее на маленькие мысли." © Марк Аврелий ▶ Категория Жизнь

>>> Шутка
Что общего у слона и балерины? Они оба на ногах стоят.

>>> Факт
Самая большая страна в мире по площади - Россия.

>>> Событие в этот день
19 сентября 1812 года состоялось Бородинское сражение.

>>> Квиз
Какой самый большой океан на Земле? ▶ Тихий океан
```

## Set api keys

Place the api_keys.json file in the project. 
Schema:
```json
{
    "api_key_bot": "your-telegram-token",
    "api_key_riddle": "your-api-token-from: api-ninjas.com"
}
```

## Support

If you have any questions or suggestions, please feel free to contact me.