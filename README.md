# Bard Knows Telegram Bot

A versatile Telegram bot that provides access to various AI models for text generation, image analysis, and multimedia processing.

## Features

- Multiple AI model support (Gemini, Nvidia, Groq, Mistral, Together, Glif)
- Image generation and analysis
- Voice and video message processing
- Custom context management
- Admin controls for user access
- Rate limiting for API requests

## Commands

### User Commands
- `/start` - Initialize bot interaction
- `/help` - Show available commands
- `/i` or `/image` - Generate images with custom parameters
  ```
  Usage examples:
  • Basic: /i your prompt here
  • With aspect ratio: /i your prompt here --ar 9:16
  • With model selection: /i your prompt here --m 1.1
  • Combined: /i your prompt here --ar 9:16 --m ultra
  ```
- `/clear` - Clear conversation context
- `/info` - Display current bot settings

### Admin Commands
- `/add_user [ID] [Name]` - Add new user access
- `/remove_user [Name]` - Remove user access
- `/context` - Manage prompt contexts
  ```
  Usage:
  /context [-i/-r/-a] context_name [| context_body]
  -i: Get context info
  -r: Remove context
  -a: Add new context
  ```
- `/conf` - Configure bot settings
  ```
  Usage examples:
  • Search on in gemini: /conf --es 1
  • Search off in gemini: /conf --es 0
  • List Gemini's models: /conf --nm list
  ```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API keys:
Create an `.env` file with environment variables:
```env
TELEGRAM_API_KEY=your-telegram-token
GEMINI_API_KEY=your-gemini-key
NVIDIA_API_KEY=your-nvidia-key
GROQ_API_KEY=your-groq-key
MISTRAL_API_KEY=your-mistral-key
TOGETHER_API_KEY=your-together-key
GLIF_API_KEY=your-glif-key
FAL_API_KEY=your-fal-key
```

3. Run the bot:
```bash
python app.py
```
OR with docker-compose
```bash
docker-compose up
```

## Usage

1. Start with `/start` command
2. Send text messages for AI responses
3. Send images for analysis
4. Send voice/video messages for processing
5. Use keyboard buttons for quick actions:
   - Change context
   - Switch AI models
   - Quick commands
   - Clear context
   - Show info

## Support

For issues or suggestions, please check the logs at `./app.log` or contact the administrator.