# Zyon AI Userbot (Groq Edition)

A powerful Telegram userbot largely based on Python, integrated with Groq AI for fast, high-quality conversational capabilities. This bot features personality modes, voice chat capabilities, and extensive dataset integrations.

## 🚀 Features

- **Groq AI Integration**: Utilizes Groq's LPU inference engine for rapid responses.
- **Personality Modes**:
  - ✨ **Standard**: Helpful and friendly.
  - 🌶️ **Edgy/Limitless**: More unrestricted and bold personalities.
  - 🧠 **Mature/RealTalk**: Engages in deeper, more authentic conversations.
  - 🎵 **Voice Chat**: Play music and stream audio in Telegram voice chats.
- **Dataset Enhancements**:
  - Uses specific datasets (UltraChat, RealTalk, etc.) to seed conversations and improve response quality.
- **Memory Management**: SQLite-based memory to remember user context and preferences.
- **Anti-Repetition**: Smart logic to prevent the AI from looping or repeating phrases.
- **Image Capabilities** (Experimental): Integration for image processing (Note: Groq textual models do not process images directly; this feature requires a vision-capable model setup).

## 🛠️ Installation

### Prerequisites

- Python 3.9+
- FFmpeg (for voice chat functionality)
- Telegram API ID and Hash (from [my.telegram.org](https://my.telegram.org))
- Groq API Key
- Tavily API Key (for web search)

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sagar-xs69/zyon-ai-userbot-groq.git
   cd zyon-ai-userbot-groq
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment:**
   Copy `.env.example` to `.env` and fill in your details:
   ```bash
   cp .env.example .env
   ```
   *   `TELEGRAM_API_ID`
   *   `TELEGRAM_API_HASH`
   *   `GROQ_API_KEY`
   *   `TAVILY_API_KEY`
   *   `OWNER_ID`

4. **Run the bot:**
   ```bash
   python vc.py
   ```

## 🐳 Docker Support

A `Dockerfile` is included for easy deployment.

```bash
docker build -t zyon-userbot .
docker run -d --env-file .env zyon-userbot
```

## ⚠️ Disclaimer

This project is for educational purposes. Please abide by Telegram's Terms of Service. Generating explicit or harmful content is not supported by the primary providers and may lead to account restrictions.

## 🤝 Contributing

Contributions are welcome! Please submit a pull request.
