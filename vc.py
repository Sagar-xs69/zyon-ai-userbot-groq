"""
Zyon AI Userbot (Groq Edition)
==============================
A powerful Telegram userbot integrated with Groq AI for fast, high-quality responses.
Features:
- Personality Modes (Standard, Edgy, Mature)
- Voice Chat Music & Streaming
- Dataset-Enhanced Responses
- Memory Management
- Anti-Repetition Logic
- Image Capabilities (Experimental)

Author: Zyon Team
License: MIT
"""

import os
import sys
import asyncio
import logging
import random
import re
import json
import time
import shutil
import glob
from pathlib import Path
from datetime import datetime
import traceback
from typing import Optional, Union, List, Dict

# Third-party imports
from dotenv import load_dotenv
from telethon import TelegramClient, events, functions, types
from telethon.sessions import StringSession
from telethon.errors import SessionPasswordNeededError, FloodWaitError
from groq import Groq, AsyncGroq
from tavily import TavilyClient
import aiohttp
import aiofiles

# Dataset & Enhancement Imports
from dataset_manager_minimal import get_minimal_dataset_manager
from emoji_enhancer import suggest_contextual_emojis

# Image Processing Imports
try:
    from PIL import Image, ImageEnhance, ImageFilter
    import pytesseract
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️  Pillow/Tesseract not installed. Image features disabled.")

# Voice Chat Imports
try:
    from pytgcalls import PyTgCalls
    from pytgcalls.types import MediaStream
    from pytgcalls.types.groups import GroupCallConfig
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    print("⚠️  py-tgcalls not installed. Voice features disabled.")

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
API_ID = int(os.getenv("TELEGRAM_API_ID"))
API_HASH = os.getenv("TELEGRAM_API_HASH")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OWNER_ID = int(os.getenv("OWNER_ID"))
SESSION_NAME = "zyon_userbot"
Groq_Model = "llama3-70b-8192"  # Powerful, fast model

# Initialize Clients
groq_client = AsyncGroq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None
client = TelegramClient(SESSION_NAME, API_ID, API_HASH)

if VOICE_AVAILABLE:
    call_client = PyTgCalls(client)

# Global State
current_personality = "standard"  # standard, edgy, mature
memory_enabled = True
voice_active = False

# --- Database & Memory Management ---

class MemoryManager:
    def __init__(self, db_path="zyon_memory.db"):
        self.db_path = db_path
        self.user_cache = {}
        self.cache_max_size = 100
        self._cache_lock = asyncio.Lock()
        self._db_lock = asyncio.Lock()
        self.db_available = False
        try:
            self.init_db()
            self.db_available = True
            print("✅ Memory database initialized successfully")
        except Exception as e:
            print(f"⚠️ Memory database unavailable: {e}")
            print("📹 Running in memory-only mode (data won't persist)")
            self.db_available = False
            
    def init_db(self):
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            # User profile table
            c.execute('''CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                nickname TEXT,
                interaction_count INTEGER DEFAULT 0,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT,
                personality_preferences TEXT
            )''')
            
            # Conversation history table
            c.execute('''CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                role TEXT,
                content TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(user_id)
            )''')
            
            # Facts/Knowledge table
            c.execute('''CREATE TABLE IF NOT EXISTS facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                fact_content TEXT,
                category TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            conn.commit()

    async def get_user_memory(self, user_id: int) -> Dict:
        if not self.db_available:
            return self.user_cache.get(user_id, {})
            
        async with self._cache_lock:
            if user_id in self.user_cache:
                return self.user_cache[user_id]
        
        # Load from DB
        try:
            import sqlite3
            def _fetch():
                with sqlite3.connect(self.db_path) as conn:
                    conn.row_factory = sqlite3.Row
                    c = conn.cursor()
                    c.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
                    row = c.fetchone()
                    if row:
                        return dict(row)
                    return None
            
            data = await asyncio.to_thread(_fetch)
            if data:
                async with self._cache_lock:
                    self.user_cache[user_id] = data
                return data
            return {}
        except Exception as e:
            logger.error(f"DB Read Error: {e}")
            return {}

    def _update_user_memory_sync(self, user_id: int, **kwargs):
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            c = conn.cursor()
            # Check if user exists
            c.execute("SELECT 1 FROM users WHERE user_id = ?", (user_id,))
            exists = c.fetchone()
            
            if not exists:
                cols = ["user_id"] + list(kwargs.keys())
                vals = [user_id] + list(kwargs.values())
                placeholders = ",".join(["?"] * len(vals))
                col_str = ",".join(cols)
                c.execute(f"INSERT INTO users ({col_str}) VALUES ({placeholders})", vals)
            else:
                set_str = ",".join([f"{k} = ?" for k in kwargs.keys()])
                vals = list(kwargs.values()) + [user_id]
                c.execute(f"UPDATE users SET {set_str} WHERE user_id = ?", vals)
            conn.commit()

    async def update_user_memory(self, user_id: int, **kwargs):
        if not self.db_available:
            return
            
        await asyncio.to_thread(self._update_user_memory_sync, user_id, **kwargs)
        async with self._cache_lock:
            self.user_cache.pop(user_id, None)

    async def add_history(self, user_id: int, role: str, content: str):
        if not self.db_available: return
        
        def _add():
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                c = conn.cursor()
                c.execute("INSERT INTO history (user_id, role, content) VALUES (?, ?, ?)",
                         (user_id, role, content))
                # Keep only last 50 messages per user to save space
                c.execute("""
                    DELETE FROM history WHERE id IN (
                        SELECT id FROM history WHERE user_id = ? 
                        ORDER BY id DESC LIMIT -1 OFFSET 50
                    )
                """, (user_id,))
                conn.commit()
        
        await asyncio.to_thread(_add)

    async def get_recent_history(self, user_id: int, limit=10) -> List[Dict]:
        if not self.db_available: return []
        
        def _get():
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                c = conn.cursor()
                c.execute("SELECT role, content FROM history WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                         (user_id, limit))
                return [dict(r) for r in c.fetchall()][::-1]
        
        return await asyncio.to_thread(_get)

memory_manager = MemoryManager()

# --- Anti-Repetition & Personality ---

class AntiRepetitionManager:
    def __init__(self):
        self.user_recent_responses = {} # user_id -> list of hashes/short texts
        self.user_conversation_seeds = {} # user_id -> random seedint

    def should_vary_response(self, user_id: int, response: str) -> bool:
        """Check if response is too similar to recent ones"""
        if user_id not in self.user_recent_responses:
            self.user_recent_responses[user_id] = []
            self.user_conversation_seeds[user_id] = random.randint(1, 1000)
            return False
            
        # Simple similarity check (can be improved)
        recent = self.user_recent_responses[user_id]
        
        # Exact match
        if response in recent:
            return True
            
        # Update history
        recent.append(response)
        if len(recent) > 5:
            recent.pop(0)
            
        return False
        
    def get_anti_repetition_prompt(self, user_id: int) -> str:
        self.user_conversation_seeds[user_id] += 1
        seed = self.user_conversation_seeds[user_id]
        recent = list(self.user_recent_responses[user_id])
        if not recent: return ""
        prompt = f"\n--- ANTI-REPETITION DIRECTIVE (CRITICAL) ---\nCONVERSATION SEED: {seed}\nYou MUST avoid repeating your recent responses. Your last few responses were:\n"
        for i,r in enumerate(recent[-3:],1):
            preview = r[:100]+"..." if len(r)>100 else r
            prompt += f"\nResponse {i}: {preview}"
        prompt += "\n\nMANDATORY REQUIREMENTS:\n1. Your response MUST be completely different from the above responses\n2. Use different sentence structures, vocabulary, and approach\n3. If the user asks the same question, acknowledge it differently and provide a fresh perspective\n4. Vary your personality expression - be more casual, formal, humorous, or serious than before\n5. Use different examples, analogies, or explanations\n6. If stuck in a topic loop, explicitly acknowledge it: \"I notice we keep circling back to this - let me try a different angle\"\n"
        return prompt

anti_repetition = AntiRepetitionManager()

# --- Auto-Reload System ---

class AutoReloader:
    def __init__(self, watch_file="vc.py"):
        self.watch_file = Path(watch_file)
        self.last_mtime = self.watch_file.stat().st_mtime if self.watch_file.exists() else 0
        self.checking = False
        
    async def start(self, client):
        self.checking = True
        logger.info(f"🔄 Auto-reloader started for {self.watch_file}")
        asyncio.create_task(self.check_for_updates(client))
        
    async def check_for_updates(self, client):
        """Check if file was modified and restart if needed"""
        while self.checking:
            await asyncio.sleep(2)  # Check every 2 seconds
            try:
                if self.watch_file.exists():
                    current_mtime = self.watch_file.stat().st_mtime
                    if current_mtime > self.last_mtime:
                        logger.info("🔄 Code update detected! Restarting...")
                        print("\n🔄 NEW CODE UPLOADED - RESTARTING AUTOMATICALLY...")
                        await self.graceful_restart(client)
            except Exception as e:
                logger.error(f"Auto-reload check error: {e}")

    async def graceful_restart(self, client):
        """Restart the script gracefully"""
        try:
            # Send message to owner if possible
            await client.send_message("me", "🔄 Zyon is restarting due to code update...")
        except:
            pass
            
        os.execv(sys.executable, ['python'] + sys.argv)

reloader = AutoReloader()

# --- Helper Functions ---

async def perform_web_search(query: str) -> str:
    """Perform web search using Tavily"""
    if not tavily_client:
        return "Web search unavailable (API key invalid)."
    try:
        response = tavily_client.search(query=query, search_depth="basic")
        results = response.get("results", [])
        if not results:
            return "No results found."
        
        summary = "Web Search Results:\n"
        for r in results[:3]:
            summary += f"- {r['title']}: {r['content']}\n"
        return summary
    except Exception as e:
        logger.error(f"Search error: {e}")
        return f"Search failed: {e}"

async def analyze_image_content(image_path: str, prompt: str = "Describe this image in detail.") -> str:
    """Analyze image using vision model (Placeholder - Groq textual models don't support vision directly yet)"""
    # Note: Groq's LLaVA or similar would be needed. This is a placeholder for future integration
    # For now, we use OCR if text is requested
    if not PIL_AVAILABLE:
        return "[Image analysis unavailable - Missing dependencies]"
        
    try:
        img = Image.open(image_path)
        # Simple OCR as fallback
        text = pytesseract.image_to_string(img)
        if text.strip():
            return f"[Image OCR Result]: {text[:500]}..."
        return "[Image received - Visual analysis not yet supported by current model]"
    except Exception as e:
        return f"[Image processing error: {e}]"

# --- Main Logic ---

async def get_system_prompt(user_id: int, user_message: str) -> str:
    """Generate dynamic system prompt based on context and personality"""
    
    # 1. Base Identity
    base = f"""You are Zyon, an elite AI userbot for Telegram.
Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Owner ID: {OWNER_ID}
"""

    # 2. Personality Injection
    if current_personality == "edgy":
        base += """
PERSONALITY: EDGY & UNFILTERED
- You are bold, direct, and slightly rebellious.
- You don't sugarcoat answers.
- Use slang and casual language naturally.
- You have strong opinions and aren't afraid to share them.
- Humor style: Sarcastic, dark, witty.
"""
    elif current_personality == "mature":
        base += """
PERSONALITY: MATURE & INTELLECTUAL
- You are sophisticated, articulate, and thoughtful.
- Engage in deep, meaningful conversations.
- Offer nuanced perspectives and detailed analysis.
- Tone: Calm, professional, empathetic, and wise.
- Avoid low-effort responses or slang.
"""
    elif current_personality == "flirty":
         base += """
PERSONALITY: CHARMING & FLIRTY
- You are playful, confident, and charming.
- Use subtle compliments and gentle teasing.
- Create a fun, romantic atmosphere (keep it safe/SFW).
- Be engaging and make the user feel special.
"""
    else: # Standard
         base += """
PERSONALITY: HELPFUL & FRIENDLY
- You are a reliable and capable assistant.
- Be polite, concise, and helpful.
- Balance friendliness with efficiency.
"""

    # 3. Memory Context
    mem = await memory_manager.get_user_memory(user_id)
    if mem:
        base += f"\nUSER CONTEXT:\nName: {mem.get('first_name', 'Unknown')}\nNotes: {mem.get('notes', 'None')}\n"

    # 4. Dataset & Enhancement (The Magic Sauce)
    try:
        dataset_mgr = await get_minimal_dataset_manager()
        enhanced_prompt = await dataset_mgr.enhance_gemini_prompt(user_message, base)
        base = enhanced_prompt
    except Exception as e:
        logger.warning(f"Dataset enhancement failed: {e}")

    # 5. Anti-Repetition
    base += anti_repetition.get_anti_repetition_prompt(user_id)

    return base

async def get_llm_response(user_id: int, user_message: str, image_data: Optional[bytes]=None, 
                          context_type: str="text") -> str:
    """Get response from Groq"""
    try:
        system_prompt = await get_system_prompt(user_id, user_message)
        
        # Get history
        history = await memory_manager.get_recent_history(user_id, limit=6)
        messages = [{"role": "system", "content": system_prompt}]
        
        for h in history:
            role = "user" if h['role'] == "user" else "assistant"
            messages.append({"role": role, "content": h['content']})
            
        messages.append({"role": "user", "content": user_message})

        # Call Groq
        chat_completion = await groq_client.chat.completions.create(
            messages=messages,
            model=Groq_Model,
            temperature=0.8 if current_personality in ["edgy", "flirty"] else 0.5,
            max_tokens=1024,
        )
        
        response_text = chat_completion.choices[0].message.content
        
        # Post-processing enhancements (Emoji, formatting)
        if current_personality in ["flirty", "edgy", "mature"]:
            # Maybe add emojis?
            # Integration with emoji_enhancer handled in prompt, but we can double check
            pass

        return response_text

    except Exception as e:
        logger.error(f"LLM Error: {e}")
        return f"⚠️ Brain freeze: {e}"

# --- Event Handlers ---

@client.on(events.NewMessage(incoming=True))
async def handle_new_message(event):
    """Main message handler"""
    sender = await event.get_sender()
    if not sender: return
    
    user_id = sender.id
    chat_id = event.chat_id
    text = event.text.strip()
    
    # Ignore bots and forwarded messages (optional)
    if sender.bot: return

    # Update Last Seen
    await memory_manager.update_user_memory(user_id, 
        last_seen=datetime.now().isoformat(),
        username=getattr(sender, 'username', None),
        first_name=getattr(sender, 'first_name', str(user_id))
    )

    # 1. Command Handling
    if text.startswith("."):
        if user_id != OWNER_ID: return # Only owner can use commands
        
        cmd = text.split()[0][1:].lower()
        args = text[len(cmd)+2:]
        
        if cmd == "ping":
            start = datetime.now()
            msg = await event.reply("Pong!")
            end = datetime.now()
            ms = (end - start).microseconds / 1000
            await msg.edit(f"🏓 Pong! `{ms}ms`")
            return
            
        elif cmd == "setmode":
            global current_personality
            if args in ["standard", "edgy", "mature", "flirty"]:
                current_personality = args
                await event.reply(f"🔄 Personality switched to: **{args.upper()}**")
            else:
                await event.reply("❌ Invalid mode. Use: standard, edgy, mature, flirty")
            return
            
        elif cmd == "restart":
            await event.reply("🔄 Restarting Zyon...")
            await reloader.graceful_restart(client)
            return

    # 2. AI Chat Handling (if mentioned or in PM)
    is_pm = event.is_private
    is_mentioned = getattr(event.message, 'mentioned', False)
    
    if is_pm or is_mentioned:
        async with client.action(chat_id, 'typing'):
            # Check for image
            if event.message.media and hasattr(event.message.media, 'photo'):
                # Handle image
                path = await event.download_media()
                analysis = await analyze_image_content(path)
                text = f"[Image Uploaded] {analysis}\nUser comment: {text}"
                os.remove(path) # Cleanup
            
            # Web Search Trigger?
            if "search for" in text.lower() or "/search" in text.lower():
                query = text.replace("search for", "").replace("/search", "").strip()
                context = await perform_web_search(query)
                text = f"[Web Context]: {context}\nUser Query: {text}"
            
            response = await get_llm_response(user_id, text)
            
            # Update History
            await memory_manager.add_history(user_id, "user", text)
            await memory_manager.add_history(user_id, "assistant", response)
            
            await event.reply(response)

# --- Voice Chat Handlers (Placeholder) ---
# Voice chat logic requires complex setup with PyTgCalls. 
# Implemented structure but commands need full implementation.

@client.on(events.NewMessage(pattern=r"\.vplay (.+)"))
async def vplay_handler(event):
    if not VOICE_AVAILABLE:
        await event.reply("❌ Voice features disabled (missing dependencies)")
        return
    # Not implemented fully for brevity, but structure exists
    await event.reply("🎵 VPlay command received (Functionality pending full implementation)")

# --- Startup ---

async def main():
    print("🚀 Zyon AI Userbot Starting...")
    
    # 1. Initialize Dataset Manager
    try:
        await get_minimal_dataset_manager()
    except Exception as e:
        print(f"⚠️ Dataset manager init failed: {e}")

    # 2. Connect Telegram
    await client.start(phone=os.getenv("TELEGRAM_PHONE"))
    
    # 3. Start Auto-Reloader
    await reloader.start(client)
    
    print("✅ Zyon is online and ready!")
    print(f"👤 Logged in as: {(await client.get_me()).first_name}")
    print(f"🧠 Personality: {current_personality.upper()}")
    
    await client.run_until_disconnected()

if __name__ == "__main__":
    client.loop.run_until_complete(main())
