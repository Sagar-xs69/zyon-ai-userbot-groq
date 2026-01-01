# ------------------------------------------------------------
#  ULTRA SUPER ZYON AI USERBOT – GROQ VERSION
# ------------------------------------------------------------
#  pip install python-dotenv telethon groq
#  pip install pytgcalls==1.0.0b1 yt-dlp tavily pillow pytesseract
#  pip install pyPDF2 python-docx pytz aiofiles aiohttp SpeechRecognition
# ------------------------------------------------------------

import asyncio, os, logging, time, tempfile, sqlite3, json, base64, hashlib, io, re, random, requests, threading, atexit, sys
from   datetime        import datetime, timedelta
from   typing          import Dict, List, Optional, Tuple
from   io              import BytesIO
from   contextlib      import asynccontextmanager
from   collections     import defaultdict, deque
import pytz
from   groq            import Groq

from PIL import Image

from   tavily            import TavilyClient
import yt_dlp
from   PIL               import Image
from   pydub             import AudioSegment
from   pydub.utils       import which

from   telethon          import TelegramClient, events, utils
from   telethon.tl.types import DocumentAttributeAudio
from   telethon.tl.custom import Button
from   telethon.tl.functions.users      import GetFullUserRequest
from   telethon.tl.functions.messages   import ImportChatInviteRequest
from   telethon.tl.functions.channels   import JoinChannelRequest, GetFullChannelRequest, JoinChannelRequest as JoinChannel
from   telethon.errors.rpcerrorlist     import UserAlreadyParticipantError, InviteHashExpiredError, ChannelInvalidError, ChannelPrivateError
from   telethon.errors                  import SessionPasswordNeededError

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    sr = None
    SPEECH_RECOGNITION_AVAILABLE = False
    print("⚠️ speech_recognition not available - voice transcription will be disabled")

from   dotenv import load_dotenv
load_dotenv()

from dataset_manager_minimal import get_minimal_dataset_manager

# Auto-reload system
import signal
from pathlib import Path

class AutoReloader:
    def __init__(self, watch_file="vc.py"):
        self.watch_file = Path(watch_file)
        self.last_mtime = self.watch_file.stat().st_mtime if self.watch_file.exists() else 0
        self.checking = True

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
        """Ensure session is properly saved before restart with verification"""
        try:
            # Save session explicitly
            print("💾 Saving session...")
            await client.session.save()
            await asyncio.sleep(2)  # Longer wait for I/O to complete

            # Verify session was saved
            session_file = f"{SESSION_NAME}.session"
            if not os.path.exists(session_file):
                logger.error("Session file missing after save!")
                return

            # Create timestamped backup before restart
            import shutil
            backup_path = f"{session_file}.{int(time.time())}.backup"
            shutil.copy2(session_file, backup_path)
            print(f"💾 Session backup created: {backup_path}")

            # Clean up resources
            print("🧹 Cleaning up...")
            await file_manager.cleanup_all()

            # Stop pytgcalls if active
            if pytgcalls:
                try:
                    await pytgcalls.stop()
                except Exception as e:
                    logger.warning(f"Pytgcalls stop error: {e}")

            # Disconnect cleanly
            print("📴 Disconnecting...")
            if client.is_connected():
                await client.disconnect()

            await asyncio.sleep(1)  # Wait for clean disconnect

            # Restart process
            print("🔄 Restarting process...")
            os.execv(sys.executable, [sys.executable] + sys.argv)

        except Exception as e:
            logger.error(f"Graceful restart error: {e}")
            # Force restart as fallback - but try to save session first
            try:
                await client.session.save()
                await asyncio.sleep(1)
            except:
                pass
            os.execv(sys.executable, [sys.executable] + sys.argv)

    def stop(self):
        self.checking = False

auto_reloader = AutoReloader("vc.py")

def validate_env():
    required = ["TELEGRAM_API_ID", "TELEGRAM_API_HASH", "GROQ_API_KEY",
                "TAVILY_API_KEY", "OWNER_ID"]
    missing  = [v for v in required if not os.getenv(v)]
    if missing:
        raise ValueError(f"Missing env-vars: {missing}")

def diagnose_pytgcalls():
    """Diagnostic function to check PyTgCalls installation"""
    print("\n🔍  Diagnosing PyTgCalls Installation...")
    
    # Check if FFmpeg is available
    ffmpeg_path = which("ffmpeg")
    if ffmpeg_path:
        print(f"✅ FFmpeg found at: {ffmpeg_path}")
    else:
        print("❌ FFmpeg not found in PATH")
    
    # Check Python version
    import sys
    print(f"🐍 Python version: {sys.version}")
    
    # Try importing PyTgCalls components individually
    try:
        import pytgcalls
        print(f"✅ pytgcalls module imported (version: {getattr(pytgcalls, '__version__', 'unknown')})")
    except ImportError as e:
        print(f"❌ Failed to import pytgcalls: {e}")
        return
    
    try:
        from pytgcalls import PyTgCalls
        print("✅ PyTgCalls class imported")
    except ImportError as e:
        print(f"❌ Failed to import PyTgCalls class: {e}")

    try:
        from pytgcalls.types import MediaStream
        print("✅ MediaStream imported")
    except ImportError as e:
        print(f"❌ Failed to import MediaStream: {e}")
        try:
            from pytgcalls.types.input_stream import InputAudioStream
            print("✅ InputAudioStream imported (alternative)")
        except ImportError as e2:
            print(f"❌ Failed to import InputAudioStream: {e2}")
    
    print("🔍  Diagnosis complete\n")

validate_env()

# ----------------------------------------------------------
#  Voice-Call imports  –  Updated for PyTgCalls 2.x compatibility
# ----------------------------------------------------------
try:
    from pytgcalls import PyTgCalls
    from pytgcalls.types import MediaStream
    from pytgcalls.exceptions import NoActiveGroupCall
    
    PYTG_CALLS_AVAILABLE = True
    print("✅ PyTgCalls imports successful")
except ImportError as e:
    PYTG_CALLS_AVAILABLE = False
    PyTgCalls = MediaStream = NoActiveGroupCall = None
    print(f"❌ PyTgCalls import failed: {e}")
except Exception as e:
    PYTG_CALLS_AVAILABLE = False
    PyTgCalls = MediaStream = NoActiveGroupCall = None
    print(f"❌ PyTgCalls initialization error: {e}")

# ----------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
#  ENVIRONMENT
# ----------------------------------------------------------
API_ID          = int(os.getenv("TELEGRAM_API_ID"))
API_HASH        = os.getenv("TELEGRAM_API_HASH")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY  = os.getenv("TAVILY_API_KEY")
OWNER_ID        = int(os.getenv("OWNER_ID"))
TELEGRAM_PHONE  = os.getenv("TELEGRAM_PHONE")  # optional
SESSION_NAME    = 'zyon_telegram_session'

GROQ_MODEL = "openai/gpt-oss-120b"  # Groq model

# Initialize clients
groq_client = Groq(api_key=GROQ_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

# YouTube cookies setup (for bypassing bot detection)
YOUTUBE_COOKIES_PATH = None

# First try to load from local file (bundled with Docker image)
local_cookies_files = ['youtube_cookies.txt', 'www.youtube.com_cookies.txt', 'cookies.txt']
for cookie_file in local_cookies_files:
    if os.path.exists(cookie_file):
        YOUTUBE_COOKIES_PATH = os.path.abspath(cookie_file)
        print(f"✅ YouTube cookies loaded from local file: {cookie_file}")
        break

# Fall back to environment variable if no local file found
if not YOUTUBE_COOKIES_PATH:
    youtube_cookies_env = os.getenv("YOUTUBE_COOKIES")
    if youtube_cookies_env:
        try:
            import tempfile
            cookies_data = base64.b64decode(youtube_cookies_env)
            cookies_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.txt', delete=False)
            cookies_file.write(cookies_data)
            cookies_file.close()
            YOUTUBE_COOKIES_PATH = cookies_file.name
            print(f"✅ YouTube cookies loaded from environment variable")
        except Exception as e:
            print(f"⚠️ Failed to load YouTube cookies from env: {e}")

if not YOUTUBE_COOKIES_PATH:
    print("⚠️ No YouTube cookies found - some videos may not play due to bot detection")
