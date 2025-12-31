#!/usr/bin/env python3
"""
MINIMAL Dataset Download for Zyon AI Bot - 15GB ONLY
Just the essentials for Gemini prompt enhancement
"""

import os
from datasets import load_dataset
from huggingface_hub import login
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_TOKEN = os.getenv("HF_TOKEN", "your_huggingface_token_here")
try:
    login(token=HF_TOKEN)
    logger.info("✓ Authenticated with HuggingFace")
except Exception as e:
    logger.warning(f"Authentication warning: {e}")

DATASET_DIR = "./zyon_datasets"
os.makedirs(DATASET_DIR, exist_ok=True)

def download_minimal_datasets():
    """Download only 5 essential datasets (15 GB total)"""
    
    logger.info("=" * 60)
    logger.info("ZYON MINIMAL DATASET COLLECTION - 15 GB")
    logger.info("=" * 60)
    
    # 1. LMSYS Chat-1M
    logger.info("\n1/5 - LMSYS Chat-1M (1.5 GB)")
    try:
        dataset = load_dataset("lmsys/lmsys-chat-1m", token=HF_TOKEN, cache_dir=f"{DATASET_DIR}/lmsys")
        logger.info("✓ LMSYS Chat-1M downloaded")
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
    
    # 2. OpenAssistant v1
    logger.info("\n2/5 - OpenAssistant (2 GB)")
    try:
        dataset = load_dataset("OpenAssistant/oasst1", cache_dir=f"{DATASET_DIR}/openassistant")
        logger.info("✓ OpenAssistant downloaded")
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
    
    # 3. Anthropic HH-RLHF
    logger.info("\n3/5 - Anthropic HH-RLHF (2 GB)")
    try:
        dataset = load_dataset("Anthropic/hh-rlhf", cache_dir=f"{DATASET_DIR}/hh_rlhf")
        logger.info("✓ Anthropic HH-RLHF downloaded")
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
    
    # 4. SQuAD v1 & v2
    logger.info("\n4/5 - SQuAD v1.1 (0.1 GB)")
    try:
        dataset = load_dataset("squad", cache_dir=f"{DATASET_DIR}/squad")
        logger.info("✓ SQuAD v1.1 downloaded")
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
    
    logger.info("\n4b/5 - SQuAD v2.0 (0.15 GB)")
    try:
        dataset = load_dataset("squad_v2", cache_dir=f"{DATASET_DIR}/squad_v2")
        logger.info("✓ SQuAD v2.0 downloaded")
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
    
    # 5. Wikipedia
    logger.info("\n5/5 - Wikipedia (10 GB)")
    logger.info("⚠️  This will take 20-40 minutes")
    try:
        dataset = load_dataset("wikipedia", "20220301.en", split="train[:15%]", cache_dir=f"{DATASET_DIR}/wikipedia")
        logger.info("✓ Wikipedia downloaded")
    except Exception as e:
        logger.error(f"✗ Failed: {e}")
    
    logger.info("\n" + "=" * 60)
    logger.info("🎉 MINIMAL DATASET COLLECTION COMPLETE!")
    logger.info("=" * 60)
    logger.info("Total: ~15 GB")

if __name__ == "__main__":
    print("ZYON AI BOT - MINIMAL DATASET COLLECTION (15 GB)")
    print("Estimated time: 30-60 minutes")
    confirm = input("Start download? (yes/no): ")
    if confirm.lower() in ["yes", "y"]:
        download_minimal_datasets()
