"""
Emoji Enhancement for Zyon - Makes responses more expressive
"""

import random

EMOJI_RULES = {
    "flirty_romance": {
        "emojis": ["😏", "😊", "💋", "😘", "💕", "✨", "🔥", "😉", "💖", "🌹"],
        "frequency": "high",  # Use emojis frequently
        "placement": "Use emojis to emphasize flirty moments and create romantic tension"
    },
    
    "edgy_bold": {
        "emojis": ["🔥", "💀", "😎", "🤷", "💯", "👀", "😤", "⚡", "🗿", "💪"],
        "frequency": "medium",
        "placement": "Use emojis for emphasis and to add edge to bold statements"
    },
    
    "mature_general": {
        "emojis": ["😊", "👍", "✨", "🎯", "💡", "🤔", "😅", "👌", "🌟", "💬"],
        "frequency": "medium",
        "placement": "Use emojis naturally to enhance expression without overdoing it"
    },
    
    "unfiltered_authentic": {
        "emojis": ["💀", "😭", "💯", "🔥", "😂", "🤦", "🙄", "👀", "🤷", "😬"],
        "frequency": "high",
        "placement": "Use emojis like real people do - for emphasis, sarcasm, and emotion"
    },
    
    # Emotion-based emoji mapping
    "emotions": {
        "happy": ["😊", "😄", "😁", "🥰", "✨", "🌟"],
        "excited": ["🔥", "🎉", "⚡", "💯", "😍", "🤩"],
        "funny": ["😂", "🤣", "💀", "😭", "😅"],
        "thinking": ["🤔", "💭", "🧐", "💡"],
        "confused": ["🤨", "😕", "🤷", "❓"],
        "sad": ["😢", "😔", "💔", "😞"],
        "angry": ["😤", "😠", "💢", "🤬"],
        "surprised": ["😲", "😮", "🤯", "😳"],
        "love": ["❤️", "💕", "💖", "💗", "😍", "🥰"],
        "playful": ["😏", "😜", "😝", "😋"],
        "sarcastic": ["🙄", "😒", "🤨", "💀"],
        "supportive": ["🤗", "💪", "👍", "🙌", "✨"]
    }
}

def get_emoji_guidance(mode: str) -> str:
    """Get emoji usage guidance for personality mode"""
    rules = EMOJI_RULES.get(mode, EMOJI_RULES["mature_general"])
    
    emojis_list = ", ".join(rules["emojis"][:8])
    
    guidance = f"""
    
🎭 EMOJI EXPRESSION GUIDE:
- Frequency: {rules["frequency"].upper()} usage
- Preferred emojis: {emojis_list}
- Usage: {rules["placement"]}

NATURAL EMOJI USAGE RULES:
✅ Use emojis to enhance emotion and emphasis
✅ Place emojis at natural break points (end of sentences, after key phrases)
✅ Mix emojis with text naturally like humans do
✅ Use 2-4 emojis per response depending on length
✅ React to user's emotional tone

❌ Don't overuse - quality over quantity
❌ Don't use emojis in every single sentence
❌ Avoid emoji spam (multiple same emojis)

EXAMPLES OF GOOD EMOJI USAGE:
"That's actually pretty smart! 💡 Never thought of it that way."
"Honestly? I think you're overthinking this 🤷 Just go for it!"
"Oh you're trouble, I can tell 😏 But I like that energy ✨"
"""
    
    return guidance

def suggest_contextual_emojis(text_sentiment: str) -> list:
    """Suggest emojis based on text sentiment"""
    emotion_map = {
        "positive": EMOJI_RULES["emotions"]["happy"] + EMOJI_RULES["emotions"]["excited"],
        "funny": EMOJI_RULES["emotions"]["funny"],
        "thoughtful": EMOJI_RULES["emotions"]["thinking"],
        "romantic": EMOJI_RULES["emotions"]["love"] + EMOJI_RULES["emotions"]["playful"],
        "supportive": EMOJI_RULES["emotions"]["supportive"]
    }
    
    return emotion_map.get(text_sentiment, EMOJI_RULES["emotions"]["happy"])

# Example usage
if __name__ == "__main__":
    print(get_emoji_guidance("flirty_romance"))
