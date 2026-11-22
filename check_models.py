"""Finds which Gemini model works with your API key and free quota.
Run:  python check_models.py
Needs GEMINI_API_KEY set in this terminal. Never prints the key."""
import time
from google import genai

SKIP = ("image", "tts", "live", "audio", "embedding", "robotics", "computer", "veo", "imagen")

client = genai.Client()
names = []
for m in client.models.list():
    name = m.name.replace("models/", "")
    if "generateContent" in (m.supported_actions or []) and "flash" in name \
            and not any(word in name for word in SKIP):
        names.append(name)
names.sort(key=lambda n: ("lite" not in n, n))     # lite models first

print(f"Testing {len(names)} models...\n")
working = []
for name in names:
    try:
        reply = client.models.generate_content(model=name, contents="Reply with the word OK")
        print(f"WORKS   {name}  ->  {(reply.text or '').strip()[:20]}")
        working.append(name)
    except Exception as error:
        text = str(error).replace("\n", " ")
        reason = "quota is 0 / exhausted" if "429" in text else text[:90]
        print(f"FAILED  {name}  ->  {type(error).__name__}: {reason}")
    time.sleep(3)

print()
if working:
    print("Use this one:")
    print(f'  $env:CLINIC_MODEL="{working[0]}"')
else:
    print("No model worked. Your key has no free quota right now - see the advice in chat.")
