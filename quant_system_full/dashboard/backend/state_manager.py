import json, os, time
from typing import Any, Dict

BASE_DIR = os.getenv("STATE_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "state")))
STATUS_PATH = os.path.join(BASE_DIR, "status.json")
KILL_PATH = os.path.join(BASE_DIR, "kill.flag")
LOG_PATH = os.path.join(BASE_DIR, "bot.log")
DAILY_DIR = os.path.join(BASE_DIR, "daily")
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(DAILY_DIR, exist_ok=True)

def atomic_write_json(path: str, data: Dict[str, Any]):
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)

def read_status() -> Dict[str, Any]:
    if not os.path.exists(STATUS_PATH):
        s = {"bot":"idle","heartbeat":None,"pnl":0.0,"positions":[],"last_signal":None,"paused":False,"reason":None}
        atomic_write_json(STATUS_PATH, s); return s
    with open(STATUS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def write_status(update: Dict[str, Any]):
    cur = read_status()
    cur.update(update)
    cur.setdefault("heartbeat", int(time.time()))
    atomic_write_json(STATUS_PATH, cur)

def is_killed() -> bool:
    return os.path.exists(KILL_PATH)

def set_kill(flag: bool, reason: str = "manual"):
    if flag:
        with open(KILL_PATH, "w") as f:
            f.write(reason or "manual")
        write_status({"paused": True, "reason": reason, "bot":"paused"})
    else:
        if os.path.exists(KILL_PATH):
            os.remove(KILL_PATH)
        write_status({"paused": False, "reason": None, "bot":"running"})

def append_log(line: str):
    date_str = time.strftime("%Y-%m-%d")
    path = os.path.join(DAILY_DIR, f"bot_{date_str}.log")
    with open(path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")

def read_log_tail(n: int = 200):
    # Read from today's daily log file instead of bot.log
    date_str = time.strftime("%Y-%m-%d")
    daily_log_path = os.path.join(DAILY_DIR, f"bot_{date_str}.log")
    if os.path.exists(daily_log_path):
        with open(daily_log_path, "r", encoding="utf-8") as f:
            return [x.rstrip() for x in f.readlines()[-n:]]
    # Fallback to legacy bot.log
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            return [x.rstrip() for x in f.readlines()[-n:]]
    return []

def write_daily_report(date_str: str, content: str):
    path = os.path.join(DAILY_DIR, f"{date_str}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path
