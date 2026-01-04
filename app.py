import os
import re
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# -----------------------------
# Utilities
# -----------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def load_json_file(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default
    except json.JSONDecodeError:
        return default


def tokenize_simple(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 2]


def best_faq_match(question: str, faq_items: List[Dict[str, Any]], min_score: int = 2) -> Optional[Dict[str, Any]]:
    """
    Very simple keyword-overlap retrieval:
      - each FAQ item is expected to have: {"q": "...", "a": "...", "keywords": [...] (optional)}
    """
    q_tokens = set(tokenize_simple(question))
    if not q_tokens:
        return None

    best = None
    best_score = 0

    for item in faq_items:
        keywords = item.get("keywords")
        if isinstance(keywords, list) and keywords:
            item_tokens = set(tokenize_simple(" ".join(map(str, keywords))))
        else:
            item_tokens = set(tokenize_simple(str(item.get("q", ""))))

        score = len(q_tokens & item_tokens)
        if score > best_score:
            best_score = score
            best = item

    if best_score >= min_score:
        return best
    return None


def find_order(order_id: str, orders_obj: Any) -> Optional[Dict[str, Any]]:
    """
    Supports a few common shapes:
      - {"orders":[{"id":"123","status":"..."}]}
      - [{"id":"123","status":"..."}]
      - {"123":{"status":"..."}}  (dict keyed by id)
    """
    if orders_obj is None:
        return None

    if isinstance(orders_obj, dict):
        if order_id in orders_obj and isinstance(orders_obj[order_id], dict):
            d = dict(orders_obj[order_id])
            d.setdefault("id", order_id)
            return d

        if "orders" in orders_obj and isinstance(orders_obj["orders"], list):
            for o in orders_obj["orders"]:
                if str(o.get("id")) == order_id:
                    return o
        return None

    if isinstance(orders_obj, list):
        for o in orders_obj:
            if isinstance(o, dict) and str(o.get("id")) == order_id:
                return o

    return None


def jsonl_write(fp: str, record: Dict[str, Any]) -> None:
    with open(fp, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# -----------------------------
# CLI Bot
# -----------------------------

class ShoplyCLIBot:
    def __init__(
        self,
        model_name: str,
        faq_path: str = "faq.json",
        orders_path: str = "orders.json",
        logs_dir: str = "logs",
    ):
        ensure_dir(logs_dir)
        self.logs_dir = logs_dir
        self.faq_path = faq_path
        self.orders_path = orders_path

        self.faq_items: List[Dict[str, Any]] = load_json_file(self.faq_path, default=[])
        self.orders_obj: Any = load_json_file(self.orders_path, default=None)

        self.store: Dict[str, InMemoryChatMessageHistory] = {}

        self.system_prompt = (
            "You are Shoply Support, a concise and polite ecommerce support assistant.\n"
            "Rules:\n"
            "1) Use ONLY the provided FAQ answer when an FAQ match is available.\n"
            "2) For order questions, you MUST rely only on the explicit order status provided.\n"
            "3) If the user asks something not covered by FAQ or the order status, say you don't have that info and "
            "ask a brief follow-up or suggest contacting support.\n"
            "4) Keep responses short, factual, and helpful. No guessing.\n"
        )

        self.chat_model = ChatOpenAI(
            model=model_name, 
            temperature=0,
            request_timeout=15,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        self.chain = self.prompt | self.chat_model

        self.chain_with_history = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        )

        # Token usage counters for the current session (accumulated)
        self.session_usage: Dict[str, Dict[str, int]] = {}

        # Created per session
        self.session_logfile: Dict[str, str] = {}

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def reset_session(self, session_id: str) -> None:
        if session_id in self.store:
            del self.store[session_id]
        self.session_usage.pop(session_id, None)

    def reload_data(self) -> None:
        # optional: call if you want hot-reload
        self.faq_items = load_json_file(self.faq_path, default=[])
        self.orders_obj = load_json_file(self.orders_path, default=None)

    def _ensure_session_logging(self, session_id: str) -> str:
        if session_id in self.session_logfile:
            return self.session_logfile[session_id]
        fp = os.path.join(self.logs_dir, f"session_{session_id}_{now_ts()}.jsonl")
        self.session_logfile[session_id] = fp
        # write session start record
        jsonl_write(fp, {
            "type": "session_start",
            "session_id": session_id,
            "ts": datetime.utcnow().isoformat() + "Z"
        })
        return fp

    def _accumulate_usage(self, session_id: str, usage: Dict[str, Any]) -> None:
        # usage from OpenAI models often: {"prompt_tokens":..., "completion_tokens":..., "total_tokens":...}
        pt = int(usage.get("prompt_tokens") or 0)
        ct = int(usage.get("completion_tokens") or 0)
        tt = int(usage.get("total_tokens") or (pt + ct))

        if session_id not in self.session_usage:
            self.session_usage[session_id] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        self.session_usage[session_id]["prompt_tokens"] += pt
        self.session_usage[session_id]["completion_tokens"] += ct
        self.session_usage[session_id]["total_tokens"] += tt

    def _log_turn(self, session_id: str, user_text: str, bot_text: str, usage: Dict[str, Any], meta: Dict[str, Any]) -> None:
        fp = self._ensure_session_logging(session_id)
        record = {
            "type": "turn",
            "session_id": session_id,
            "ts": datetime.utcnow().isoformat() + "Z",
            "user": user_text,
            "assistant": bot_text,
            "usage": {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
            },
            "meta": meta,
            "session_usage_total": self.session_usage.get(session_id, {}),
        }
        jsonl_write(fp, record)

    def _log_event(self, session_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        fp = self._ensure_session_logging(session_id)
        jsonl_write(fp, {
            "type": event_type,
            "session_id": session_id,
            "ts": datetime.utcnow().isoformat() + "Z",
            **payload
        })

    def handle_order_command(self, session_id: str, user_text: str) -> str:
        # /order 12345
        m = re.match(r"^\s*/order\s+([A-Za-z0-9_-]+)\s*$", user_text)
        if not m:
            return "Usage: /order <id>"

        order_id = m.group(1)
        order = find_order(order_id, self.orders_obj)

        if not order:
            reply = f"Sorry — I couldn’t find order {order_id}. Please double-check the ID."
        else:
            status = order.get("status", "unknown")
            extras = []
            for k in ("eta", "tracking", "carrier"):
                if k in order and order[k]:
                    extras.append(f"{k}: {order[k]}")
            extra_text = f" ({', '.join(extras)})" if extras else ""
            reply = f"Order {order_id} status: {status}{extra_text}."

        self._ensure_session_logging(session_id)
        self._log_event(session_id, "order_lookup", {"order_id": order_id, "found": bool(order)})
        self._log_turn(
            session_id=session_id,
            user_text=user_text,
            bot_text=reply,
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            meta={"route": "order_command"}
        )
        return reply

    def handle_faq_or_llm(self, session_id: str, user_text: str) -> str:
        faq_match = best_faq_match(user_text, self.faq_items, min_score=2)
        if faq_match:
            reply = str(faq_match.get("a", "")).strip() or "Sorry — I don't have that answer in the FAQ."
            self._ensure_session_logging(session_id)
            self._log_turn(
                session_id=session_id,
                user_text=user_text,
                bot_text=reply,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                meta={"route": "faq", "faq_q": faq_match.get("q")}
            )
            return reply

        try:
            result = self.chain_with_history.invoke(
                {"question": user_text},
                {"configurable": {"session_id": session_id}}
            )
        except Exception as e:
            self._log_event(session_id, "error", {"message": str(e)})
            return f"[Error] {e}"

        bot_text = (result.content or "").strip()

        usage = {}
        meta = {"route": "llm"}

        md = getattr(result, "response_metadata", None) or {}
        if isinstance(md, dict):
            if isinstance(md.get("token_usage"), dict):
                usage = md["token_usage"]
            elif isinstance(md.get("usage"), dict):
                usage = md["usage"]
            elif isinstance(md.get("tokenUsage"), dict):
                usage = md["tokenUsage"]

        self._accumulate_usage(session_id, usage)
        self._log_turn(session_id, user_text, bot_text, usage, meta)

        return bot_text

    def run(self, session_id: str) -> None:
        self._ensure_session_logging(session_id)

        print(
            "Shoply Support Bot (CLI)\n"
            "Commands:\n"
            "  /order <id>   - check order status\n"
            "  reset         - clear conversation context\n"
            "  exit          - quit\n"
        )

        while True:
            try:
                user_text = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nBot: Shutting down. Bye!")
                self._log_event(session_id, "session_end", {"reason": "interrupt"})
                break

            if not user_text:
                continue

            msg = user_text.lower().strip()

            if msg in ("exit", "quit"):
                print("Bot: Goodbye!")
                self._log_event(session_id, "session_end", {"reason": "user_exit"})
                break

            if msg == "reset":
                self.reset_session(session_id)
                print("Bot: Context cleared.")
                self._log_event(session_id, "context_reset", {})
                continue

            if re.match(r"^\s*/order\s+", user_text):
                reply = self.handle_order_command(session_id, user_text)
                print(f"Bot: {reply}")
                continue

            reply = self.handle_faq_or_llm(session_id, user_text)
            print(f"Bot: {reply}")


if __name__ == "__main__":
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    session_id = os.getenv("SHOPLY_SESSION_ID", f"user_{int(time.time())}")

    bot = ShoplyCLIBot(model_name=model)
    bot.run(session_id)
