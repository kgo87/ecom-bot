import os
import re
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


ORDERS_PATH = "data/orders.json"
FAQ_PATH = "data/faq.json"

with open(ORDERS_PATH, "r", encoding="utf-8") as f:
    orders: Dict[str, Any] = json.load(f)

with open(FAQ_PATH, "r", encoding="utf-8") as f:
    faq_items: List[Dict[str, Any]] = json.load(f)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def now_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")

def jsonl_write(fp: str, record: Dict[str, Any]) -> None:
    with open(fp, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()

def tokenize_unicode(text: str) -> List[str]:
    text = normalize_text(text)
    tokens = re.split(r"[^\w]+", text, flags=re.UNICODE)
    return [t for t in tokens if len(t) > 2]

def find_order(order_id: str) -> Optional[Dict[str, Any]]:
    if not isinstance(orders, dict):
        return None
    return orders.get(str(order_id))

def find_faq_answer(question: str) -> Optional[str]:
    if not isinstance(faq_items, list) or not question.strip():
        return None

    q_norm = normalize_text(question)
    q_tokens = set(tokenize_unicode(question))

    best_answer: Optional[str] = None
    best_score = 0

    for item in faq_items:
        if not isinstance(item, dict):
            continue

        q_field = str(item.get("q", "") or "")
        a_field = str(item.get("a", "") or "")
        if not q_field or not a_field:
            continue

        keywords = item.get("keywords")
        if isinstance(keywords, list) and keywords:
            item_text = " ".join(map(str, keywords))
        else:
            item_text = q_field

        item_tokens = set(tokenize_unicode(item_text))
        score = len(q_tokens & item_tokens)

        item_q_norm = normalize_text(q_field)
        if item_q_norm and (item_q_norm in q_norm or q_norm in item_q_norm):
            score += 2

        if score > best_score:
            best_score = score
            best_answer = a_field

    return best_answer if best_score >= 2 else None

# -----------------------------
# CLI Bot
# -----------------------------
class ShoplyCLIBot:
    def __init__(self, model_name: str, logs_dir: str = "logs"):
        ensure_dir(logs_dir)
        self.logs_dir = logs_dir

        self.store: Dict[str, InMemoryChatMessageHistory] = {}

        self.system_prompt = (
            "You are Shoply Support, a concise and polite ecommerce support assistant.\n"
            "Rules:\n"
            "1) If an FAQ answer is available, respond using ONLY that FAQ answer.\n"
            "2) For /order lookups, rely ONLY on the order data provided.\n"
            "3) If info is not in FAQ or order data, say you don’t have that info and suggest contacting support.\n"
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

        self.session_usage: Dict[str, Dict[str, int]] = {}
        self.session_logfile: Dict[str, str] = {}

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def reset_session(self, session_id: str) -> None:
        if session_id in self.store:
            del self.store[session_id]
        self.session_usage.pop(session_id, None)

    def _ensure_session_logging(self, session_id: str) -> str:
        if session_id in self.session_logfile:
            return self.session_logfile[session_id]
        fp = os.path.join(self.logs_dir, f"session_{session_id}_{now_ts()}.jsonl")
        self.session_logfile[session_id] = fp
        jsonl_write(fp, {
            "type": "session_start",
            "session_id": session_id,
            "ts": datetime.utcnow().isoformat() + "Z"
        })
        return fp

    def _accumulate_usage(self, session_id: str, usage: Dict[str, Any]) -> None:
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
        jsonl_write(fp, {
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
        })

    def _log_event(self, session_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        fp = self._ensure_session_logging(session_id)
        jsonl_write(fp, {
            "type": event_type,
            "session_id": session_id,
            "ts": datetime.utcnow().isoformat() + "Z",
            **payload
        })

    def _format_order_reply(self, order_id: str, order: Dict[str, Any]) -> str:
        status = str(order.get("status", "unknown"))
        extras = []
        for k in ("eta_days", "delivered_at", "carrier", "tracking", "note"):
            if k in order and order[k] is not None and str(order[k]).strip() != "":
                extras.append(f"{k}: {order[k]}")
        extra_text = f" ({', '.join(extras)})" if extras else ""
        return f"Order {order_id} status: {status}{extra_text}."

    def handle_order_command(self, session_id: str, user_text: str) -> str:
        m = re.match(r"^\s*/order\s+([A-Za-z0-9_-]+)\s*$", user_text)
        if not m:
            reply = "Usage: /order <id>"
            self._log_turn(session_id, user_text, reply, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}, {"route": "order_command"})
            return reply

        order_id = m.group(1)
        order = find_order(order_id)

        if not order:
            reply = f"Sorry — I couldn’t find order {order_id}. Please double-check the ID."
        else:
            reply = self._format_order_reply(order_id, order)

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
        answer = find_faq_answer(user_text)
        if answer:
            reply = answer.strip()
            self._log_turn(
                session_id=session_id,
                user_text=user_text,
                bot_text=reply,
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                meta={"route": "faq"}
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

        usage: Dict[str, Any] = {}
        md = getattr(result, "response_metadata", None) or {}
        if isinstance(md, dict):
            if isinstance(md.get("token_usage"), dict):
                usage = md["token_usage"]
            elif isinstance(md.get("usage"), dict):
                usage = md["usage"]
            elif isinstance(md.get("tokenUsage"), dict):
                usage = md["tokenUsage"]

        self._accumulate_usage(session_id, usage)
        self._log_turn(session_id, user_text, bot_text, usage, {"route": "llm"})
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
