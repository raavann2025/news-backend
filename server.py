import asyncio
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Literal, Optional

import bcrypt
import jwt
import requests
from dotenv import load_dotenv
from fastapi import (APIRouter, Depends, FastAPI, File, Header, HTTPException,
                     Query, Request, UploadFile)
from fastapi.responses import Response, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.cors import CORSMiddleware

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

MONGO_URL = os.environ["MONGO_URL"]
DB_NAME = os.environ["DB_NAME"]
JWT_SECRET = os.environ["JWT_SECRET"]
JWT_ALGORITHM = "HS256"
EMERGENT_LLM_KEY = os.environ.get("EMERGENT_LLM_KEY", "")

# ---- OpenAI (user's own API key) ----
from openai import AsyncOpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
OPENAI_TTS_MODEL = os.environ.get("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_STT_MODEL = os.environ.get("OPENAI_STT_MODEL", "whisper-1")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

client = AsyncIOMotorClient(MONGO_URL)
db = client[DB_NAME]

app = FastAPI(title="The Rabbit Hole API")
api_router = APIRouter(prefix="/api")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rabbithole")

# ============ Object storage ============
STORAGE_BASE = (
    os.environ.get("INTEGRATION_PROXY_URL") or ""
).strip() or "https://integrations.emergentagent.com"
STORAGE_URL = STORAGE_BASE.rstrip("/") + "/objstore/api/v1/storage"
APP_NAME = "rabbithole"
_storage_key = None


def init_storage(force: bool = False):
    global _storage_key
    if _storage_key and not force:
        return _storage_key
    resp = requests.post(
        f"{STORAGE_URL}/init", json={"emergent_key": EMERGENT_LLM_KEY}, timeout=30
    )
    resp.raise_for_status()
    _storage_key = resp.json()["storage_key"]
    return _storage_key


def put_object(path: str, data: bytes, content_type: str) -> dict:
    key = init_storage()
    resp = requests.put(
        f"{STORAGE_URL}/objects/{path}",
        headers={"X-Storage-Key": key, "Content-Type": content_type},
        data=data,
        timeout=120,
    )
    if resp.status_code == 404:
        key = init_storage(force=True)
        resp = requests.put(
            f"{STORAGE_URL}/objects/{path}",
            headers={"X-Storage-Key": key, "Content-Type": content_type},
            data=data,
            timeout=120,
        )
    resp.raise_for_status()
    return resp.json()


def get_object(path: str):
    key = init_storage()
    resp = requests.get(
        f"{STORAGE_URL}/objects/{path}", headers={"X-Storage-Key": key}, timeout=60
    )
    if resp.status_code == 404:
        key = init_storage(force=True)
        resp = requests.get(
            f"{STORAGE_URL}/objects/{path}", headers={"X-Storage-Key": key}, timeout=60
        )
    resp.raise_for_status()
    return resp.content, resp.headers.get("Content-Type", "application/octet-stream")


MIME_TYPES = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}

# ============ Auth helpers ============


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(user_id: str, email: str, role: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(days=7),
        "type": "access",
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


bearer_scheme = HTTPBearer(auto_error=False)


async def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
) -> dict:
    if not creds or not creds.credentials:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        payload = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    user = await db.users.find_one(
        {"id": payload["sub"]}, {"_id": 0, "password_hash": 0}
    )
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user


async def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


# ============ Models ============
Status = Literal["draft", "published"]


class Source(BaseModel):
    title: str = ""
    url: str = ""


class CardBase(BaseModel):
    headline: str = ""
    body: str = ""
    category: str = "Insight"
    sources: List[Source] = Field(default_factory=list)
    image_url: Optional[str] = None
    backstory: Optional[str] = None
    question_label: Optional[str] = None
    status: Status = "draft"
    is_hot: bool = False


class CardCreate(CardBase):
    parent_id: Optional[str] = None
    story_id: Optional[str] = None
    created_at: Optional[str] = None


class CardUpdate(BaseModel):
    headline: Optional[str] = None
    body: Optional[str] = None
    category: Optional[str] = None
    sources: Optional[List[Source]] = None
    image_url: Optional[str] = None
    backstory: Optional[str] = None
    question_label: Optional[str] = None
    status: Optional[Status] = None
    is_hot: Optional[bool] = None
    created_at: Optional[str] = None
    order_index: Optional[int] = None


class SignupRequest(BaseModel):
    email: str
    password: str
    name: str = ""


class LoginRequest(BaseModel):
    email: str
    password: str


class GoogleSessionRequest(BaseModel):
    session_id: str


# Emails that should be granted admin (owner) role on first Google login.
ADMIN_EMAILS = {
    os.environ["ADMIN_EMAIL"].lower(),
    "smsathe99@gmail.com",
}


class RegisterRequest(BaseModel):
    email: str
    password: str
    name: str = ""
    role: str = "editor"


class DeepDiveRequest(BaseModel):
    card_id: str
    question: str
    device_id: str


class TTSRequest(BaseModel):
    text: str
    voice: str = "sage"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean(doc: dict) -> dict:
    doc.pop("_id", None)
    return doc


# ============ Startup ============
@app.on_event("startup")
async def startup():
    await db.users.create_index("email", unique=True)
    await db.cards.create_index("parent_id")
    await db.cards.create_index("story_id")
    await db.cards.create_index([("is_hot", 1), ("status", 1)])
    # Seed admin
    admin_email = os.environ["ADMIN_EMAIL"].lower()
    admin_password = os.environ["ADMIN_PASSWORD"]
    existing = await db.users.find_one({"email": admin_email})
    if existing is None:
        await db.users.insert_one(
            {
                "id": str(uuid.uuid4()),
                "email": admin_email,
                "password_hash": hash_password(admin_password),
                "name": os.environ.get("ADMIN_NAME", "Admin"),
                "role": "admin",
                "created_at": now_iso(),
            }
        )
        logger.info("Seeded admin %s", admin_email)
    elif not verify_password(admin_password, existing["password_hash"]):
        await db.users.update_one(
            {"email": admin_email},
            {"$set": {"password_hash": hash_password(admin_password)}},
        )
    try:
        init_storage()
        logger.info("Storage initialized")
    except Exception as e:
        logger.error("Storage init failed: %s", e)


@app.on_event("shutdown")
async def shutdown():
    client.close()


# ============ Auth endpoints ============
@api_router.post("/auth/login")
async def login(req: LoginRequest):
    user = await db.users.find_one({"email": req.email.lower().strip()})
    if not user or not verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token(user["id"], user["email"], user["role"])
    return {
        "token": token,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user.get("name", ""),
            "role": user["role"],
        },
    }


@api_router.get("/auth/me")
async def me(user: dict = Depends(get_current_user)):
    return user


# ---- Public self-service signup (simple: email + password + name) ----
# Anyone can create an editor account; their details are stored so they can
# sign back in later. Owner emails are auto-promoted to admin.
@api_router.post("/auth/signup")
async def signup(req: SignupRequest):
    email = req.email.lower().strip()
    if "@" not in email:
        raise HTTPException(status_code=400, detail="Enter a valid email")
    if len(req.password) < 6:
        raise HTTPException(
            status_code=400, detail="Password must be at least 6 characters"
        )
    if await db.users.find_one({"email": email}):
        raise HTTPException(
            status_code=400, detail="Email already registered — please sign in"
        )
    role = "admin" if email in ADMIN_EMAILS else "editor"
    doc = {
        "id": str(uuid.uuid4()),
        "email": email,
        "password_hash": hash_password(req.password),
        "name": req.name.strip() or email.split("@")[0],
        "role": role,
        "provider": "password",
        "created_at": now_iso(),
    }
    await db.users.insert_one(doc)
    token = create_access_token(doc["id"], doc["email"], doc["role"])
    return {
        "token": token,
        "user": {
            "id": doc["id"],
            "email": doc["email"],
            "name": doc["name"],
            "role": doc["role"],
        },
    }


# ---- Google login (Emergent-managed OAuth) ----
# Exchanges the Emergent session_id for the user's profile, auto-provisions an
# editor account, and returns OUR JWT so the rest of the app keeps using bearer tokens.
@api_router.post("/auth/google/session")
async def google_session(req: GoogleSessionRequest):
    try:
        resp = requests.get(
            "https://demobackend.emergentagent.com/auth/v1/env/oauth/session-data",
            headers={"X-Session-ID": req.session_id},
            timeout=20,
        )
        resp.raise_for_status()
        profile = resp.json()
    except Exception:
        raise HTTPException(status_code=401, detail="Google sign-in failed")
    email = (profile.get("email") or "").lower().strip()
    if not email:
        raise HTTPException(status_code=401, detail="Google account has no email")
    user = await db.users.find_one({"email": email})
    role = "admin" if email in ADMIN_EMAILS else "editor"
    if not user:
        user = {
            "id": str(uuid.uuid4()),
            "email": email,
            "password_hash": hash_password(
                uuid.uuid4().hex
            ),  # random; login is via Google
            "name": profile.get("name", email.split("@")[0]),
            "picture": profile.get("picture"),
            "role": role,
            "provider": "google",
            "created_at": now_iso(),
        }
        await db.users.insert_one(user)
    token = create_access_token(user["id"], user["email"], user["role"])
    return {
        "token": token,
        "user": {
            "id": user["id"],
            "email": user["email"],
            "name": user.get("name", ""),
            "role": user["role"],
        },
    }


@api_router.post("/auth/register")
async def register(req: RegisterRequest, admin: dict = Depends(require_admin)):
    email = req.email.lower().strip()
    if await db.users.find_one({"email": email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    doc = {
        "id": str(uuid.uuid4()),
        "email": email,
        "password_hash": hash_password(req.password),
        "name": req.name,
        "role": req.role if req.role in ("admin", "editor") else "editor",
        "created_at": now_iso(),
    }
    await db.users.insert_one(doc)
    return {"id": doc["id"], "email": email, "name": doc["name"], "role": doc["role"]}


@api_router.get("/auth/editors")
async def list_editors(admin: dict = Depends(require_admin)):
    users = await db.users.find({}, {"_id": 0, "password_hash": 0}).to_list(500)
    return users


@api_router.delete("/auth/editors/{user_id}")
async def delete_editor(user_id: str, admin: dict = Depends(require_admin)):
    if user_id == admin["id"]:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")
    await db.users.delete_one({"id": user_id, "role": {"$ne": "admin"}})
    return {"ok": True}


# ============ Card helpers ============
async def _next_order(parent_id: Optional[str], story_id: str) -> int:
    q = (
        {"parent_id": parent_id}
        if parent_id
        else {"parent_id": None, "story_id": story_id}
    )
    if parent_id:
        q = {"parent_id": parent_id}
    else:
        q = {"parent_id": None}
    count = await db.cards.count_documents(q)
    return count


async def _collect_descendants(card_id: str) -> List[str]:
    ids = []
    stack = [card_id]
    while stack:
        cur = stack.pop()
        children = await db.cards.find({"parent_id": cur}, {"id": 1}).to_list(1000)
        for c in children:
            ids.append(c["id"])
            stack.append(c["id"])
    return ids


async def _build_tree(card_id: str) -> dict:
    card = await db.cards.find_one({"id": card_id}, {"_id": 0})
    if not card:
        return None
    children_docs = (
        await db.cards.find({"parent_id": card_id}, {"_id": 0})
        .sort("order_index", 1)
        .to_list(1000)
    )
    card["children"] = [await _build_tree(c["id"]) for c in children_docs]
    return card


# ============ Editor CRUD ============
@api_router.get("/admin/stories")
async def list_stories(
    user: dict = Depends(get_current_user),
    status: Optional[str] = None,
    feed: Optional[str] = None,
):
    q = {"parent_id": None, "is_private": {"$ne": True}}
    if status:
        q["status"] = status
    if feed == "hot":
        q["is_hot"] = True
    elif feed == "old":
        q["is_hot"] = False
    stories = await db.cards.find(q, {"_id": 0}).sort("created_at", -1).to_list(1000)
    for s in stories:
        s["descendant_count"] = len(await _collect_descendants(s["id"]))
    return stories


@api_router.get("/admin/cards/tree/{story_id}")
async def get_tree(story_id: str, user: dict = Depends(get_current_user)):
    tree = await _build_tree(story_id)
    if not tree:
        raise HTTPException(status_code=404, detail="Story not found")
    return tree


@api_router.get("/admin/cards/{card_id}")
async def get_card(card_id: str, user: dict = Depends(get_current_user)):
    card = await db.cards.find_one({"id": card_id}, {"_id": 0})
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    return card


@api_router.post("/admin/cards")
async def create_card(req: CardCreate, user: dict = Depends(get_current_user)):
    card_id = str(uuid.uuid4())
    parent_id = req.parent_id
    if parent_id:
        parent = await db.cards.find_one({"id": parent_id}, {"_id": 0})
        if not parent:
            raise HTTPException(status_code=404, detail="Parent not found")
        story_id = parent["story_id"]
    else:
        story_id = card_id
    order = await db.cards.count_documents({"parent_id": parent_id})
    doc = {
        "id": card_id,
        "parent_id": parent_id,
        "story_id": story_id,
        "headline": req.headline,
        "body": req.body,
        "category": req.category,
        "sources": [s.model_dump() for s in req.sources],
        "image_url": req.image_url,
        "backstory": req.backstory,
        "question_label": req.question_label,
        "status": req.status,
        "is_hot": req.is_hot,
        "is_private": False,
        "device_id": None,
        "origin": "editor",
        "order_index": order,
        "created_at": req.created_at or now_iso(),
        "updated_at": now_iso(),
        "author": user.get("email"),
    }
    await db.cards.insert_one(doc)
    return clean(doc)


@api_router.put("/admin/cards/{card_id}")
async def update_card(
    card_id: str, req: CardUpdate, user: dict = Depends(get_current_user)
):
    card = await db.cards.find_one({"id": card_id})
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    patch = {
        k: v for k, v in req.model_dump(exclude_unset=True).items() if v is not None
    }
    if "sources" in patch:
        patch["sources"] = [
            s if isinstance(s, dict) else s.model_dump() for s in patch["sources"]
        ]
    patch["updated_at"] = now_iso()
    await db.cards.update_one({"id": card_id}, {"$set": patch})
    updated = await db.cards.find_one({"id": card_id}, {"_id": 0})
    return updated


@api_router.delete("/admin/cards/{card_id}")
async def delete_card(card_id: str, user: dict = Depends(get_current_user)):
    descendants = await _collect_descendants(card_id)
    all_ids = [card_id] + descendants
    await db.cards.delete_many({"id": {"$in": all_ids}})
    return {"deleted": len(all_ids)}


@api_router.post("/admin/cards/reorder")
async def reorder_cards(payload: dict, user: dict = Depends(get_current_user)):
    ordered_ids = payload.get("ordered_ids", [])
    for idx, cid in enumerate(ordered_ids):
        await db.cards.update_one({"id": cid}, {"$set": {"order_index": idx}})
    return {"ok": True}


@api_router.post("/admin/upload")
async def upload_image(
    file: UploadFile = File(...), user: dict = Depends(get_current_user)
):
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else "png"
    content_type = MIME_TYPES.get(ext, file.content_type or "image/png")
    path = f"{APP_NAME}/uploads/{user['id']}/{uuid.uuid4()}.{ext}"
    data = await file.read()
    result = put_object(path, data, content_type)
    await db.files.insert_one(
        {
            "id": str(uuid.uuid4()),
            "storage_path": result["path"],
            "original_filename": file.filename,
            "content_type": content_type,
            "size": result.get("size", len(data)),
            "is_deleted": False,
            "created_at": now_iso(),
        }
    )
    backend = os.environ.get("REACT_APP_BACKEND_URL", "")
    return {"url": f"/api/files/{result['path']}", "path": result["path"]}


@api_router.get("/files/{path:path}")
async def serve_file(path: str):
    record = await db.files.find_one({"storage_path": path, "is_deleted": False})
    if not record:
        raise HTTPException(status_code=404, detail="File not found")
    data, content_type = get_object(path)
    return Response(content=data, media_type=record.get("content_type", content_type))


# ============ Community AI queue ============
@api_router.get("/admin/community")
async def community_queue(user: dict = Depends(get_current_user)):
    cards = (
        await db.cards.find({"origin": "user_ai", "is_private": True}, {"_id": 0})
        .sort("created_at", -1)
        .to_list(500)
    )
    for c in cards:
        parent = await db.cards.find_one(
            {"id": c.get("parent_id")}, {"_id": 0, "headline": 1}
        )
        c["parent_headline"] = parent.get("headline") if parent else None
    return cards


@api_router.post("/admin/community/{card_id}/publish")
async def publish_community(
    card_id: str, req: CardUpdate, user: dict = Depends(get_current_user)
):
    card = await db.cards.find_one({"id": card_id})
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    patch = {
        k: v for k, v in req.model_dump(exclude_unset=True).items() if v is not None
    }
    if "sources" in patch:
        patch["sources"] = [
            s if isinstance(s, dict) else s.model_dump() for s in patch["sources"]
        ]
    patch.update(
        {
            "is_private": False,
            "origin": "editor",
            "status": "published",
            "device_id": None,
            "updated_at": now_iso(),
        }
    )
    await db.cards.update_one({"id": card_id}, {"$set": patch})
    return await db.cards.find_one({"id": card_id}, {"_id": 0})


@api_router.delete("/admin/community/{card_id}")
async def reject_community(card_id: str, user: dict = Depends(get_current_user)):
    descendants = await _collect_descendants(card_id)
    await db.cards.delete_many({"id": {"$in": [card_id] + descendants}})
    return {"ok": True}


# ============ Consumer feed (public) ============
def _to_consumer(c: dict) -> dict:
    return {
        "id": c["id"],
        "topic": c.get("category", "Insight"),
        "headline": c.get("headline", ""),
        "text": c.get("body", ""),
        "category": c.get("category", "Insight"),
        "image_url": c.get("image_url"),
        "source_url": (
            (c.get("sources") or [{}])[0].get("url") if c.get("sources") else None
        ),
        "source_title": (
            (c.get("sources") or [{}])[0].get("title") if c.get("sources") else None
        ),
        "sources": c.get("sources", []),
        "question_label": c.get("question_label"),
        "backstory": c.get("backstory"),
        "has_backstory": bool((c.get("backstory") or "").strip()),
        "has_children": c.get("_has_children", False),
        "status": c.get("status"),
        "created_at": c.get("created_at"),
    }


async def _attach_has_children(cards: List[dict]):
    for c in cards:
        c["_has_children"] = (
            await db.cards.count_documents(
                {
                    "parent_id": c["id"],
                    "status": "published",
                    "is_private": {"$ne": True},
                }
            )
            > 0
        )
    return cards


@api_router.get("/feed/hot")
async def feed_hot():
    cards = (
        await db.cards.find(
            {
                "parent_id": None,
                "status": "published",
                "is_hot": True,
                "is_private": {"$ne": True},
            },
            {"_id": 0},
        )
        .sort("created_at", -1)
        .to_list(100)
    )
    await _attach_has_children(cards)
    return {"cards": [_to_consumer(c) for c in cards], "generated_at": now_iso()}


@api_router.get("/feed/old")
async def feed_old(
    cursor: Optional[str] = None, direction: str = "older", limit: int = 12
):
    base = {
        "parent_id": None,
        "status": "published",
        "is_hot": False,
        "is_private": {"$ne": True},
    }
    if cursor:
        anchor = await db.cards.find_one({"id": cursor}, {"_id": 0, "created_at": 1})
        anchor_date = anchor["created_at"] if anchor else cursor
        if direction == "newer":
            base["created_at"] = {"$gt": anchor_date}
            cards = (
                await db.cards.find(base, {"_id": 0})
                .sort("created_at", 1)
                .to_list(limit)
            )
            cards = list(reversed(cards))
        else:
            base["created_at"] = {"$lt": anchor_date}
            cards = (
                await db.cards.find(base, {"_id": 0})
                .sort("created_at", -1)
                .to_list(limit)
            )
    else:
        cards = (
            await db.cards.find(base, {"_id": 0}).sort("created_at", -1).to_list(limit)
        )
    await _attach_has_children(cards)
    return {"cards": [_to_consumer(c) for c in cards]}


@api_router.get("/cards/{card_id}")
async def public_card(card_id: str):
    c = await db.cards.find_one({"id": card_id, "status": "published"}, {"_id": 0})
    if not c:
        raise HTTPException(status_code=404, detail="Card not found")
    c["_has_children"] = (
        await db.cards.count_documents(
            {"parent_id": card_id, "status": "published", "is_private": {"$ne": True}}
        )
        > 0
    )
    return _to_consumer(c)


# Public preview — returns a card in ANY status (draft or published) so editors
# can share a live preview link of exactly what the reader will see.
@api_router.get("/preview/{card_id}")
async def preview_card(card_id: str):
    c = await db.cards.find_one({"id": card_id}, {"_id": 0})
    if not c:
        raise HTTPException(status_code=404, detail="Card not found")
    c["_has_children"] = await db.cards.count_documents({"parent_id": card_id}) > 0
    return _to_consumer(c)


@api_router.get("/cards/{card_id}/children")
async def public_children(card_id: str, device_id: Optional[str] = None):
    q = {
        "parent_id": card_id,
        "status": "published",
        "$or": [{"is_private": {"$ne": True}}],
    }
    if device_id:
        q["$or"].append({"is_private": True, "device_id": device_id})
    cards = await db.cards.find(q, {"_id": 0}).sort("order_index", 1).to_list(100)
    await _attach_has_children(cards)
    return {"cards": [_to_consumer(c) for c in cards]}


@api_router.get("/cards/{card_id}/questions")
async def public_questions(card_id: str):
    cards = (
        await db.cards.find(
            {
                "parent_id": card_id,
                "status": "published",
                "is_private": {"$ne": True},
                "question_label": {"$nin": [None, ""]},
            },
            {"_id": 0},
        )
        .sort("order_index", 1)
        .to_list(50)
    )
    return {"questions": [{"id": c["id"], "label": c["question_label"]} for c in cards]}


# ============ Private AI deep-dive ============
# Best-effort web search. Enabled only when TAVILY_API_KEY is configured;
# otherwise the AI answers grounded purely in the journalist's card content.
def web_search(query: str, max_results: int = 4):
    key = os.environ.get("TAVILY_API_KEY")
    if not key:
        return []
    try:
        r = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
            },
            timeout=15,
        )
        r.raise_for_status()
        return [
            {
                "title": x.get("title", ""),
                "url": x.get("url", ""),
                "content": x.get("content", ""),
            }
            for x in r.json().get("results", [])
        ]
    except Exception:
        logger.warning("web search failed")
        return []


async def _gen_ai_card(card_id: str, question: str, device_id: str):
    parent = await db.cards.find_one({"id": card_id}, {"_id": 0})
    context = ""
    if parent:
        context = (
            f"THE JOURNALIST'S CARD (primary source — ground your answer in this):\n"
            f"Headline: {parent.get('headline')}\nBody: {parent.get('body')}\n"
            f"Category: {parent.get('category')}"
        )
    # Augment with a live web search (Tavily).
    results = await asyncio.to_thread(
        web_search, f"{parent.get('headline','') if parent else ''} {question}"
    )
    web_ctx = ""
    web_sources = []
    if results:
        web_ctx = "\n\nLIVE WEB RESULTS:\n" + "\n".join(
            f"- {r['title']}: {r['content'][:280]} ({r['url']})" for r in results
        )
        web_sources = [
            {"title": r["title"][:80], "url": r["url"]} for r in results if r.get("url")
        ][:3]

    system = (
        "You are a sharp newsroom explainer for a swipe-based news app. A reader asked a "
        "custom follow-up question about the current insight card. FIRST rely on the "
        "journalist's card content; use the live web results only to add current, factual "
        "detail. Answer as a single crisp insight card: a punchy HEADLINE line, then 3-4 "
        "factual sentences. Format EXACTLY as:\nHEADLINE: <headline>\nBODY: <3-4 sentences>"
    )
    prompt = f"{context}{web_ctx}\n\nReader's question: {question}\n\nWrite the deeper insight card."

    buffer = ""
    stream = await openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            buffer += delta
            yield ("delta", delta)

    headline, body = "", buffer.strip()
    for line in buffer.splitlines():
        if line.strip().upper().startswith("HEADLINE:"):
            headline = line.split(":", 1)[1].strip()
    if "BODY:" in buffer:
        body = buffer.split("BODY:", 1)[1].strip()
    if not headline:
        headline = question[:80]
    new_id = str(uuid.uuid4())
    story_id = parent["story_id"] if parent else new_id
    order = await db.cards.count_documents({"parent_id": card_id})
    doc = {
        "id": new_id,
        "parent_id": card_id,
        "story_id": story_id,
        "headline": headline,
        "body": body,
        "category": parent.get("category", "Insight") if parent else "Insight",
        "sources": web_sources,
        "image_url": None,
        "backstory": None,
        "question_label": question,
        "status": "published",
        "is_hot": False,
        "is_private": True,
        "device_id": device_id,
        "origin": "user_ai",
        "order_index": order,
        "created_at": now_iso(),
        "updated_at": now_iso(),
    }
    await db.cards.insert_one(doc)
    yield ("done", _to_consumer(clean(doc)))


@api_router.post("/ai/deepdive/stream")
async def ai_deepdive_stream(req: DeepDiveRequest):
    import json

    async def gen():
        try:
            async for phase, data in _gen_ai_card(
                req.card_id, req.question, req.device_id
            ):
                if phase == "delta":
                    yield f"data: {json.dumps({'phase': 'delta', 'delta': data})}\n\n"
                elif phase == "done":
                    yield f"data: {json.dumps({'phase': 'done', 'card': data})}\n\n"
        except Exception as e:
            logger.exception("deepdive failed")
            yield f"data: {json.dumps({'phase': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@api_router.post("/ai/tts")
async def ai_tts(req: TTSRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text required")
    try:
        resp = await openai_client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=req.voice or "sage",
            input=req.text[:4000],
            response_format="mp3",
        )
        audio = resp.read()
        return Response(content=audio, media_type="audio/mpeg")
    except Exception:
        logger.exception("tts failed")
        raise HTTPException(status_code=500, detail="TTS failed")


@api_router.post("/ai/stt")
async def ai_stt(file: UploadFile = File(...)):
    import tempfile

    tmp_path = None
    try:
        data = await file.read()
        suffix = "." + (
            file.filename.rsplit(".", 1)[-1] if "." in file.filename else "webm"
        )
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        with open(tmp_path, "rb") as fh:
            tr = await openai_client.audio.transcriptions.create(
                model=OPENAI_STT_MODEL, file=fh
            )
        return {"text": tr.text}
    except Exception:
        logger.exception("stt failed")
        raise HTTPException(status_code=500, detail="Transcription failed")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


@api_router.get("/")
async def root():
    return {"service": "The Rabbit Hole API", "status": "ok"}


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
