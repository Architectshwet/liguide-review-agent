import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import BaseModel

from liquide.liquide_agent import create_liquide_agent_for_request, initialize_liquide_agent, reset_liquide_agent
from liquide.services.langfuse_service import langfuse_service
from liquide.services.live_ingest_background_service import live_ingest_background_service
from liquide.services.postgres_db_service import PostgresDBService
from liquide.services.review_rag_service import review_rag_service
from liquide.state.checkpointer import get_checkpointer, get_pool
from liquide.utils.logger import get_logger

logger = get_logger(__name__)

db_service = PostgresDBService()

_cors_origins_env = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
ALLOWED_ORIGINS = [origin.strip() for origin in _cors_origins_env.split(",") if origin.strip()] if _cors_origins_env else ["*"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up Liquide Agent API...")

    try:
        langfuse_service.initialize()
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse service: {e}")

    checkpointer_ready = False
    checkpointer = get_checkpointer()
    if checkpointer:
        try:
            pool = get_pool()
            if pool:
                await pool.open(wait=True, timeout=30)
                logger.info("PostgreSQL connection pool opened")
            await checkpointer.setup()
            logger.info("Async PostgreSQL checkpointer initialized")
            checkpointer_ready = True
        except Exception as e:
            logger.error(f"Failed to initialize checkpointer: {e}")
            logger.warning("Will use in-memory checkpointer as fallback")

    use_memory_fallback = not checkpointer_ready
    try:
        if use_memory_fallback:
            logger.warning("Initializing Liquide Agent with in-memory checkpointer")
        initialize_liquide_agent(use_memory_checkpointer=use_memory_fallback)
        logger.info("Liquide Agent initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Liquide Agent: {e}")
        if not use_memory_fallback:
            reset_liquide_agent()
            initialize_liquide_agent(use_memory_checkpointer=True)
            logger.warning("Liquide Agent initialized with in-memory fallback")
        else:
            raise RuntimeError("Critical: Liquide Agent initialization failed.") from e

    try:
        await db_service.initialize()
        logger.info("Conversation history database service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize conversation history service: {e}")

    yield

    logger.info("Shutting down Liquide Agent API...")

    try:
        langfuse_service.flush()
    except Exception as e:
        logger.error(f"Error flushing Langfuse: {e}")

    try:
        await db_service.close()
    except Exception as e:
        logger.error(f"Error closing database service: {e}")

    pool = get_pool()
    if pool:
        try:
            await pool.close()
        except Exception as e:
            logger.error(f"Error closing checkpointer pool: {e}")


app = FastAPI(
    title="Liquide Agent API",
    description="Liquide review RAG assistant with streaming endpoint",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


ROLE_TO_TYPE = {
    "user": "human",
    "assistant": "ai",
    "system": "system",
    "tool": "tool",
}


def _to_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return " ".join(parts)
    return str(content)


def _to_lc_message(msg: dict[str, Any]) -> BaseMessage:
    msg_type = msg.get("type") or ROLE_TO_TYPE.get(str(msg.get("role", "")).lower())
    content = _to_text_content(msg.get("content"))
    if msg_type == "human":
        return HumanMessage(content=content)
    if msg_type == "ai":
        return AIMessage(content=content)
    if msg_type == "system":
        return SystemMessage(content=content)
    if msg_type == "tool":
        return ToolMessage(content=content, tool_call_id=msg.get("tool_call_id", ""))
    return HumanMessage(content=content)


def normalize_input(payload: dict[str, Any]) -> dict[str, Any]:
    data = payload or {}
    if "input" in data:
        inner = data["input"]
        if isinstance(inner, str):
            return {"messages": [HumanMessage(content=inner)]}
        if isinstance(inner, dict):
            if "messages" in inner and isinstance(inner["messages"], list):
                lc_messages = [_to_lc_message(m) for m in inner["messages"]]
                return {"messages": lc_messages}
            if "content" in inner:
                return {"messages": [_to_lc_message(inner)]}
            return inner
    if "messages" in data and isinstance(data["messages"], list):
        lc_messages = [_to_lc_message(m) for m in data["messages"]]
        return {"messages": lc_messages}
    return {"messages": [HumanMessage(content=str(data))]}


class ChatMessage(BaseModel):
    role: str
    content: list[dict[str, Any]] | str


class ChatInput(BaseModel):
    messages: list[ChatMessage]
    caller_number: str | None = None


class StreamRequest(BaseModel):
    input: ChatInput
    thread_id: str | None = None


class ReviewLiveRequest(BaseModel):
    max_pages: int = 2
    fallback_to_sample: bool = True
    run_in_background: bool = True


@app.get("/reviews/ingest/sample")
async def ingest_reviews_sample():
    try:
        await review_rag_service.clear_vector_store_async()
        payload = live_ingest_background_service.load_sample_reviews_payload()
        return await review_rag_service.index_reviews_from_payload_async(payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/reviews/ingest/live")
async def ingest_reviews_live(request: ReviewLiveRequest, background_tasks: BackgroundTasks):
    if request.run_in_background:
        job_id = live_ingest_background_service.create_job(
            max_pages=request.max_pages,
            fallback_to_sample=request.fallback_to_sample,
        )
        background_tasks.add_task(
            live_ingest_background_service.run_job,
            job_id,
            request.max_pages,
            request.fallback_to_sample,
        )
        return {
            "status": "accepted",
            "job_id": job_id,
            "message": "Live review ingestion started in background. This may take a few minutes for large max_pages.",
            "poll_status_endpoint": f"/reviews/ingest/live/jobs/{job_id}",
        }

    try:
        return await live_ingest_background_service.execute_live_ingest(
            max_pages=request.max_pages,
            fallback_to_sample=request.fallback_to_sample,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/reviews/ingest/live/jobs/{job_id}")
async def get_live_ingest_job_status(job_id: str):
    job = live_ingest_background_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@app.post("/reviews/live-preview")
async def preview_live_reviews(request: ReviewLiveRequest):
    google_play_app_id = os.getenv("GOOGLE_PLAY_APP_ID", "life.liquide.app")
    apple_product_id = os.getenv("APPLE_PRODUCT_ID", "1624726081")
    apple_country = os.getenv("APPLE_COUNTRY", "in")
    try:
        items = await review_rag_service.fetch_live_reviews_async(
            google_play_app_id=google_play_app_id,
            apple_product_id=apple_product_id,
            apple_country=apple_country,
            max_pages=request.max_pages,
        )
        if not items:
            raise ValueError("No live reviews fetched from Google Play/App Store")
        return await review_rag_service.preview_reviews_from_payload_async(items)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/reviews/sample-preview")
async def preview_sample_reviews():
    try:
        payload = live_ingest_background_service.load_sample_reviews_payload()
        return await review_rag_service.preview_reviews_from_payload_async(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/reviews/data")
async def get_review_data(limit: int = 20):
    try:
        return await review_rag_service.get_data_async(limit=limit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    
@app.post("/chat/stream")
async def chat_stream(request: StreamRequest):
    try:
        normalized_input = normalize_input(request.input.model_dump())

        user_message_content = ""
        if normalized_input.get("messages"):
            last_message = normalized_input["messages"][-1]
            if isinstance(last_message, HumanMessage):
                user_message_content = last_message.content

        if not request.thread_id or not request.thread_id.strip():
            conversation_id = f"liquide-thread-{uuid.uuid4()}"
        else:
            conversation_id = request.thread_id

        if user_message_content and conversation_id:
            await db_service.append_conversation_message(
                conversation_id=conversation_id,
                role="user",
                content=user_message_content,
                agent_name="LiquideAgent",
            )

        config = {"configurable": {"thread_id": conversation_id}}
        if request.input.caller_number:
            config["configurable"]["caller_number"] = request.input.caller_number

        langfuse_handler = langfuse_service.get_handler()
        if langfuse_handler:
            config["callbacks"] = [langfuse_handler]

        async def generate_stream():
            assistant_response = ""
            # Emit thread_id immediately so UI can show session
            yield f"data: {json.dumps({'thread_id': conversation_id, 'type': 'session', 'timestamp': time.time()})}\n\n"
            try:
                agent = create_liquide_agent_for_request()
                async for event in agent.astream_events(normalized_input, config=config, version="v2"):
                    event_type = event.get("event")
                    event_data = event.get("data", {})
                    event_name = event.get("name", "")

                    if event_type == "on_tool_start":
                        tool_start_data = {
                            "thread_id": conversation_id,
                            "type": "tool_start",
                            "tool_name": event_name,
                            "tool_arguments": event_data.get("input", {}),
                            "timestamp": time.time(),
                        }
                        yield f"data: {json.dumps(tool_start_data)}\n\n"

                    elif event_type == "on_tool_end":
                        output = event_data.get("output")
                        tool_output = getattr(output, "content", output) if output is not None else ""
                        if isinstance(tool_output, (dict, list)):
                            try:
                                tool_response = json.dumps(tool_output)
                            except Exception:
                                tool_response = str(tool_output)
                        elif isinstance(tool_output, str):
                            tool_response = tool_output
                        else:
                            tool_response = str(tool_output)

                        tool_end_data = {
                            "thread_id": conversation_id,
                            "type": "tool_end",
                            "tool_name": event_name,
                            "tool_response": tool_response,
                            "timestamp": time.time(),
                        }
                        yield f"data: {json.dumps(tool_end_data)}\n\n"

                    elif event_type == "on_chat_model_stream":
                        chunk = event_data.get("chunk")
                        if chunk and hasattr(chunk, "content") and chunk.content:
                            text_content = _to_text_content(chunk.content)
                            assistant_response += text_content
                            token_data = {
                                "thread_id": conversation_id,
                                "type": "token",
                                "content": text_content,
                                "timestamp": time.time(),
                            }
                            yield f"data: {json.dumps(token_data)}\n\n"

                    elif event_type == "on_custom_event":
                        if event_name == "say_message" and isinstance(event_data, dict):
                            say_data = {
                                "thread_id": conversation_id,
                                "type": "say",
                                "message": event_data.get("message", ""),
                                "tool": event_data.get("tool", ""),
                                "timestamp": time.time(),
                            }
                            yield f"data: {json.dumps(say_data)}\n\n"

                logger.info(
                    "Assistant response completed for thread_id=%s: %s",
                    conversation_id,
                    assistant_response,
                )

                if assistant_response and conversation_id:
                    await db_service.append_conversation_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=assistant_response,
                        agent_name="LiquideAgent",
                    )

                yield f"data: {json.dumps({'thread_id': conversation_id, 'type': 'end_of_response', 'content': 'end of response', 'timestamp': time.time()})}\n\n"

            except Exception as e:
                import traceback

                logger.error(f"Error in streaming: {traceback.format_exc()}")
                error_data = {"error": str(e), "type": type(e).__name__}
                yield f"data: {json.dumps(error_data)}\n\n"

        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream; charset=utf-8",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/web", response_class=HTMLResponse)
def web_interface():
    """Web-based chat interface for the Liquide Review RAG assistant."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Liquide - Review Assistant</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --liquide-teal: #0D9488;
            --liquide-teal-dark: #0F766E;
            --liquide-teal-light: #14B8A6;
            --liquide-navy: #0E3342;
            --liquide-slate: #1E3A4A;
            --liquide-gold: #D4A853;
            --liquide-silver: #94A3B8;
            --bg-light: #F0FDFA;
            --bg-panel: #FFFFFF;
            --text-dark: #0F172A;
            --text-muted: #64748B;
            --border: #E2E8F0;
        }
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'DM Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(145deg, var(--liquide-navy) 0%, var(--liquide-slate) 50%, var(--liquide-teal-dark) 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .chat-container {
            position: relative;
            width: 100%;
            max-width: 920px;
            height: 92vh;
            background: var(--bg-panel);
            border-radius: 24px;
            box-shadow: 0 25px 80px rgba(13, 148, 136, 0.25), 0 10px 30px rgba(0, 0, 0, 0.2);
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .chat-header {
            background: linear-gradient(135deg, var(--liquide-navy) 0%, var(--liquide-teal-dark) 100%);
            color: white;
            padding: 22px 28px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header-info { display: flex; align-items: center; gap: 16px; }
        .logo {
            width: 52px; height: 52px;
            background: linear-gradient(145deg, var(--liquide-teal-light), var(--liquide-gold));
            border-radius: 14px;
            display: flex; align-items: center; justify-content: center;
            font-weight: 700; color: var(--liquide-navy); font-size: 18px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        .header-text h1 { font-size: 21px; font-weight: 700; margin-bottom: 4px; }
        .header-text p { font-size: 13px; opacity: 0.85; }
        .status-badge { display: flex; align-items: center; gap: 8px; background: rgba(255,255,255,0.15); padding: 8px 14px; border-radius: 20px; }
        .status-indicator { width: 10px; height: 10px; background: #4ade80; border-radius: 50%; animation: pulse 2s infinite; }
        @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
        .chat-messages { flex: 1; overflow-y: auto; padding: 28px; background: linear-gradient(180deg, var(--bg-light) 0%, #FAFAFA 100%); }
        .message { margin-bottom: 18px; display: flex; gap: 14px; animation: slideIn 0.35s ease; }
        @keyframes slideIn { from{opacity:0;transform:translateY(12px)} to{opacity:1;transform:translateY(0)} }
        .message.user { flex-direction: row-reverse; }
        .message-avatar {
            width: 42px; height: 42px; border-radius: 12px;
            display: flex; align-items: center; justify-content: center;
            font-weight: 700; font-size: 14px; flex-shrink: 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .message.user .message-avatar { background: linear-gradient(135deg, var(--liquide-teal), var(--liquide-teal-light)); color: white; }
        .message.assistant .message-avatar { background: linear-gradient(135deg, var(--liquide-navy), var(--liquide-teal-dark)); color: white; }
        .message-content {
            max-width: 72%; padding: 16px 20px; border-radius: 18px;
            line-height: 1.6; word-wrap: break-word; font-size: 15px; white-space: pre-wrap;
        }
        .message.user .message-content {
            background: linear-gradient(135deg, var(--liquide-teal-dark), var(--liquide-teal));
            color: white; border-bottom-right-radius: 6px;
            box-shadow: 0 3px 12px rgba(13,148,136,0.3);
        }
        .message.assistant .message-content {
            background: white; color: var(--text-dark); border-bottom-left-radius: 6px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08); border: 1px solid var(--border);
        }
        .typing-indicator { display: none; padding: 16px 20px; background: white; border-radius: 18px; width: fit-content; }
        .typing-indicator.active { display: flex; gap: 6px; }
        .typing-dot { width: 9px; height: 9px; background: var(--liquide-teal); border-radius: 50%; animation: typing 1.4s infinite; }
        .typing-dot:nth-child(2){animation-delay:0.2s} .typing-dot:nth-child(3){animation-delay:0.4s}
        @keyframes typing { 0%,60%,100%{transform:translateY(0);opacity:0.4} 30%{transform:translateY(-8px);opacity:1} }
        .chat-input-container { padding: 20px 28px 24px; background: white; border-top: 1px solid var(--border); }
        .chat-input-wrapper { display: flex; gap: 14px; align-items: center; }
        #userInput {
            flex: 1; padding: 16px 22px; border: 2px solid var(--border); border-radius: 28px;
            font-size: 15px; font-family: inherit; outline: none; transition: all 0.25s ease; background: var(--bg-light);
        }
        #userInput:focus { border-color: var(--liquide-teal); box-shadow: 0 0 0 4px rgba(13,148,136,0.12); background: white; }
        #sendButton {
            width: 54px; height: 54px;
            background: linear-gradient(135deg, var(--liquide-teal-dark), var(--liquide-teal));
            color: white; border: none; border-radius: 50%; cursor: pointer;
            display: flex; align-items: center; justify-content: center; font-size: 20px;
            box-shadow: 0 4px 14px rgba(13,148,136,0.35); transition: all 0.25s ease;
        }
        #sendButton:hover:not(:disabled) { transform: scale(1.08); box-shadow: 0 6px 20px rgba(13,148,136,0.45); }
        #sendButton:disabled { opacity: 0.5; cursor: not-allowed; }
        .welcome-message { text-align: center; padding: 50px 24px; color: var(--text-muted); }
        .welcome-icon {
            width: 80px; height: 80px;
            background: linear-gradient(145deg, var(--liquide-teal), var(--liquide-teal-light));
            border-radius: 24px; display: flex; align-items: center; justify-content: center;
            margin: 0 auto 24px; font-size: 36px;
            box-shadow: 0 8px 24px rgba(13,148,136,0.3);
        }
        .welcome-message h2 { color: var(--liquide-navy); margin-bottom: 12px; font-size: 26px; font-weight: 700; }
        .welcome-message p { margin-bottom: 8px; font-size: 15px; line-height: 1.6; }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <div class="header-info">
                <div class="logo">LQ</div>
                <div class="header-text">
                    <h1>Liquide Review Assistant</h1>
                    <p>Session: <span id="sessionIdDisplay">-</span></p>
                </div>
            </div>
            <div class="status-badge"><div class="status-indicator"></div><span class="status-text">Online</span></div>
        </div>
        <div class="chat-messages" id="chatMessages">
            <div class="welcome-message">
                <div class="welcome-icon">💬</div>
                <h2>Liquide Review RAG</h2>
                <p>Ask questions about app store reviews.</p>
                <p>I'll search indexed reviews and answer based on real feedback.</p>
            </div>
            <div class="typing-indicator" id="typingIndicator">
                <div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>
            </div>
        </div>
        <div class="chat-input-container">
            <div class="chat-input-wrapper">
                <input type="text" id="userInput" placeholder="Ask about reviews..." autocomplete="off"/>
                <button id="sendButton" onclick="sendMessage()">➤</button>
            </div>
        </div>
    </div>
    <script>
        const chatMessages = document.getElementById('chatMessages');
        const userInput = document.getElementById('userInput');
        const sendButton = document.getElementById('sendButton');
        const typingIndicator = document.getElementById('typingIndicator');
        let threadId = null;
        let isProcessing = false;

        function addMessage(role, content) {
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message ' + role;
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = role === 'user' ? 'U' : 'L';
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = content;
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(contentDiv);
            chatMessages.insertBefore(messageDiv, typingIndicator);
            scrollToBottom();
        }

        function showTyping() { typingIndicator.classList.add('active'); scrollToBottom(); }
        function hideTyping() { typingIndicator.classList.remove('active'); }
        function scrollToBottom() { chatMessages.scrollTop = chatMessages.scrollHeight; }

        async function sendMessage(messageOverride = null, showUserMessage = true) {
            const rawMessage = messageOverride === null ? userInput.value : messageOverride;
            const message = (rawMessage || '').trim();
            if (!message || isProcessing) return;
            isProcessing = true;
            sendButton.disabled = true;
            if (messageOverride === null) {
                userInput.value = '';
            }
            if (showUserMessage) {
                addMessage('user', message);
            }
            showTyping();

            let assistantMessage = '';
            let currentAgentMessage = '';

            try {
                const requestBody = {
                    input: {
                        messages: [{ role: 'user', content: message }],
                        caller_number: null
                    }
                };
                if (threadId) {
                    requestBody.thread_id = threadId;
                }

                const response = await fetch('/chat/stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(requestBody)
                });

                if (!response.ok) throw new Error('HTTP ' + response.status);

                hideTyping();
                const reader = response.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';
                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;
                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\\n');
                    buffer = lines.pop() || '';
                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6);
                            if (!data.trim()) continue;
                            try {
                                const parsed = JSON.parse(data);
                                if (parsed.thread_id && !threadId) {
                                    threadId = parsed.thread_id;
                                    const sessionIdDisplay = document.getElementById('sessionIdDisplay');
                                    if (sessionIdDisplay) sessionIdDisplay.textContent = threadId;
                                }
                                if (parsed.type === 'token') {
                                    currentAgentMessage += parsed.content || '';
                                } else if (parsed.type === 'end_of_response') {
                                    assistantMessage = currentAgentMessage;
                                } else if (parsed.error) {
                                    assistantMessage = 'Error: ' + (parsed.error || 'An error occurred');
                                }
                            } catch (e) { console.error('Parse error:', e); }
                        }
                    }
                }

                if (assistantMessage) {
                    const welcomeMsg = document.querySelector('.welcome-message');
                    if (welcomeMsg) welcomeMsg.remove();
                    addMessage('assistant', assistantMessage);
                } else {
                    const welcomeMsg = document.querySelector('.welcome-message');
                    if (welcomeMsg) welcomeMsg.remove();
                    addMessage('assistant', 'No response received.');
                }
            } catch (error) {
                console.error('Error:', error);
                hideTyping();
                addMessage('assistant', 'Sorry, could not connect to the server.');
            } finally {
                isProcessing = false;
                sendButton.disabled = false;
                userInput.focus();
            }
        }

        async function initializeGreeting() {
            await sendMessage('Hi', false);
        }

        userInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });

        window.addEventListener('load', function() {
            initializeGreeting();
        });
    </script>
</body>
</html>
"""
