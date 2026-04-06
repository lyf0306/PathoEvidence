# local_deep_research/config.py
import logging
import os
from types import SimpleNamespace
from langchain_openai import ChatOpenAI
from pathlib import Path
import tomllib

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH  = PROJECT_ROOT / "_settings" / ".secrets.toml"

# 安全加载配置
if CONFIG_PATH.exists():
    with CONFIG_PATH.open("rb") as f:
        secrets = tomllib.load(f)
else:
    print(f"⚠️ Warning: Config file not found at {CONFIG_PATH}")
    secrets = {}

def get_secret(section, key, default=""):
    return secrets.get(section, {}).get(key, default)

settings = SimpleNamespace(
    quick    = SimpleNamespace(iteration=2, questions_per_iteration=4),
    detailed = SimpleNamespace(iteration=2, questions_per_iteration=6),
    embedding_api_key = get_secret("embedding", "api_key", "EMPTY_KEY"),
    embedding_cache   = get_secret("embedding", "cache", "embedding_cache.pkl"),
)

endpoint_openai_api_base_url   = get_secret("openai", "api_base", "https://api.openai.com/v1")
endpoint_openai_api_key        = get_secret("openai", "api_key", "EMPTY_KEY")

deepseek__openai_api_base_url  = get_secret("deepseek", "api_base", "https://api.deepseek.com")
deepseek_openai_api_key        = get_secret("deepseek", "api_key", "EMPTY_KEY")

# 🔧 修正MCP服务器URL和端口
mcp_url = get_secret("mcp", "server_url", "http://localhost:8788")  # ✅ 改为8788

template_embedding_api_base_url = get_secret("template", "api_base", "")
template_embedding_api_key      = get_secret("template", "api_key", "EMPTY_KEY")

def get_gpt4_1() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4.1",
        api_key=endpoint_openai_api_key,
        openai_api_base=endpoint_openai_api_base_url,
        temperature=0.6,
        top_p=0.9,
        max_tokens=32000,
    )

def get_gpt4_1_mini() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4.1-mini",
        api_key=endpoint_openai_api_key,
        openai_api_base=endpoint_openai_api_base_url,
        temperature=0.6,
        top_p=0.9,
        max_tokens=32000,
    )

def get_claude_openai() -> ChatOpenAI:
    return ChatOpenAI(
        model="claude-3-opus-20240229",
        api_key=endpoint_openai_api_key,
        openai_api_base=endpoint_openai_api_base_url,
        temperature=0.6,
        top_p=0.9,
        max_tokens=32000,
    )

def get_deepseek_r1() -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek-reasoner",
        api_key=deepseek_openai_api_key,
        openai_api_base=deepseek__openai_api_base_url,
        temperature=0.6,
        top_p=0.9,
        max_tokens=32000,
    )

def get_deepseek_v3() -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek-chat",
        api_key=deepseek_openai_api_key,
        openai_api_base=deepseek__openai_api_base_url,
        temperature=0.6,
        top_p=0.9,
        max_tokens=32000,
    )

def get_local_model(temperature: float = 0.1, request_timeout: float = 600.0):
    """
    连接本地 vLLM 部署的模型。
    """
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("Please install langchain_openai")
    return ChatOpenAI(
        model="OriClinical", 
        base_url="http://localhost:8000/v1",
        api_key="EMPTY", 
        temperature=temperature,
        max_tokens=16384,
        request_timeout=request_timeout, # ✅ 允许传递超时时间
    )
