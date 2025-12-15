"""
该模块提供了一个函数 `create_client`，用于根据指定的提供方创建 OpenAI 兼容的 API 客户端。
支持的提供方包括 OpenAI、SiliconFlow 和 DeepSeek。

支持的提供方：
- openai：从 `OPENAI_API_KEY`（可选 `OPENAI_BASE_URL`）读取
- siliconflow：从 `SILICONFLOW_API_KEY`（可选 `SILICONFLOW_BASE_URL`，默认 `https://api.siliconflow.cn/v1`）读取
- deepseek：从 `DEEPSEEK_API_KEY`（可选 `DEEPSEEK_BASE_URL`，默认 `https://api.deepseek.com`）读取
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

# 加载.env文件
load_dotenv()

# 如果.env文件不在当前目录，尝试在项目根目录查找
env_path = Path(__file__).parent.parent / '.env'
if env_path.exists():
    load_dotenv(env_path)

def create_client(provider, api_key=None, base_url=None):
    """根据指定提供方创建 OpenAI 兼容客户端

    参数：
    - provider: 提供方标识（'openai' | 'siliconflow' | 'deepseek'，大小写不敏感）
    - api_key: 可选，若为空则从对应环境变量读取
    - base_url: 可选，若为空则从对应环境变量读取；部分提供方有默认值

    返回：
    - OpenAI 客户端实例，可用于 `chat.completions.create(...)` 等接口

    异常：
    - ValueError: 当提供方不受支持时抛出
    """
    p = (provider or "").lower()
    if p == "openai":
        key = api_key or os.getenv("OPENAI_API_KEY")# 此处将api_key填写为你的OPENAI_API_KEY
        return OpenAI(api_key=key) 
    if p == "siliconflow":
        key = api_key or os.getenv("SILICONFLOW_API_KEY")# 此处将api_key填写为你的SILICONFLOW_API_KEY
        url = base_url or os.getenv("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1"
        return OpenAI(api_key=key, base_url=url)
    if p == "deepseek":
        key = api_key or os.getenv("DEEPSEEK_API_KEY")# 此处将api_key填写为你的DEEPSEEK_API_KEY
        url = base_url or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com"
        return OpenAI(api_key=key, base_url=url)
    raise ValueError("unsupported provider")

