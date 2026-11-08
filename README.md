# LLM Course Practice

大模型应用开发练习仓库，围绕三条主线：**图像理解**、**结构化输出**、**多轮对话与调试**。
代码可直接运行，换掉 `.env` 里的接口地址即可对接任意 OpenAI 兼容服务。

## 内容

| 文件 | 说明 |
| --- | --- |
| `01_图像理解.ipynb` | 构造图文混合消息，检查模型实际看到的内容，区分「观察」与「推断」 |
| `02_结构化输出.ipynb` | 用 Pydantic 定义 Schema，把模型输出约束成可直接渲染的强类型 JSON |
| `03_多轮对话与调试.ipynb` | 会话状态与上下文管理，以及多轮场景下的排查方法 |
| `course_api.py` | 统一的 API 封装，配置全部从 `.env` 读取 |

## 运行

1. `python -m venv .venv`
2. `.venv\Scripts\activate`（macOS / Linux 用 `source .venv/bin/activate`）
3. `pip install -r requirements.txt`
4. 复制 `.env.example` 为 `.env`，填入自己的 `LLM_API_KEY`
5. `jupyter lab`

## 说明

- 仓库内不含任何真实密钥，`.env` 已被 `.gitignore` 忽略。
- 默认对接 DeepSeek 的 OpenAI 兼容接口，模型名 `deepseek-flash`，可在 `.env` 中替换。
- 每次运行都会产生真实请求和费用，重跑前留意额度。
