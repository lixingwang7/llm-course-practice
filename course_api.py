"""真实 API 调用与课堂显示工具。密钥仅从资料目录 .env / 环境变量读取。"""
from pathlib import Path
import os, json, base64, mimetypes, time
import requests
from dotenv import load_dotenv
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env', override=False)

class CourseAPIError(RuntimeError):
    pass

def pretty(value):
    print(json.dumps(value, ensure_ascii=False, indent=2, default=str))

def answer(text):
    from IPython.display import display, Markdown
    display(Markdown(str(text)))

def config_status(embedding=False):
    result = {'Python配置': '请同时检查 sys.executable',
              '对话模型': os.getenv('LLM_MODEL', 'deepseek-flash'),
              '对话地址': os.getenv('LLM_BASE_URL', 'https://api.deepseek.com'),
              '对话密钥已配置': bool(os.getenv('LLM_API_KEY'))}
    if embedding:
        result.update({'向量模型': os.getenv('EMBEDDING_MODEL', 'text-embedding-v4'),
                       '向量地址': os.getenv('EMBEDDING_BASE_URL', ''),
                       '向量密钥已配置': bool(os.getenv('DASHSCOPE_API_KEY'))})
    return result

def _post(base, route, key, payload):
    if not key or key.startswith('请'):
        raise CourseAPIError('请先在本目录 .env 中配置相应 API Key，然后重启内核。')
    if not base or '你的' in base or '{' in base:
        raise CourseAPIError('请从服务控制台复制本业务空间的 API 地址，填写 .env。')
    try:
        r = requests.post(base.rstrip('/') + route,
            headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
            json=payload, timeout=(10, 90))
    except requests.RequestException as exc:
        raise CourseAPIError('网络连接失败或超时，请检查网络、服务地址及代理。') from None
    if not r.ok:
        hints = {400: '检查模型能力、请求字段或图片格式', 401: '检查密钥及所属服务',
                 402: '检查账户余额', 403: '检查模型权限和地域',
                 404: '检查 API 地址和模型名', 429: '请求过快，请稍后再试'}
        raise CourseAPIError(f'HTTP {r.status_code}：' + hints.get(r.status_code, '服务异常，请稍后重试'))
    return r.json()

def chat(messages, *, json_mode=False, tools=None, max_tokens=1200):
    """Chat Completions，关闭思考以便讲解工具消息协议。每次调用均请求真实服务。"""
    payload = {'model': os.getenv('LLM_MODEL', 'deepseek-flash'), 'messages': messages,
               'max_tokens': max_tokens, 'temperature': 0.2, 'thinking': {'type': 'disabled'}}
    if json_mode:
        payload['response_format'] = {'type': 'json_object'}
    if tools:
        payload.update(tools=tools, tool_choice='auto')
    started = time.perf_counter()
    result = _post(os.getenv('LLM_BASE_URL', 'https://api.deepseek.com'),
                   '/chat/completions', os.getenv('LLM_API_KEY'), payload)
    choice = result['choices'][0]
    if choice.get('finish_reason') == 'length':
        raise CourseAPIError('输出达到 max_tokens 限制，请缩短任务或适当提高上限后重试。')
    return {'message': choice['message'], 'usage': result.get('usage', {}),
            'seconds': round(time.perf_counter() - started, 2)}

def text(messages, **kwargs):
    result = chat(messages, **kwargs)
    content = result['message'].get('content')
    if not content:
        raise CourseAPIError('模型未返回正文。请检查工具调用结果或缩短任务后重试。')
    return content

def image_part(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('请选择不超过 8MB 的课堂图片。')
    mime = mimetypes.guess_type(path.name)[0]
    if mime not in {'image/png', 'image/jpeg', 'image/webp', 'image/gif'}:
        raise ValueError('仅接收 PNG/JPEG/WebP/GIF 图片。')
    encoded = base64.b64encode(path.read_bytes()).decode('ascii')
    return {'type': 'image_url', 'image_url': {'url': f'data:{mime};base64,{encoded}'}}

def vision(path, prompt):
    return text([{'role': 'user', 'content': [
        {'type': 'text', 'text': prompt}, image_part(path)]}])

def embed(texts):
    """百炼 OpenAI 兼容向量 API。v4 每批最多10条，保持响应索引顺序。"""
    texts = list(texts)
    vectors = []
    for start in range(0, len(texts), 10):
        batch = texts[start:start+10]
        payload = {'model': os.getenv('EMBEDDING_MODEL', 'text-embedding-v4'),
                   'input': batch, 'dimensions': int(os.getenv('EMBEDDING_DIMENSIONS', '1024')),
                   'encoding_format': 'float'}
        data = _post(os.getenv('EMBEDDING_BASE_URL', ''), '/embeddings',
                     os.getenv('DASHSCOPE_API_KEY'), payload)['data']
        ordered = sorted(data, key=lambda x: x['index'])
        if len(ordered) != len(batch):
            raise CourseAPIError('向量响应条数与输入不一致。')
        vectors.extend(row['embedding'] for row in ordered)
    return vectors
