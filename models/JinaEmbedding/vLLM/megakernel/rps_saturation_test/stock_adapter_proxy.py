import asyncio
import aiohttp
from aiohttp import web
import sys

VLLM_URL = "http://127.0.0.1:8001"
MODEL_NAME = "jinaai/jina-embeddings-v2-small-en"

async def init_app():
    app = web.Application(client_max_size=10 * 1024 * 1024)
    session = aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=1000, limit_per_host=1000, keepalive_timeout=60)
    )
    app["session"] = session

    async def handle_prompt(request):
        try:
            data = await request.json()
            text_input = data.get("text", data.get("input", ""))
            payload = {
                "model": MODEL_NAME,
                "input": text_input,
                "truncate_prompt_tokens": 2048
            }
            async with app["session"].post(f"{VLLM_URL}/v1/embeddings", json=payload) as resp:
                resp_json = await resp.json()
                return web.json_response(resp_json, status=resp.status)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_health(request):
        try:
            async with app["session"].get(f"{VLLM_URL}/health") as resp:
                if resp.status == 200:
                    return web.json_response({"status": "ok"})
                return web.json_response({"status": "starting"}, status=503)
        except Exception:
            return web.json_response({"status": "starting"}, status=503)

    async def cleanup(app):
        await app["session"].close()

    app.router.add_post("/prompt_c2", handle_prompt)
    app.router.add_post("/mcp_c2", handle_prompt)
    app.router.add_post("/v1/embeddings", handle_prompt)
    app.router.add_get("/health", handle_health)
    app.router.add_get("/healthz", handle_health)
    app.on_cleanup.append(cleanup)
    return app

if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"Starting PANW Async Adapter Proxy (BF16, max-model-len=2048) on port {port} -> forwarding to {VLLM_URL}")
    web.run_app(init_app(), host="0.0.0.0", port=port)
