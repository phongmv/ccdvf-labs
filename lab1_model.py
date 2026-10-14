"""
Lab 1 – Đo token, so sánh model, bật prompt caching
Chạy: python lab1_models.py
"""

import time
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()  # đọc ANTHROPIC_API_KEY từ file .env
client = Anthropic()

MODELS = ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"]

# Giá USD cho mỗi 1 triệu token. Điền theo trang pricing chính thức:
# https://platform.claude.com/docs/en/about-claude/pricing
PRICES = {
    "claude-haiku-5-5": {"input": 0.0, "output": 0.0},
    "claude-sonnet-5-5": {"input": 0.0, "output": 0.0},
    "claude-opus-5-5": {"input": 0.0, "output": 0.0},
}

EASY = "Phân loại đánh giá sau thành tích cực / tiêu cực / trung tính, chỉ trả về 1 từ: 'Giao hàng chậm nhưng hàng đẹp'"
HARD = (
    "Có 5 ngôi nhà màu khác nhau. Người Anh ở nhà đỏ. Người Tây Ban Nha nuôi chó. "
    "Nhà xanh lá ở ngay bên phải nhà trắng. Người ở nhà giữa uống sữa. Người Na Uy ở nhà đầu tiên, "
    "cạnh nhà xanh dương. Hỏi: nhà đầu tiên màu gì? Giải thích ngắn gọn từng bước."
)


# ---------- Bước 2: hàm gọi model và đo ----------
def run(model: str, prompt: str) -> dict:
    start = time.perf_counter()
    resp = client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = time.perf_counter() - start
    text_block = next((b for b in resp.content if b.type == "text"), None)
    return {
        "model": model,
        "latency_s": round(latency, 2),
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
        "text": text_block.text.strip() if text_block else f"[no text block — stop_reason: {resp.stop_reason}]",
    }


# ---------- Bước 8: ước tính chi phí ----------
def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICES[model]
    return (input_tokens * p["input"] + output_tokens * p["output"]) / 1_000_000


def compare(task_name: str, prompt: str):
    print(f"\n===== {task_name} =====")
    print(f"{'Model':<20}{'Độ trễ(s)':>10}{'In':>8}{'Out':>8}{'Chi phí $':>12}")
    results = []
    for m in MODELS:
        r = run(m, prompt)
        cost = estimate_cost(m, r["input_tokens"], r["output_tokens"])
        print(
            f"{m:<20}{r['latency_s']:>10}{r['input_tokens']:>8}{r['output_tokens']:>8}{cost:>12.6f}"
        )
        results.append(r)
    print("\n--- Câu trả lời (để bạn tự chấm chất lượng) ---")
    for r in results:
        print(f"\n[{r['model']}]\n{r['text'][:600]}")


# ---------- Bước 5: đếm token trước khi gửi ----------
def count_tokens_demo():
    print("\n===== Đếm token trước khi gửi =====")
    for prompt in (EASY, HARD):
        n = client.messages.count_tokens(
            model="claude-sonnet-5-5",
            messages=[{"role": "user", "content": prompt}],
        )
        print(f"{n.input_tokens:>6} token  ←  {prompt[:50]}...")


# ---------- Bước 6–7: prompt caching ----------
def build_long_doc() -> str:
    # Tạo "chính sách đổi trả" giả, đủ dài để vượt ngưỡng tối thiểu của cache
    clause = (
        "Điều {i}. Khách hàng được đổi trả sản phẩm trong vòng 30 ngày kể từ ngày nhận hàng "
        "nếu sản phẩm còn nguyên tem mác, chưa qua sử dụng và có hoá đơn mua hàng hợp lệ. "
        "Phí vận chuyển đổi trả do cửa hàng chịu nếu lỗi thuộc về nhà sản xuất. "
    )
    return "\n".join(clause.format(i=i) for i in range(1, 121))


def caching_demo():
    print("\n===== Prompt caching =====")
    doc = build_long_doc()
    system = [
        {
            "type": "text",
            "text": "Bạn là trợ lý CSKH. Trả lời dựa trên chính sách dưới đây.",
        },
        {"type": "text", "text": doc, "cache_control": {"type": "ephemeral"}},
    ]
    for i in (1, 2):
        start = time.perf_counter()
        resp = client.messages.create(
            model="claude-sonnet-5-5",
            max_tokens=200,
            system=system,
            messages=[
                {
                    "role": "user",
                    "content": "Tôi mua hàng 20 ngày trước, có được đổi không?",
                }
            ],
        )
        u = resp.usage
        print(
            f"Lần {i}: {time.perf_counter() - start:.2f}s | input={u.input_tokens} "
            f"| ghi cache={u.cache_creation_input_tokens} | đọc cache={u.cache_read_input_tokens}"
        )
    print("Kỳ vọng: lần 1 'ghi cache' > 0, lần 2 'đọc cache' > 0.")
    print(
        "Nếu cả hai = 0: tài liệu chưa đủ dài → tăng range(1, 121) trong build_long_doc()."
    )


if __name__ == "__main__":
    count_tokens_demo()
    compare("Tác vụ DỄ – phân loại", EASY)
    compare("Tác vụ KHÓ – suy luận nhiều bước", HARD)
    caching_demo()
