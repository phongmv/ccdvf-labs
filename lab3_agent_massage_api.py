import anthropic

client = anthropic.Anthropic()

# Gửi yêu cầu khởi đầu tới Claude
response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=1024,
    messages=[{"role": "user", "content": "Hi!"}],
)

print("Stop Reason:", response.stop_reason)
print("Content:", response.content)
