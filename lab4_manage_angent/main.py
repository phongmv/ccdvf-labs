import anthropic

AGENT_ID = "agent_01K3PtAfs9GqJvWTyFaRSD9y"
ENVIRONMENT_ID = "env_015cZxtxriMJE4SsEv7SPfTL"

client = anthropic.Anthropic()

session = client.beta.sessions.create(
    agent=AGENT_ID,
    environment_id=ENVIRONMENT_ID,
    title="Quickstart session",
)

print(f"Session ID: {session.id}")

# Mở luồng, gửi tin nhắn, rồi đọc phản hồi
with client.beta.sessions.events.stream(session.id) as stream:
    # Gửi tin nhắn SAU khi đã mở luồng
    client.beta.sessions.events.send(
        session.id,
        events=[
            {
                "type": "user.message",
                "content": [
                    {
                        "type": "text",
                        "text": "Create a Python script that generates the first 20 Fibonacci numbers and saves them to fibonacci.txt",
                    },
                ],
            },
        ],
    )

    # Xử lý từng sự kiện khi nó đến
    for event in stream:
        match event.type:
            case "agent.message":
                for block in event.content:
                    if block.type == "text":
                        print(block.text, end="")
            case "agent.tool_use":
                print(f"\n[Using tool: {event.name}]")
            case "session.status_idle":
                print("\n\nAgent finished.")
                break
