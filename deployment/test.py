from vertexai.preview import reasoning_engines
from agents.agent import root_agent

app = reasoning_engines.AdkApp(
    agent=root_agent,
    enable_tracing=True,
)

session = app.create_session(user_id="u_123")
print(session)
print(session.id)
print(app.list_sessions(user_id="u_123"))

session = app.get_session(user_id="u_123", session_id=session.id)

for event in app.stream_query(
    user_id="u_123",
    session_id=session.id,
    message="Help me infernece the NIFTI file provided.",
):
    print(event)