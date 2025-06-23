import os
import vertexai
from absl import app, flags
from dotenv import load_dotenv
from vertexai import agent_engines

FLAGS = flags.FLAGS

flags.DEFINE_string("project_id", None, "GCP project ID.")
flags.DEFINE_string("location", None, "GCP location.")
flags.DEFINE_string("bucket", None, "GCP bucket.")
flags.DEFINE_string(
    "resource_id",
    None,
    "AgentEngine resource ID (projects/.../agentEngines/...)",
)
flags.DEFINE_string("user_id", None, "User ID for this session.")

flags.mark_flag_as_required("resource_id")
flags.mark_flag_as_required("user_id")

def main(argv: list[str]) -> None:
    del argv  # unused
    load_dotenv()

    project_id = FLAGS.project_id or os.getenv("GOOGLE_CLOUD_PROJECT")
    location = FLAGS.location or os.getenv("GOOGLE_CLOUD_LOCATION")
    bucket = FLAGS.bucket or os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET")

    if not project_id or not location or not bucket:
        print("Missing required environment variables.")
        return

    vertexai.init(
        project=project_id,
        location=location,
        staging_bucket=f"gs://{bucket}",
    )

    print(f"Using AgentEngine: {FLAGS.resource_id}")
    agent = agent_engines.get(FLAGS.resource_id)

    session = agent.create_session(user_id=FLAGS.user_id)
    print(f"Session created for user: {FLAGS.user_id}")
    print("\nType your input below. Type 'quit' to exit.\n")

    try:
        while True:
            user_input = input("Input: ")
            if user_input.strip().lower() == "quit":
                break

            for event in agent.stream_query(
                user_id=FLAGS.user_id,
                session_id=session["id"],
                message=user_input
            ):
                if "content" in event and "parts" in event["content"]:
                    for part in event["content"]["parts"]:
                        if "text" in part:
                            print(f"Response: {part['text']}")
    finally:
        agent.delete_session(user_id=FLAGS.user_id, session_id=session["id"])
        print(f"Session deleted for user: {FLAGS.user_id}")

if __name__ == "__main__":
    app.run(main)