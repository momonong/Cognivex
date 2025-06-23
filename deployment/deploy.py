import os
import vertexai
from dotenv import load_dotenv
from absl import app, flags
from vertexai import agent_engines
from vertexai.preview.reasoning_engines import AdkApp
from agents.agent import root_agent  

FLAGS = flags.FLAGS
flags.DEFINE_string("project_id", None, "GCP project ID.")
flags.DEFINE_string("location", None, "GCP location.")
flags.DEFINE_string("bucket", None, "GCP bucket.")
flags.DEFINE_string("resource_id", None, "ReasoningEngine resource ID.")

flags.DEFINE_bool("list", False, "List all agents.")
flags.DEFINE_bool("create", False, "Creates a new agent.")
flags.DEFINE_bool("delete", False, "Deletes an existing agent.")
flags.mark_bool_flags_as_mutual_exclusive(["create", "delete"])


def create() -> None:
    """Creates an agent engine."""
    adk_app = AdkApp(agent=root_agent, enable_tracing=True)

    remote_agent = agent_engines.create(
        adk_app,
        display_name=root_agent.name,
        requirements=[
            # Google Agent Engine & ADK
            "google-adk>=1.3.0",
            "google-cloud-aiplatform[agent_engines]>=1.95.1,<2.0.0",
            # Generative AI
            "google-genai>=1.17.0,<2.0.0",
            # GraphDB and Knowledge Graph
            "neo4j>=5.28.0,<6.0.0",
            # LLM and RAG
            "langchain-google-genai",
            # Pydantic for data validation
            "pydantic>=2.10.6,<3.0.0",
            "absl-py>=2.2.1,<3.0.0",
            # .env Management
            "python-dotenv>=1.0.0",
            # Google Cloud Storage
            "google-cloud-storage>=2.18.0,<3.0.0",
            # Data Science Libraries
            "numpy>=1.24.0",
            "pandas>=2.2.0",
            "scikit-learn>=1.4.0",
            "matplotlib>=3.10.0",
        ],
    )
    print(f"Created remote agent: {remote_agent.resource_name}")


def delete(resource_id: str) -> None:
    remote_agent = agent_engines.get(resource_id)
    remote_agent.delete(force=True)
    print(f"Deleted remote agent: {resource_id}")


def list_agents() -> None:
    remote_agents = agent_engines.list()
    template = """
        {agent.name} ("{agent.display_name}")
        - Create time: {agent.create_time}
        - Update time: {agent.update_time}
        - Resource name: {agent.resource_name}
        """
    remote_agents_string = "\n".join(
        template.format(agent=agent) for agent in remote_agents
    )
    print(f"All remote agents:\n{remote_agents_string}")


def main(argv: list[str]) -> None:
    del argv  # unused
    load_dotenv()

    project_id = FLAGS.project_id or os.getenv("PROJECT_ID")
    location = FLAGS.location or os.getenv("LOCATION")
    bucket = FLAGS.bucket or os.getenv("BUCKET_ID")

    print(f"PROJECT: {project_id}")
    print(f"LOCATION: {location}")
    print(f"BUCKET: {bucket}")

    if not all([project_id, location, bucket]):
        print("Missing required environment variables: project_id, location, bucket")
        return

    vertexai.init(
        project=project_id,
        location=location,
        staging_bucket=f"gs://{bucket}",
    )

    if FLAGS.list:
        list_agents()
    elif FLAGS.create:
        create()
    elif FLAGS.delete:
        if not FLAGS.resource_id:
            print("resource_id is required for delete")
            return
        delete(FLAGS.resource_id)
    else:
        print("Unknown command")


if __name__ == "__main__":
    app.run(main)