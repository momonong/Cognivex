from vertexai.preview import reasoning_engines
from vertexai import agent_engines, init
import os
from dotenv import load_dotenv

from agents.agent import root_agent

load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS", "./gcp-service-account.json"
)

# 1. 封裝 root_agent
app = reasoning_engines.AdkApp(
    agent=root_agent, enable_tracing=True  # 替換為您的 root_agent 實例  # 啟用追蹤功能
)

# 2. 初始化 Vertex AI 環境
init(
    project=os.getenv("PROJECT_ID"),  # 您的 GCP 專案 ID
    location="us-central1",  # 部署區域
    staging_bucket=os.getenv("BUCKET_PATH"),  # 替換為您的 Cloud Storage 儲存桶
)

# 3. 部署到 Agent Engine
remote_agent = agent_engines.create(
    app,
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
    env_vars={
        "NEO4J_URI": os.getenv("NEO4J_URI"),
        "NEO4J_USER": os.getenv("NEO4J_USER"),
        "NEO4J_PASSWORD": os.getenv("NEO4J_PASSWORD"),
    },
    display_name="fMRI Explain Agent",
    description="結合知識圖譜的 RAG 代理",
)

# 4. 取得部署資源 ID
print(f"部署成功！資源 ID: {remote_agent.resource_name}")
# 輸出範例: projects/123456/locations/us-central1/reasoningEngines/abcdefg
