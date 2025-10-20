import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
load_dotenv()

# 從環境變數讀取 Neo4j 的連接資訊
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

def get_neo4j_driver():
    """return a Neo4j driver instance"""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

