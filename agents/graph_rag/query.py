import json
from agents.llm_client.gemini_client import gemini_chat


def build_cypher_prompt(user_question: str, schema: str) -> str:
    """
    根據使用者問題與圖譜 schema 建構 prompt，要求 LLM 回傳 Cypher 查詢語句。
    """
    return f"""
        You are a Cypher expert. Given the following graph schema and a user question, generate a valid Cypher query.

        Schema format:
        {schema}

        User question:
        {user_question}

        Instructions:
        - Generate **only** a Cypher query that answers the user's question.
        - Do NOT explain the query.
        - Only output the Cypher code block, no markdown.
        - Use fuzzy search if exact match may not be guaranteed (e.g. WHERE toLower(r.name) CONTAINS '...').
        - Prefer pattern-matching over property exact match.

        Cypher Query:
    """


def generate_cypher_query(prompt: str) -> str:
    """
    呼叫 Gemini，輸出純文字 Cypher 查詢語句。
    """
    return gemini_chat(prompt=prompt, mime_type="text/plain")


def get_cypher_from_question(user_question: str, schema: dict) -> str:
    """
    高階封裝函式：從問題與 schema 推論 Cypher 查詢。
    """
    prompt = build_cypher_prompt(user_question, schema)
    cypher_query = generate_cypher_query(prompt)
    return cypher_query


if __name__ == "__main__":
    example_schema = """
        This graph database schema describes a structure centered around **brain regions, networks, and their functions**.

        Here's a breakdown:

        1.  **Kinds of entities (node types):**
            *   `YeoNetwork`: Represents specific brain networks, possibly derived from the Yeo parcellation.
            *   `Function`: Describes a particular cognitive or physiological function.
            *   `Region`: Represents an anatomical brain region.

        2.  **Relationships between entities:**
            *   `:BELONGS_TO`: Indicates a membership or association relationship, likely between a `Region` and a `YeoNetwork`.
            *   `:HAS_FUNCTION`: Links an entity (likely `Region` or `YeoNetwork`) to a `Function`.

        3.  **Properties of each node/relationship type:**
            *   **Nodes:**
                *   `YeoNetwork`: Has a `name` (String), a unique `yeo_id` (Long), and a `label` (String).
                *   `Function`: Has a `name` (String) and a `label` (String).
                *   `Region`: Has a `name` (String), an `ad_associated` flag (Boolean, possibly indicating association with Alzheimer's Disease or similar), and a `label` (String).
            *   **Relationships:**
                *   `:BELONGS_TO`: Has a `percentage` property (Double), indicating the degree or proportion of belonging.
                *   `:HAS_FUNCTION`: Has no specific properties.

        4.  **High-level summary:**
            This graph schema appears to model **brain organization and functionality**, specifically focusing on how different **brain regions** relate to established **brain networks (YeoNetworks)** and what **functions** they perform. The `ad_associated` property on `Region` and the `percentage` on `BELONGS_TO` suggest it could be used for research in neuroscience, potentially investigating the involvement of specific regions or networks in conditions like Alzheimer's disease, or mapping functional contributions of brain areas to networks.
    """

    # Test query
    question = "Is it possible that Angular are related to Alzheimer's disease? If so, what are the functions of these regions?"

    print("User Question:")
    print(question)

    cypher_query = get_cypher_from_question(question, example_schema)

    print("\nGenerated Cypher Query:")
    print(cypher_query)
