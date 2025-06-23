from pathlib import Path
import mimetypes
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

# 替換為你的 API Key（如果尚未設置過）
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 選擇模型（推薦使用 gemini-1.5-flash 或 gemini-1.5-pro）
client = genai.Client(api_key=GOOGLE_API_KEY)
model = 'gemini-2.5-flash'  # 或 'gemini-2.5-pro'，根據需要選擇

# 圖片路徑（請換成你電腦上任一張 PNG）
image_path = Path("figures/agent_test/agent_test_capsnet_conv3/activation_map_mosaic.png")

# 讀取圖片
with open(image_path, "rb") as f:
    image_data = f.read()

# 自動判別 MIME type
mime_type, _ = mimetypes.guess_type(str(image_path))

# 發送請求
response = client.models.generate_content(
    model=model,
    contents=[
        {
            "role": "user",
            "parts": [
                {"text": "What does this brain image show?"},
                {"inline_data": {"mime_type": mime_type, "data": image_data}},
            ],
        }
    ]
)

# 顯示結果
print(response.text)
