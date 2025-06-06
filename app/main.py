from app.kg_query import BrainKG

if __name__ == "__main__":
    kg = BrainKG()  # 換成你的密碼
    region = "Angular_R"
    functions = kg.get_region_function(region)

    print(f"🧠 {region} 的功能有：")
    for func in functions:
        print(f" - {func}")

    kg.close()
