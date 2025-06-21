from agents.llm_client.gemini_client import gemini_image

response = gemini_image(
    image_path="figures/module_test/module_test_capsnet_conv3/activation_map_mosaic.png",
    prompt=(
        "This is a brain activation map derived from an fMRI-based deep learning model. "
        "Please analyze and explain the significance of the highlighted regions, including:\n"
        "1. Which brain areas show the strongest activation?\n"
        "2. What cognitive functions or neurological roles are these areas commonly associated with?\n"
        "3. Are there any symmetric or asymmetric patterns?\n"
        "4. How might these findings relate to Alzheimer's disease or other cognitive disorders?"
    )
)

print(response)