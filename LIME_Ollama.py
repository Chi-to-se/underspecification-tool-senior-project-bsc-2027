import base64
from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


# GLOBAL CONFIGURATION
# ====================
MODEL = "gemma4:12b"
TEMPERATURE = 1
image = "results/result_xception_dog.png"
# ====================


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def call_llm_for_response(image_path: str, session_id: str = None) -> str:
    """
    Multi-turn approach (mirrors working notebook):
      Round 1: Send task description (text only)
      Round 2: Send image only — model reads image with prior context in memory
    """
    # Dynamic session ID so each call gets a clean memory
    if session_id is None:
        import time
        session_id = f"lime_{int(time.time() * 1000)}"

    # Initialize a LLM Model
    llm = ChatOllama(model=MODEL, temperature=TEMPERATURE)

    # Set up memory for multi-turn conversation
    mem = {}
    def get_session_history(sid: str):
        if sid not in mem:
            mem[sid] = InMemoryChatMessageHistory()
        return mem[sid]

    chain = RunnableWithMessageHistory(llm, get_session_history)
    config = {"configurable": {"session_id": session_id}}

    # --- Round 1: Task description (no image) ---
    messages_1 = [
        SystemMessage(
            content="**Role:** You are an expert AI Computer Vision Analyst specializing in Explainable AI (XAI) and LIME (Local Interpretable Model-agnostic Explanations) interpretation."
        ),
        HumanMessage(content="""
**Task:** Analyze an evaluation image containing LIME explanations and extract visual features (super-pixels) that influenced the model's classification.

**Input Image Layout:** Single row, 4 columns:
1. **Model Name:** The classification model being tested.
2. **Original Image:** The raw input image.
3. **Super-Pixel Overlay:** LIME overlay — Green = Positive, Red = Negative features.
4. **Masked Image:** Only top influential super-pixels shown.

**Instructions:**
1. Identify the object/class being evaluated.
2. Inspect Column 3 (overlay) and Column 4 (masked).
3. Map colored regions to real-world parts, objects, or textures.
4. List features as Positive and Negative sets.

**Output Format (respond strictly):**
**Model Name:** [name]
**Positive Feature Set (Green):** [e.g., Head, Whiskers, Left Eye]
**Negative Feature Set (Red):** [e.g., Background Grass, Shadow, Collar]
        """)
    ]

    # --- Round 2: Image only ---
    base64_image = encode_image(image_path)
    messages_2 = [
        HumanMessage(
            content=[
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                }
            ]
        )
    ]

    chain.invoke(input=messages_1, config=config)
    response = chain.invoke(input=messages_2, config=config)

    return response.content


# Core loop
if __name__ == "__main__":
    response = call_llm_for_response(image)
    print(response)
