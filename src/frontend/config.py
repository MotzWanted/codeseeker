SAMPLING_PARAMS = {"temperature": 0.9}


CHUNKING_TEMPLATE = [
    {
        "role": "user",
        "content": """You are a world-renowned linguist and expert in natural language processing.
Your task is to segment a transcript into coherent and meaningful chunks.

Instruction guidelines:
- Decompose the transcript into a list of distinct and logically complete segments, each represented as a string.
- Ensure each chunk encapsulates a single, coherent idea or topic.
- Make sure the segmentation maintains the natural flow and structure of the conversation.
- Pay special attention to semantic boundaries to preserve the integrity and clarity of the content.

Transcript: "{{ transcript }}"
""",
    }
]

ALIGMENT_PARAMS = {
    "prompt_name": "loft",
    "num_shots": 0,
    "temperature": 0.1,
    "token_limit": 8000,
    "seed": 42,
}


MAX_ATTEMPTS = 5
