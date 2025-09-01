from google import genai
from google.genai import types
import pathlib, json5

def parse_pdf_with_gemini(pdf_path: str, model="gemini-2.5-flash"):
    client = genai.Client()

    prompt = """
    You are a document parsing assistant.
    Please extract and return:
    - The full text of the document in markdown,
    - Any tables in valid JSON format (headers + rows),
    - Describe any embedded images.
    Respond as valid JSON with fields: 'text','tables','images'.
    """

    response = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(
                data=pathlib.Path(pdf_path).read_bytes(),
                mime_type='application/pdf',
            ),
            prompt
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    # clean json
    text = response.text.strip()
    if text.startswith("```json"): text = text[7:]
    if text.endswith("```"): text = text[:-3]
    data = json5.loads(text)
    return data
