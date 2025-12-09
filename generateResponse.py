import requests
import json
from typing import Optional, Dict, Any

def call_gemini_api(
    api_key: str,
    gemini_user_prompt: str,
    gemini_system_instruction: Optional[Dict[str, Any]] = None,
    gemini_max_tokens: int = 8192,
    gemini_temperature: float = 1.0,
    gemini_top_p: float = 0.95,
    gemini_top_k: int = 40,
    gemini_candidate_count: int = 1,
    gemini_safety_threshold: str = "BLOCK_ONLY_HIGH",
    model: str = "gemini-3-pro-preview"
) -> Dict[str, Any]:
    """
    Make an API call to Google Gemini.
    
    Args:
        api_key: Your Google API key
        gemini_user_prompt: The user prompt text
        gemini_system_instruction: System instruction object (optional)
        gemini_max_tokens: Maximum output tokens
        gemini_temperature: Temperature for generation
        gemini_top_p: Top-p sampling parameter
        gemini_top_k: Top-k sampling parameter
        gemini_candidate_count: Number of candidates to generate
        gemini_safety_threshold: Safety threshold level
        model: Gemini model to use
    
    Returns:
        API response as a dictionary
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": gemini_user_prompt}
                ]
            }
        ],
        "generationConfig": {
            "maxOutputTokens": gemini_max_tokens,
            "temperature": gemini_temperature,
            "topP": gemini_top_p,
            "topK": gemini_top_k,
            "candidateCount": gemini_candidate_count
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": gemini_safety_threshold},
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": gemini_safety_threshold},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": gemini_safety_threshold},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": gemini_safety_threshold}
        ]
    }
    
    if gemini_system_instruction:
        payload["systemInstruction"] = gemini_system_instruction
    
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    
    return response.json()


if __name__ == "__main__":
    # Example usage
    API_KEY = "AIzaSyD_Zw7NFoqMcIWu0KQBvacmhJq5arbV4GQ  "
    
    result = call_gemini_api(
        api_key=API_KEY,
        gemini_user_prompt="Hello, how are you?",
        gemini_system_instruction={"parts": [{"text": "You are a helpful assistant."}]},
        gemini_max_tokens=1024,
        gemini_temperature=0.7
    )
    
    print(json.dumps(result, indent=2))