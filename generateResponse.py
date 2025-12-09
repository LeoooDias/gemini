import requests
import json
import sys
import base64
import mimetypes
from datetime import datetime
from typing import Optional, Dict, Any, List

def call_gemini_api(
    api_key: str,
    gemini_user_prompt: str,
    gemini_max_tokens: int,
    gemini_system_instruction: Optional[Dict[str, Any]] = None,
    gemini_inline_data: Optional[Dict[str, str]] = None,
    gemini_temperature: float = 1.0,
    gemini_top_p: float = 0.95,
    gemini_top_k: int = 40,
    gemini_candidate_count: int = 1,
    gemini_safety_threshold: str = "BLOCK_ONLY_HIGH",
    gemini_api_version: str = "v1beta",
    model: str = "gemini-2.5-pro"
) -> Dict[str, Any]:
    """
    Make an API call to Google Gemini.
    
    Args:
        api_key: Your Google API key
        gemini_user_prompt: The user prompt text
        gemini_max_tokens: Maximum output tokens
        gemini_system_instruction: System instruction object (optional)
        gemini_inline_data: Inline data object with mime_type and data fields (optional)
        gemini_temperature: Temperature for generation
        gemini_top_p: Top-p sampling parameter
        gemini_top_k: Top-k sampling parameter
        gemini_candidate_count: Number of candidates to generate
        gemini_safety_threshold: Safety threshold level
        gemini_api_version: API version to use (e.g., 'v1beta', 'v1')
        model: Gemini model to use
    
    Returns:
        API response as a dictionary
    """
    url = f"https://generativelanguage.googleapis.com/{gemini_api_version}/models/{model}:generateContent?key={api_key}"
    
    # Build parts array
    parts: List[Dict[str, Any]] = [{"text": gemini_user_prompt}]
    if gemini_inline_data:
        parts.append({"inline_data": gemini_inline_data})
    
    payload = {
        "contents": [
            {
                "parts": parts
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
    
    if not response.ok:
        print(f"Error {response.status_code}: {response.text}")
        response.raise_for_status()
    
    return response.json()


if __name__ == "__main__":
    # Check if arguments are provided
    if len(sys.argv) < 4:
        print("Usage: python generateResponse.py <max_tokens> <system_instruction_file> <user_prompt_file> [binary_file]")
        sys.exit(1)
    
    # Read max_tokens from first argument
    max_tokens = int(sys.argv[1])
    
    # Read system instruction from second argument
    system_instruction_file = sys.argv[2]
    with open(system_instruction_file, 'r') as f:
        system_instruction_text = f.read()
    
    # Read user prompt from third argument
    user_prompt_file = sys.argv[3]
    with open(user_prompt_file, 'r') as f:
        user_prompt_text = f.read()
    
    # Read binary file if provided as fourth argument
    inline_data = None
    if len(sys.argv) >= 5:
        binary_file = sys.argv[4]
        mime_type, _ = mimetypes.guess_type(binary_file)
        if not mime_type:
            mime_type = "application/octet-stream"
        
        with open(binary_file, 'rb') as f:
            binary_content = f.read()
            encoded_data = base64.b64encode(binary_content).decode('utf-8')
        
        inline_data = {
            "mime_type": mime_type,
            "data": encoded_data
        }
    
    # Read API key from file
    with open('api.key', 'r') as f:
        API_KEY = f.read().strip()
    
    result = call_gemini_api(
        api_key=API_KEY,
        gemini_user_prompt=user_prompt_text,
        gemini_system_instruction={"parts": [{"text": system_instruction_text}]},
        gemini_inline_data=inline_data,
        gemini_max_tokens=max_tokens
    )
    
    # Check for binary outputs in the response and write them to files
    if "candidates" in result:
        for idx, candidate in enumerate(result["candidates"]):
            if "content" in candidate and "parts" in candidate["content"]:
                for part_idx, part in enumerate(candidate["content"]["parts"]):
                    if "inline_data" in part:
                        # Generate unique filename with datetime
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        mime_type = part["inline_data"].get("mime_type", "application/octet-stream")
                        extension = mimetypes.guess_extension(mime_type) or ".bin"
                        filename = f"output_{timestamp}_c{idx}_p{part_idx}{extension}"
                        
                        # Decode and write binary data
                        binary_data = base64.b64decode(part["inline_data"]["data"])
                        with open(filename, 'wb') as f:
                            f.write(binary_data)
                        print(f"Binary output written to: {filename}")
    
    print(json.dumps(result, indent=2))