import requests

class OllamaAdapter:
    def __init__(self, model="qwen2.5-coder:32b", host="http://127.0.0.1:11434"):
        """
        Initializes the local LLM routing. 
        Target models verified in manifest: 'qwen2.5-coder:32b', 'qwen3:32b', 'deepseek-r1:14b'
        """
        self.model = model
        self.host = host
        self.api_url = f"{self.host}/api/chat"

    def generate_response(self, messages, temperature=0.7):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except requests.exceptions.RequestException as e:
            print(f"Effector System Error - Local LLM Routing Failed: {e}")
            return None
    
    def switch_model(self, new_model):
        """Allows runtime switching between execution (Coder) and deliberation (DASP) models."""
        self.model = new_model