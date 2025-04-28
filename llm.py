import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple

# Hugging Face token for authentication
HF_TOKEN = "***REMOVED***"

class ModelInference:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-3B"):
        """Initialize the model and tokenizer."""
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
            token=HF_TOKEN
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            token=HF_TOKEN
        )
        self.device = self.model.device
        print(f"Model loaded on device: {self.device}")

    def _process_logits(self, logits: torch.Tensor, topk: int = 3) -> List[Dict[str, Any]]:
        """
        Process logits to get top-k tokens and their probabilities.
        
        Args:
            logits: Logits from the model (batch_size, vocab_size)
            topk: Number of top tokens to return
            
        Returns:
            List of dicts with token, token_str, and probability
        """
        probabilities = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probabilities, k=topk, dim=-1)
        
        result = []
        for i in range(topk):
            token_id = topk_indices[0, i].item()
            token_str = self.tokenizer.decode([token_id])
            prob = topk_probs[0, i].item()
            result.append({
                "token_id": token_id,
                "token_str": token_str,
                "probability": prob
            })
        
        return result

    def generate_with_token_probs(self, prompt: str, max_length: int = 100) -> Dict[str, Any]:
        """
        Generate text and track token probabilities.
        
        Args:
            prompt: Input prompt text
            max_length: Maximum length of generated sequence
            
        Returns:
            Dict with generated text and token probability information
        """
        # Tokenize the prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Storage for tokens and their probabilities
        generated_tokens = []
        token_data = []
        
        # Starting position after input prompt
        cur_len = input_ids.shape[1]
        
        # Generation loop
        with torch.no_grad():
            while cur_len < max_length:
                # Forward pass
                outputs = self.model(input_ids)
                next_token_logits = outputs.logits[:, -1, :]
                
                # Get top-k token predictions and probabilities
                topk_tokens = self._process_logits(next_token_logits)
                
                # Get the next token (top-1)
                next_token = topk_tokens[0]["token_id"]
                
                # Store token data
                token_data.append({
                    "selected_token": {
                        "token_id": next_token,
                        "token_str": topk_tokens[0]["token_str"],
                        "probability": topk_tokens[0]["probability"]
                    },
                    "top_alternatives": topk_tokens[1:]
                })
                
                # Add token to generated sequence
                input_ids = torch.cat([input_ids, torch.tensor([[next_token]]).to(self.device)], dim=1)
                generated_tokens.append(next_token)
                cur_len += 1
                
                # Check for EOS token to stop generation
                if next_token == self.tokenizer.eos_token_id:
                    break
        
        # Decode the full generated text
        full_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {
            "generated_text": full_text,
            "token_data": token_data
        }

    def process_questions(self, questions: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Process a list of questions.
        
        Args:
            questions: List of question dictionaries
            
        Returns:
            List of results with question, answer, and token data
        """
        results = []
        
        for q in questions:
            question = q["question"]
            print(f"Processing question: {question}")
            
            # Create a prompt by simply using the question
            prompt = f"Question: {question}\nAnswer:"
            
            # Generate response with token probabilities
            response = self.generate_with_token_probs(prompt)
            
            # Store results
            results.append({
                "question": question,
                "generated_answer": response["generated_text"],
                "token_data": response["token_data"]
            })
        
        return results


def main():
    # Load the test questions
    with open("test_set.json", "r") as f:
        test_data = json.load(f)
    
    # Initialize model
    inference = ModelInference("Qwen/Qwen2.5-3B")
    
    # Process both difficult and easy questions
    output = {
        "difficult_questions": inference.process_questions(test_data["difficult_questions"]),
        "easy_questions": inference.process_questions(test_data["easy_questions"])
    }
    
    # Save results
    with open("output_set.json", "w") as f:
        json.dump(output, f, indent=2)
    
    print("Processing complete. Results saved to output_set.json")


if __name__ == "__main__":
    main()
