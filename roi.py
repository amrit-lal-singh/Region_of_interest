import json
import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List, Optional

# Gemini API configuration
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"  # Replace with your actual key when running locally
GEMINI_MODEL = "gemini-2.5-pro-preview-03-25"

# Configure the Gemini API
genai.configure(api_key=GEMINI_API_KEY)

# Define the structured output model for ROI identification
class RegionOfInterest(BaseModel):
    roi: str = Field(description="A very short phrase (1-2 words) from the generated answer that best answers the question")
    explanation: Optional[str] = Field(description="Brief explanation of why this region is the answer")

def process_question_answer(question: str, answer: str) -> dict:
    """
    Use Gemini to identify the region of interest in the generated answer.
    
    Args:
        question: The original question
        answer: The generated answer from the model
    
    Returns:
        Dictionary containing the identified ROI and explanation
    """
    prompt = f"""
    Given the following question and answer, identify the specific 1-2 word phrase in the answer that most directly addresses the question.
    
    Question: {question}
    Generated Answer: {answer}
    
    Your task is to extract only the specific 1-2 word phrase from the answer that constitutes the core response to the question.
    For example, if the question is "What is the capital of France?" and the answer is "The capital of France is Paris, a city known for...", 
    the region of interest would be "Paris".
    
    Output ONLY a JSON in this exact format:
    {{
      "roi": "1-2 word phrase that answers the question",
      "explanation": "Brief explanation of why this is the answer"
    }}
    """
    
    # Call Gemini API
    model = genai.GenerativeModel(model_name=GEMINI_MODEL)
    response = model.generate_content(prompt)
    
    # Parse the response text as JSON
    try:
        import re
        # Extract JSON part if there's any markdown formatting
        response_text = response.text
        json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_str = response_text
            
        # Clean up the JSON string if needed
        json_str = json_str.strip()
        
        # Parse JSON
        result = json.loads(json_str)
        
        # Ensure we have the expected keys
        if "roi" not in result:
            result["roi"] = "unknown"
        if "explanation" not in result:
            result["explanation"] = None
            
        return result
    except Exception as e:
        print(f"  Error parsing response: {str(e)}")
        print(f"  Response text: {response.text}")
        return {"roi": "parsing_error", "explanation": str(e)}

def main():
    # Load the output_set.json file
    print("Loading output_set.json...")
    try:
        with open("output_set.json", "r") as f:
            output_data = json.load(f)
    except FileNotFoundError:
        print("Error: output_set.json not found. Please run the LLM inference first.")
        return
    
    # Process difficult questions
    print("Processing difficult questions...")
    for i, question_data in enumerate(output_data.get("difficult_questions", [])):
        question = question_data.get("question", "")
        answer = question_data.get("generated_answer", "")
        
        if question and answer:
            print(f"Processing question {i+1}: {question[:50]}...")
            try:
                roi_result = process_question_answer(question, answer)
                
                # Add ROI to the question data
                output_data["difficult_questions"][i]["region_of_interest"] = roi_result["roi"]
                output_data["difficult_questions"][i]["roi_explanation"] = roi_result["explanation"]
                
                print(f"  ROI identified: '{roi_result['roi']}'")
            except Exception as e:
                print(f"  Error processing question: {str(e)}")
    
    # Process easy questions
    print("Processing easy questions...")
    for i, question_data in enumerate(output_data.get("easy_questions", [])):
        question = question_data.get("question", "")
        answer = question_data.get("generated_answer", "")
        
        if question and answer:
            print(f"Processing question {i+1}: {question[:50]}...")
            try:
                roi_result = process_question_answer(question, answer)
                
                # Add ROI to the question data
                output_data["easy_questions"][i]["region_of_interest"] = roi_result["roi"]
                output_data["easy_questions"][i]["roi_explanation"] = roi_result["explanation"]
                
                print(f"  ROI identified: '{roi_result['roi']}'")
            except Exception as e:
                print(f"  Error processing question: {str(e)}")
    
    # Save the updated output_set.json
    print("Saving updated output_set.json...")
    with open("output_set.json", "w") as f:
        json.dump(output_data, f, indent=2)
    
    print("Processing complete.")

if __name__ == "__main__":
    main()
