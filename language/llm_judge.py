"""
not for right now 
LLM-based Judge for Pairwise Comparisons

Future module for using an LLM to judge which peptide should be preferred
under a given prompt, instead of using simple rule-based comparisons.
"""

from typing import Callable, Optional
import openai


def create_llm_judge(
    model_name: str = "gpt-4",
    api_key: Optional[str] = None
) -> Callable[[str, str, str], int]:
    """
    Create an LLM judge function for pairwise comparisons.
    
    Args:
        model_name: Name of LLM to use
        api_key: Optional API key (if None, will try to use environment variable)
        
    Returns:
        Judge function that takes (peptide_a, peptide_b, prompt) and returns:
        - 1 if peptide_a is preferred
        - -1 if peptide_b is preferred
        - 0 if equal/no preference
    """
    if api_key:
        openai.api_key = api_key
    
    def judge(peptide_a: str, peptide_b: str, prompt: str) -> int:
        """
        Judge which peptide is preferred under the given prompt.
        
        Args:
            peptide_a: First peptide SMILES string
            peptide_b: Second peptide SMILES string
            prompt: Natural language prompt describing desired properties
            
        Returns:
            1 if a preferred, -1 if b preferred, 0 if equal
        """
        # random prompt for LLM
        llm_prompt = f"""You are evaluating two peptide candidates for therapeutic use.

Design requirements: {prompt}

Peptide A: {peptide_a}
Peptide B: {peptide_b}

Which peptide better satisfies the design requirements? Respond with only:
- "A" if peptide A is preferred
- "B" if peptide B is preferred  
- "EQUAL" if they are equally good

Your response:"""
        
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an expert in peptide therapeutics."},
                    {"role": "user", "content": llm_prompt}
                ],
                temperature=0.0
            )
            
            answer = response.choices[0].message.content.strip().upper()
            
            if "A" in answer and "B" not in answer:
                return 1
            elif "B" in answer and "A" not in answer:
                return -1
            else:
                return 0  # equal or unclear
                
        except Exception as e:
            print(f"Error calling LLM judge: {e}")
            return 0  # default to equal on error
    
    return judge


def create_simple_llm_judge(
    model: Optional[any] = None
) -> Callable[[str, str, str], int]:
    """
    Create a simple LLM judge using a local model.
    
    This is a placeholder for using local LLMs (e.g., via transformers).
    """
    def judge(peptide_a: str, peptide_b: str, prompt: str) -> int:
        """Simple judge - placeholder for local LLM implementation."""
        # in practice, would use local model here
        # for now, return 0 (equal)
        return 0
    
    return judge

