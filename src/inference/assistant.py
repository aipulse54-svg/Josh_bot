"""Inference module for the LLM Assistant."""
import logging
from typing import Dict, List, Optional, Tuple
import re
import sympy as sp
from sympy import symbols, solve, simplify, expand, factor
import torch
from src.model.llm import LLMAssistant

logger = logging.getLogger(__name__)


class CodeMathAssistant:
    """AI Assistant for code generation and math solving."""

    def __init__(
        self,
        model: LLMAssistant,
        task_type: str = "general",
    ):
        self.model = model
        self.task_type = task_type

    def solve_math_problem(self, problem: str) -> Dict[str, any]:
        """Solve mathematical problems using sympy.
        
        Args:
            problem: Mathematical problem statement
            
        Returns:
            Dictionary with solution and steps
        """
        logger.info(f"Solving math problem: {problem}")
        
        try:
            # Try to parse and solve using sympy
            # This is a simplified approach - can be extended
            
            # Extract equation patterns
            equation_pattern = r'([\w+\-*/()\^]+)\s*=\s*([\w+\-*/()\^]+)'
            matches = re.findall(equation_pattern, problem)
            
            solutions = []
            if matches:
                for lhs, rhs in matches:
                    try:
                        x = symbols('x')
                        equation = sp.sympify(f"({lhs}) - ({rhs})")
                        sol = solve(equation, x)
                        solutions.extend(sol)
                    except Exception as e:
                        logger.warning(f"Could not solve equation: {e}")
            
            return {
                "problem": problem,
                "solutions": solutions,
                "method": "symbolic_algebra",
                "success": len(solutions) > 0,
            }
        except Exception as e:
            logger.error(f"Error solving math problem: {e}")
            return {
                "problem": problem,
                "solutions": [],
                "error": str(e),
                "success": False,
            }

    def generate_code(
        self,
        prompt: str,
        language: str = "python",
        max_length: int = 512,
    ) -> Dict[str, any]:
        """Generate code based on prompt.
        
        Args:
            prompt: Code generation prompt
            language: Programming language
            max_length: Maximum length of generated code
            
        Returns:
            Dictionary with generated code
        """
        logger.info(f"Generating {language} code for: {prompt}")
        
        # Enhance prompt for code generation
        enhanced_prompt = f"""Write {language} code for:
{prompt}

Code:
```{language}"""
        
        try:
            generated = self.model.generate(
                prompt=enhanced_prompt,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
            )
            
            # Extract code block
            code = generated[0] if generated else ""
            
            return {
                "prompt": prompt,
                "language": language,
                "code": code,
                "success": len(code) > 0,
            }
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            return {
                "prompt": prompt,
                "language": language,
                "code": "",
                "error": str(e),
                "success": False,
            }

    def answer_question(
        self,
        question: str,
        context: Optional[str] = None,
    ) -> Dict[str, any]:
        """Answer general questions.
        
        Args:
            question: Question to answer
            context: Optional context information
            
        Returns:
            Dictionary with answer
        """
        logger.info(f"Answering question: {question}")
        
        prompt = question
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        
        try:
            answer = self.model.generate(
                prompt=prompt,
                max_length=256,
                temperature=0.5,
                top_p=0.9,
            )
            
            return {
                "question": question,
                "answer": answer[0] if answer else "",
                "context": context,
                "success": len(answer) > 0,
            }
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return {
                "question": question,
                "answer": "",
                "error": str(e),
                "success": False,
            }

    def explain_code(
        self,
        code: str,
    ) -> Dict[str, any]:
        """Explain what a piece of code does.
        
        Args:
            code: Code to explain
            
        Returns:
            Dictionary with explanation
        """
        logger.info("Explaining code...")
        
        prompt = f"""Explain this code:

{code}

Explanation:"""
        
        try:
            explanation = self.model.generate(
                prompt=prompt,
                max_length=256,
                temperature=0.5,
            )
            
            return {
                "code": code,
                "explanation": explanation[0] if explanation else "",
                "success": len(explanation) > 0,
            }
        except Exception as e:
            logger.error(f"Error explaining code: {e}")
            return {
                "code": code,
                "explanation": "",
                "error": str(e),
                "success": False,
            }
