"""Inference script for LLM Assistant."""
import logging
import argparse
from src.model.llm import LLMAssistant
from src.inference.assistant import CodeMathAssistant

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(args):
    """Run inference on prompts."""
    logger.info(f"Loading model from {args.model_path}")
    
    # Load model
    llm = LLMAssistant(model_name=args.model_path)
    assistant = CodeMathAssistant(llm)
    
    if args.task == "code":
        logger.info(f"Generating code for: {args.prompt}")
        result = assistant.generate_code(args.prompt, language=args.language)
        print("\n=== Generated Code ===")
        print(result["code"])
        
    elif args.task == "math":
        logger.info(f"Solving math problem: {args.prompt}")
        result = assistant.solve_math_problem(args.prompt)
        print("\n=== Solution ===")
        print(f"Solutions: {result['solutions']}")
        
    elif args.task == "explain":
        logger.info("Explaining code...")
        result = assistant.explain_code(args.prompt)
        print("\n=== Code Explanation ===")
        print(result["explanation"])
        
    else:
        result = assistant.answer_question(args.prompt)
        print("\n=== Answer ===")
        print(result["answer"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Assistant inference")
    parser.add_argument(
        "--model_path",
        type=str,
        default="gpt2",
        help="Path to model or model name",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="general",
        choices=["code", "math", "explain", "general"],
        help="Task type",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="python",
        help="Programming language for code generation",
    )
    
    args = parser.parse_args()
    main(args)
