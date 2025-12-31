import argparse
from utils.model_config import models_to_use
from utils import helper_inference
import os
import pandas as pd
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run psychopathology rating with LLMs')
    parser.add_argument('--outer_runs', type=int, default=1, help='Number of outer runs (default: 1)')
    parser.add_argument('--number_of_definitions', type=int, default=10, help='Number of definitions per run (default: 10, 0 for no definitions)')
    parser.add_argument('--temperature', type=float, default=0.0, help='Temperature for model generation (default: 0.5)')
    parser.add_argument('--prompt_language', type=str, choices=['german', 'english'], default='german',)

    parser.add_argument('--prompt_dir', type=str, default="../data/01_prompts/", 
                        help='Directory containing prompt files')
    parser.add_argument('--transcript_dir', type=str, default="../data/02_transcripts_example/",
                        help='Directory containing transcript files')
    parser.add_argument('--reference_file', type=str, default="../data/00_ratings/reference.csv",
                        help='Path to reference rating CSV file')
    parser.add_argument('--output_dir', type=str, default="../outputs/AI_ratings/",
                        help='Directory for output ratings')
    
    return parser.parse_args()

def start_inference(args: argparse.Namespace, models: list) -> pd.DataFrame:
    #PRERATATION STEPS
    clients = helper_inference.initialize_clients(models)
    schema_instruction, psychopathologies_custom_schema, basic_psychopathologies = helper_inference.get_schema_instruction(args.prompt_language)
    prompt, basic_prompt, definitions = helper_inference.load_prompts(args.prompt_dir, args.prompt_language)
    transkripts = helper_inference.build_transcripts_dict(args.transcript_dir, args.prompt_language)
    instruction_prompts_list = helper_inference.generate_instructions_prompts(definitions,
                                                         args.number_of_definitions,
                                                         prompt,
                                                         basic_prompt)
    reference_df = pd.read_csv(args.reference_file)
    os.makedirs(args.output_dir, exist_ok=True)

    #RUNNING INFERENCE
    all_results_df = helper_inference.run_psychopathology_rating(
    models=models,
    transkripts=transkripts,
    instruction_prompts_list=instruction_prompts_list,
    outer_runs=args.outer_runs,
    temperature=args.temperature,
    number_of_definitions_in_one_run=args.number_of_definitions,
    clients=clients,
    psychopathologies_schema = psychopathologies_custom_schema,
    schema_instruction=schema_instruction,
    Psychopathologies=basic_psychopathologies,
    reference_df=reference_df,
    rating_folder=args.output_dir
)
    return all_results_df
    
def main() -> None:
    args = parse_args()
    print("Parsed arguments:")
    print(args)
    print(f"Final output will be saved to: {args.output_dir}")

    time.sleep(5)
    models = models_to_use
    for _, model_name in models:
        print(f"Starting inference for model: {model_name}")
    all_results_df = start_inference(args, models)
    print("Inference completed. Results:")
    print(all_results_df.head())

if __name__ == "__main__":
    main()