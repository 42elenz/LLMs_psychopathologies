# LLM Performance in Psychopathological Assessment of Simulated Clinical Interviews

This repository contains the code and data for our paper evaluating large language models on psychopathology rating tasks and comparing their performance with practicing clinicians.

## Experimental Design
Simulated patient interviews assessed via three pathways: practicing clinicians (n=108), expert consensus panel (n=3 AMDP trainers), and LLMs (n=10 models). LLMs evaluated transcripts under four conditions: temperature 0.0 or 0.5 (with majority voting for T>0) × with/without AMDP definitions. All ratings compared against expert ground truth.

![Experimental Design](data/03_ressources/Experimental_design.png)

## Results Visualization mentioned in publication 
Interactive visualizations comparing model performance with human ratings:

- **Mania**: https://42elenz.github.io/LLMs_psychopathologies/selection__error_rate_scatter_video_7.html
- **Depression**: https://42elenz.github.io/LLMs_psychopathologies/selection__error_rate_scatter_video_8.html
- **Schizophrenia**: https://42elenz.github.io/LLMs_psychopathologies/selection__error_rate_scatter_video_9.html

These interactive plots show:
- Model vs. reference vs.  Clinician rating errors
- Reasoning of the model (Hover to investigate!) 

## Repository Structure

- `src/00_call_models.py`: Main script for LLM inference and rating generation
- `src/01_comparision_and_statistics.ipynb`: Analysis notebook containing all experiments and comparisons from the paper
- `data/`: Contains prompts, transcripts, and reference ratings
- `outputs/`: Generated model ratings and statistical analyses
- `docs/`: Interactive visualizations
- `src/utils/model_config.py`: Model configuration and selection

## Setup Instructions

### 1. Environment Setup
Install the necessary packages via pip. The necassary requirement.txt can be found in the src/ directory. Install via
`pip install -r requirement.txt`

### 2. API Keys Configuration
1. Navigate to `api_keys/` directory
2. Rename `.your_env` to `.env`
3. Add your API keys to the `.env` file:
   ```
   OPENAI_API_KEY=your_openai_key
   ANTHROPIC_API_KEY=your_anthropic_key
   GEMINI_API_KEY=your_gemini_key
   MISTRAL_API_KEY=your_mistral_key
   TOGETHER_API_KEY=your_together_key
   ```
   **Important:** Do not change the `.env` file path (don't move the .env somewhere else) as it is hard-coded.

### 3. Model Selection
Models to be evaluated are configured in `src/utils/model_config.py`. Currently set as an example to:
```python
models_to_use = [
    ("gemini", "gemini-2.5-flash"),
]
```

Available models and their required API keys:
- **OpenAI models** (`OPENAI_API_KEY`):
  - `("openai", "gpt-4-turbo")`
  - `("openai", "gpt-5.1")`
  
- **Anthropic models** (`ANTHROPIC_API_KEY`):
  - `("anthropic", "claude-sonnet-4-5")`
  
- **Gemini models** (`GEMINI_API_KEY`):
  - `("gemini", "gemini-2.5-flash")`
  - `("gemini", "gemini-3-pro-preview")`
  
- **Mistral models** (`MISTRAL_API_KEY`):
  - `("mistral", "mistral-large-latest")`
  
- **Open-source models** (`TOGETHER_API_KEY`):
  - `("open_source", "Qwen/Qwen3-Next-80B-A3B-Thinking")`
  - `("open_source", "deepseek-ai/DeepSeek-R1")`
  - `("open_source", "moonshotai/Kimi-K2-Thinking")`

## Running the Experiments

### Step 1: Generate Model Ratings
The current setup runs inference on the Mania transcript as an example:

```bash
cd src
python 00_call_models.py \
  --outer_runs 1 \
  --number_of_definitions 10 \
  --temperature 0.0 \
  --prompt_language german \
  --transcript_dir ../data/02_transcripts_example/
```

Parameters:
- `--outer_runs`: Number of repetitions (default: 1)
- `--number_of_definitions`: Number definitions added to the prompt per run (default: 10, use 0 to disable)
- `--temperature`: Model generation temperature (default: 0.0)
- `--prompt_language`: Choose between `german` or `english`. This loads the English or German prompts and input data.
- `--transcript_dir`: Default uses example transcripts; change to `../data/02_transcripts/` for full dataset or your own transcripts

Outputs:
- Timestamped CSV files in `outputs/AI_ratings/`
- Raw model responses in `src/raw_responses/`
- Parsing errors logged separately

### Step 2: Analyze Results
IMPORTANT: If you just want to check the current analyses you can just run the src/01_comparision_and_statistics.ipynb as it is. The data loaded is the data used for the publication.
1. Open `src/01_comparision_and_statistics.ipynb`
2. Update the path to your newly generated CSV from `outputs/AI_ratings/`
3. Run all cells to reproduce the paper's analyses

The notebook contains:
- All statistical comparisons from the paper
- Performance metrics calculations
- Comparison tables exported to `outputs/tables/`



## Data Description

- **Prompts** (`data/01_prompts/`): GRASCEF scale definitions and rating instructions in German/English
- **Transcripts** (`data/02_transcripts/`): Patient interview transcripts for Mania, Depression, and Schizophrenia cases
- **Reference Ratings** (`data/00_ratings/reference.csv`): Expert consensus ratings used as ground truth
- **Human Ratings** (`data/00_ratings/human_master.csv`): Individual clinician ratings for comparison
- **R-Analyses** The R-Models and their code can be found in src/R_analyses

## Notes for Reviewers

- The provided example runs inference on a single transcript (Mania) with one model for demonstration
- To reproduce full paper results, uncomment additional models in `model_config.py` and ensure corresponding API keys are set
- The analysis notebook (`01_comparision_and_statistics.ipynb`) contains all statistical tests and visualizations from the paper
- Raw outputs are preserved for full reproducibility and error analysis
