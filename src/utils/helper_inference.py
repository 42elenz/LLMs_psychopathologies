import os
import json
from dotenv import load_dotenv
import pandas as pd
from pydantic import BaseModel
from typing import List
from collections import defaultdict, Counter
import re
from typing import Dict, Any
import time
from google.genai.types import GenerateContentConfig
from together import Together
from openai import OpenAI
from google import genai
import openai
import anthropic
from anthropic import transform_schema
from mistralai import Mistral
from dotenv import load_dotenv
import datetime

load_dotenv(dotenv_path="../api_keys/.env")

def get_model_max_tokens(client_together, model_name: str) -> int:
    """
    Get the maximum output tokens for a Together AI model.
    Falls back to reasonable defaults if not available.
    """
    max_tokens = None
    try:
        # Query model info from Together API
        models = client_together.models.list()
        for model in models:
            if model.id == model_name:
                # Most models have max_tokens or context_length attributes
                if hasattr(model, 'max_tokens'):
                    max_tokens = model.max_tokens
                elif hasattr(model, 'context_length'):
                    # Use 75% of context length as safe max output
                    max_tokens = int(model.context_length * 0.75)
                print(f"Run model with max tokens of {int(model.context_length * 0.75)}")
    except Exception as e:
        print(f"Could not retrieve max tokens for {model_name}: {e}")
    if max_tokens is None:
        # Fallback defaults based on model patterns
        if "deepseek" in model_name.lower():
            max_tokens = 8192
        elif "qwen" in model_name.lower():
            max_tokens = 8192
        elif "gpt-oss" in model_name.lower():
            max_tokens = 4096
        else:
            max_tokens = 4096  # Conservative default
    print(f"run model {model_name} with max tokens of {max_tokens}")
    return max_tokens

def a_schema():
    class Psychopathologie(BaseModel):
        Nummer: int
        Merkmal: str
        Schweregrad: int
        Begründung: str
    class Psychopathologies(BaseModel):
        psychopathologies: List[Psychopathologie]
    return Psychopathologie, Psychopathologies


def eng_schema():
    class Psychopathologie(BaseModel):
        Number: int
        Trait: str
        Severness: int
        Reasoning: str
    class Psychopathologies(BaseModel):
        psychopathologies: List[Psychopathologie]
    return Psychopathologie, Psychopathologies

def extract_psy_cols(df):
    return [c for c in df.columns if re.match(r'^p\d+_.*_final$', c)]

def parse_definitions(definitions_text: str):
    """
    Split the big definitions string into structured items.
    Separator assumed to be lines starting with #### (or ###).
    Returns list of dicts: {number, name, full_text}
    """
    raw_chunks = re.split(r'\n?#+\n?', definitions_text)
    items = []
    for chunk in raw_chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        # First line like: 1. Bewusstseinsverminderung (Fremdeinschätzung)
        first_line = chunk.splitlines()[0].strip()
        m = re.match(r'^(\d+)\.\s*(.+)$', first_line)
        if not m:
            continue
        num = int(m.group(1))
        name = m.group(2).strip()
        items.append({
            "number": num,
            "name": name,
            "full_text": chunk
        })
    # Sort by number to ensure order
    items.sort(key=lambda x: x["number"])
    return items

def initialize_clients(models: list) -> dict:
    """
    Initialize only the API clients needed for the given models.
    
    Args:
        models: List of tuples (api_family, model_name)
    
    Returns:
        Dictionary of initialized clients
    """
    clients = {}
    api_families = {api_family for api_family, _ in models}
    
    if "openai" in api_families:
        clients["openai"] = OpenAI()
    
    if "gemini" in api_families:
        clients["gemini"] = genai.Client(http_options={'api_version': 'v1beta'})
    
    if "anthropic" in api_families:
        clients["anthropic"] = anthropic.Anthropic(timeout=900.0)
    
    if "mistral" in api_families:
        clients["mistral"] = Mistral(api_key=os.environ["MISTRAL_API_KEY"])
    
    if "open_source" in api_families:
        clients["together"] = Together()
    
    return clients


def generate_instructions_prompts(definitions_text: str,
                                  number_of_definitions_in_one_run: int,
                                  base_prompt: str,
                                  basic_prompt: str):
    """
    Create a list of prompt variants each containing a subset of definitions.
    Replaces placeholders:
      <ANZAHL_MERKMALE>
      <MERKMAL_TUPLE>
      <DEFINITIONEN>
    Returns list of dicts: {prompt, numbers, names}
    """
    if number_of_definitions_in_one_run <= 0:
        print("returning one chunk with basic prompt and no definitions")
        return [{
            "prompt": basic_prompt,
            "numbers": list(range(1, 101)),
            "names": []
        }]
    items = parse_definitions(definitions_text)
    chunks = []
    for i in range(0, len(items), number_of_definitions_in_one_run):
        subset = items[i:i+number_of_definitions_in_one_run]
        anzahl = len(subset)
        merkmals_tuple = "\n".join([f"{it['number']};{it['name']}" for it in subset])
        definitions_block = "\n\n".join([it["full_text"] for it in subset])

        prompt_variant = (base_prompt
                          .replace("<ANZAHL_MERKMALE>", str(anzahl))
                          .replace("<MERKMAL_TUPLE>", merkmals_tuple)
                          .replace("<DEFINITIONEN>", definitions_block))

        chunks.append({
            "prompt": prompt_variant,
            "numbers": [it["number"] for it in subset],
            "names": [it["name"] for it in subset]
        })
    return chunks

def merge_partial_rows(partial_rows: list):
    """
    Union columns from all chunk rows and take first non-null per column.
    """
    if not partial_rows:
        return None

    # 1) Union all columns
    all_cols = set()
    for df in partial_rows:
        all_cols.update(df.columns)

    merged = partial_rows[0].copy()
    for col in all_cols:
        if col not in merged.columns:
            merged[col] = None

    # 2) Fill with first non-null across chunks
    for df in partial_rows[1:]:
        for col in df.columns:
            base_val = merged.iloc[0].get(col, None)
            new_val = df.iloc[0][col]
            if (base_val is None or (isinstance(base_val, float) and pd.isna(base_val)) or pd.isna(base_val)) and not pd.isna(new_val):
                merged.at[merged.index[0], col] = new_val

    return merged

def finalize_merged_row(merged_df: pd.DataFrame, reference_df: pd.DataFrame):
    """
    Make column order deterministic and drop chunk-only columns.
    """
    meta_cols = ['video_id','video_type','site','api','model_name','id','rater_type','run','txt_name']
    psy_cols = extract_psy_cols(reference_df)
    # keep only known meta + psy columns we actually have
    keep = [c for c in meta_cols if c in merged_df.columns] + psy_cols
    # add any other columns (e.g., begründungen) that match the pattern and exist
    others = [c for c in merged_df.columns if c not in keep and c.startswith('p') and c.endswith('_final')]
    keep += others
    out = merged_df.reindex(columns=keep)
    # drop chunk_index if present
    if 'chunk_index' in out.columns:
        out = out.drop(columns=['chunk_index'])
    return out


# In[7]:


def find_number_key(data_item):
    """Find the key that contains the number (Nummer/Number/nummer)"""
    possible_keys = ['Nummer', 'nummer', 'Number', 'number']
    for key in possible_keys:
        if key in data_item:
            return key
    # Fuzzy match if exact not found
    for key in data_item.keys():
        if 'numm' in key.lower() or 'numb' in key.lower():
            return key
    return None

def find_severity_key(data_item):
    """Find the severity key (Schweregrad/Severity)"""
    possible_keys = ['Schweregrad', 'schweregrad', 'Severity', 'severity', 'Severness', 'severness']
    for key in possible_keys:
        if key in data_item:
            return key
    # Fuzzy match
    for key in data_item.keys():
        if 'schwere' in key.lower() or 'severity' in key.lower():
            return key
    return None

def find_begründung_key(data_item):
    """Find the Begründung key"""
    possible_keys = ['Begründung', 'begründung', 'Begrundung', 'begrundung', 'Justification', 'justification', 'Reason', 'reason', 'Reasoning', 'reasoning']
    for key in possible_keys:
        if key in data_item:
            return key
    # Fuzzy match
    for key in data_item.keys():
        if 'begründ' in key.lower() or 'begrund' in key.lower() or 'justif' in key.lower():
            return key
    return None

def create_psy_number_mapping(reference_df):
    """Create mapping from psychopathology number (1-100) to column name"""
    psy_cols = extract_psy_cols(reference_df)
    mapping = {}
    for col in psy_cols:
        # Extract number from column name like 'p1_bewusstseinsverminderung_final'
        match = re.match(r'p(\d+)_', col)
        if match:
            num = int(match.group(1))
            mapping[num] = col
    return mapping

def _extract_list_from_response_json(parsed):
    """
    Extract the psychopathologies list from JSON response.
    Handles both:
    - Direct list: [{"Nummer": 1, ...}, ...]
    - Wrapped dict: {"psychopathologies": [{"Nummer": 1, ...}, ...]}
    """
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        # Try common wrapper keys
        for key in ("psychopathologies", "Psychopathologies", "items", "data"):
            if key in parsed and isinstance(parsed[key], list):
                return parsed[key]
    return []

def json_to_dataframe(data_list, video_id, video_type, site, api, model_name, rater_id, 
                      reference_df,
                      allowed_numbers=None,
                      only_allowed_columns=False):
    data_list = _extract_list_from_response_json(data_list)
    psy_mapping = create_psy_number_mapping(reference_df)

    row_data = {
        'video_id': video_id,
        'video_type': video_type,
        'site': site,
        'api': api,
        'model_name': model_name,
        'id': rater_id,
        'rater_type': 'AI'
    }

    psy_cols_all = extract_psy_cols(reference_df)

    if only_allowed_columns and allowed_numbers:
        # initialize only columns for current chunk (ratings + their begründung if present in ref)
        cols_to_init = []
        for n in sorted(allowed_numbers):
            rating_col = psy_mapping.get(n)
            if rating_col:
                cols_to_init.append(rating_col)
            begr_col = f"p{n}_begründung_final"
            if begr_col in psy_cols_all:
                cols_to_init.append(begr_col)
        for col in cols_to_init:
            row_data[col] = None
    else:
        # initialize all columns (current behavior)
        for col in psy_cols_all:
            row_data[col] = None

    if data_list:
        number_key = find_number_key(data_list[0])
        severity_key = find_severity_key(data_list[0])
        begruendung_key = find_begründung_key(data_list[0])
        if number_key and severity_key:
            for item in data_list:
                num = item.get(number_key)
                if allowed_numbers and num not in allowed_numbers:
                    continue
                severity = item.get(severity_key)
                if isinstance(num, int) and 1 <= num <= 100:
                    col_name = psy_mapping.get(num)
                    if col_name:
                        row_data[col_name] = severity
                        row_data[f'p{num}_begründung_final'] = item.get(begruendung_key, '')

    return pd.DataFrame([row_data])


def get_schema_instruction(prompt_language: str) -> str:
    if prompt_language == 'english':
        Psychopathologie, Psychopathologies = eng_schema()
        # Create JSON schema as text for the prompt
        json_schema = Psychopathologies.model_json_schema()
        schema_instruction = f"""

                            You must respond with ONLY valid JSON matching this exact schema:

                            ```json
                            {json.dumps(json_schema, indent=2)}
                            ```

                            Example output format:
                            ```json
                            {{
                            "psychopathologies": [
                                {{
                                "Number": 1,
                                "Trait": "Impaired consciousness (quantitative)",
                                "Severness": 2,
                                "Reasoning": "Patient shows small problems..."
                                }},
                                {{
                                "Number": 5,
                                "Trait": "Disorientation in time ",
                                "Severness": 3,
                                "Reasoning": "Sever problems reagarding orientation of time ..."
                                }}
                            ]
                            }}
                            ```

                            Respond ONLY with the JSON, no additional text or markdown formatting.
                            """
    else:
        Psychopathologie, Psychopathologies = a_schema()
        json_schema = Psychopathologies.model_json_schema()
        schema_instruction = f"""

                            You must respond with ONLY valid JSON matching this exact schema:

                            ```json
                            {json.dumps(json_schema, indent=2)}
                            ```

                            Example output format:
                            ```json
                            {{
                            "psychopathologies": [
                                {{
                                "Nummer": 1,
                                "Merkmal": "Bewusstseinsverminderung",
                                "Schweregrad": 2,
                                "Begründung": "Patient zeigt leichte Beeinträchtigung..."
                                }},
                                {{
                                "Nummer": 5,
                                "Merkmal": "Aufmerksamkeitsstörung",
                                "Schweregrad": 3,
                                "Begründung": "Deutliche Konzentrationsprobleme erkennbar..."
                                }}
                            ]
                            }}
                            ```

                            Respond ONLY with the JSON, no additional text or markdown formatting.
                            """
    psychopathologie_item_schema = Psychopathologie.model_json_schema()
    if '$schema' in psychopathologie_item_schema:
        del psychopathologie_item_schema['$schema']
    psychopathologies_custom_schema = {
            "type": "object",
            "properties": {
                "psychopathologies": {
                    "type": "array",
                    "description": "A list of identified psychopathologies.",
                    "items": psychopathologie_item_schema  
                }
            },
            "required": ["psychopathologies"]
        }

    return schema_instruction, psychopathologies_custom_schema, Psychopathologies

def load_prompts(prompt_dir: str, prompt_language: str):
    """
    Load prompt files based on language.
    Expects files:
      prompt_{language}.txt
      basic_prompt_{language}.txt
      definitions_{language}.txt
    """
    prompt_file = os.path.join(prompt_dir, f"GRASCEF_definition_{prompt_language}.txt")
    basic_prompt_file = os.path.join(prompt_dir, f"GRASCEF_{prompt_language}.txt")
    definitions_file = os.path.join(prompt_dir, f"All_Items_{prompt_language}.txt")

    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompt = f.read().strip()

    with open(basic_prompt_file, 'r', encoding='utf-8') as f:
        basic_prompt = f.read().strip()

    with open(definitions_file, 'r', encoding='utf-8') as f:
        definitions = f.read().strip()

    return prompt, basic_prompt, definitions

def build_transcripts_dict(folder_path: str, prompt_language: str) -> dict:
    """
    Automatically build transcripts dictionary from files in folder.
    Assumes files are named like: NN_Name_{prompt_language}.txt
    Maps video number to video_type based on the name.
    
    Args:
        folder_path: Path to transcripts folder
        prompt_language: Language suffix (e.g., 'englisch', 'deutsch')
    
    Returns:
        Dictionary with transcript metadata
    """
    transcripts = {}
    
    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith(f"_{prompt_language}.txt"):
            continue
        
        # Extract number and name from filename (e.g., "07_Manie_englisch.txt")
        match = re.match(r"(\d+)_(.+)_" + re.escape(prompt_language) + r"\.txt", filename)
        if not match:
            continue
        
        video_id = int(match.group(1))
        video_type = match.group(2)
        
        # Build key (e.g., "07_Manie")
        key = f"{video_id:02d}_{video_type}"
        
        transcripts[key] = {
            "path": os.path.join(folder_path, filename),
            "video_id": video_id,
            "video_type": video_type,
        }
    
    return transcripts


def get_api_provider(api_family: str) -> str:
    """Map API family to provider name."""
    provider_map = {
        "gemini": "google",
        "openai": "openai",
        "open_source": "together",
        "anthropic": "anthropic",
        "mistral": "mistral",
    }
    return provider_map.get(api_family, api_family)


def call_api(api_family: str, model_name: str, chunk_prompt: str, transcript: str, 
             temperature: float, run: int, txt_name: str, chunk_idx: int,
             clients: dict, schema_instruction: str, Psychopathologies, psychopathologies_schema: dict) -> str:
    """
    Call the appropriate API based on api_family.
    Returns the rating text response.
    Raises exception if API call fails.
    """
    if api_family == "openai":
        print(f"Run {run} {txt_name} {model_name} chunk {chunk_idx}")
        reasoning_arg = {}
        if model_name in ["gpt-5.1", "gpt-5.1-preview", "gpt-4o"]:
            reasoning_arg = {"reasoning": {"effort": "medium"}}
        
        response = clients["openai"].responses.parse(
            model=model_name,
            instructions=chunk_prompt,
            input=transcript,
            text_format=Psychopathologies,
            **reasoning_arg,
        )
        rating_text = response.output_text
        if rating_text is None:
            raise ValueError(f"OpenAI returned None response")
    
    elif api_family == "gemini":
        print(f"Run {run} {txt_name} {model_name} chunk {chunk_idx}")
        response = clients["gemini"].models.generate_content(
            model=model_name,
            contents=transcript,
            config=GenerateContentConfig(
                system_instruction=chunk_prompt,
                temperature=temperature,
                candidate_count=1,
                response_mime_type="application/json",
                response_schema=psychopathologies_schema,
            ),
        )
        rating_text = response.text
        if rating_text is None:
            raise ValueError(f"Gemini returned None response")
    
    elif api_family == "anthropic":
        print(f"Calling Anthropic model {model_name} for run {run} on {txt_name} chunk {chunk_idx}")
        response = clients["anthropic"].beta.messages.create(
            model=model_name,
            max_tokens=64000,
            betas=["structured-outputs-2025-11-13"],
            temperature=temperature,
            system=chunk_prompt,
            messages=[{"role": "user", "content": transcript}],
            output_format={
                "type": "json_schema",
                "schema": transform_schema(Psychopathologies),
            }
        )
        rating_text = response.content[0].text
        if rating_text is None:
            raise ValueError(f"Anthropic returned None response")
    
    elif api_family == "mistral":
        print(f"Calling Mistral model {model_name} for run {run} on {txt_name} chunk {chunk_idx}")
        response = clients["mistral"].chat.complete(
            model=model_name,
            temperature=temperature,
            messages=[
                {"role": "system", "content": chunk_prompt + schema_instruction},
                {"role": "user", "content": transcript}
            ],
            response_format={"type": "json_object"},
        )
        rating_text = response.choices[0].message.content
        if rating_text is None:
            raise ValueError(f"Mistral returned None response")
        
        # Clean up markdown formatting
        if "```json" in rating_text:
            rating_text = rating_text.split("```json")[1].split("```")[0].strip()
        elif "```" in rating_text:
            rating_text = rating_text.split("```")[1].split("```")[0].strip()
    
    elif api_family == "open_source":
        print(f"Calling Together model {model_name} for run {run} on {txt_name} chunk {chunk_idx}")
        json_schema = Psychopathologies.model_json_schema()
        response = clients["together"].chat.completions.create(
            model=model_name,
            max_tokens=get_model_max_tokens(clients["together"], model_name),
            messages=[
                {"role": "system", "content": chunk_prompt + schema_instruction},
                {"role": "user", "content": transcript}
            ],
            temperature=temperature,
            response_format={"type": "json_schema", "schema": json_schema}
        )
        rating_text = response.choices[0].message.content
        if rating_text is None:
            raise ValueError(f"Together returned None response")
    
    else:
        raise ValueError(f"Unknown API family: {api_family}")
    
    return rating_text


def process_chunk_with_retry(api_family: str, model_name: str, chunk_prompt: str, transcript: str,
                             temperature: float, run: int, txt_name: str, chunk_idx: int,
                             clients: dict, schema_instruction: str, Psychopathologies, psychopathologies_schema: dict,
                             max_retries: int = 5, retry_delay: int = 20) -> str:
    """
    Call API with exponential backoff retry logic.
    Returns rating_text on success.
    """
    for attempt in range(max_retries):
        try:
            rating_text = call_api(api_family, model_name, chunk_prompt, transcript,
                                 temperature, run, txt_name, chunk_idx,
                                 clients, schema_instruction, Psychopathologies, psychopathologies_schema)
            return rating_text
        
        except Exception as e:
            error_msg = str(e)
            is_retryable = any(x in error_msg.lower() for x in 
                             ['503', 'overloaded', 'unavailable', 'timeout', '429', 'rate limit'])
            
            if attempt < max_retries - 1 and is_retryable:
                wait_time = retry_delay * (2 ** attempt)
                print(f"  Retryable error: {error_msg}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {attempt + 1} attempts: {error_msg}")
                raise


def save_raw_response(rating_text: str, response_folder: str, api_family: str, 
                     model_name: str, run: int, txt_name: str, chunk_idx: int) -> None:
    """Save raw API response to file."""
    os.makedirs(response_folder, exist_ok=True)
    chunk_filename = os.path.join(
        response_folder, 
        f"response_{api_family}_{model_name.replace('/', '_')}_run{run}_{txt_name}_chunk{chunk_idx}.txt"
    )
    with open(chunk_filename, "w", encoding="utf-8") as f:
        f.write(rating_text if rating_text else "No rating text available")


def process_single_model(api_family: str, model_name: str, transkripts: dict, 
                        instruction_prompts_list: list, outer_runs: int, temperature: float,
                        number_of_definitions_in_one_run: int, clients: dict, 
                        schema_instruction: str, Psychopathologies, psychopathologies_schema: dict, reference_df) -> tuple[list, bool]:
    """
    Process all transcripts for a single model.
    Returns (results_list, model_failed_flag)
    """
    api_provider = get_api_provider(api_family)
    model_parse_failed = False
    all_model_results = []
    
    for txt_name, meta in transkripts.items():
        with open(meta["path"], encoding="utf-8") as f:
            transcript = f.read().strip()
        
        rows = []
        
        for run in range(1, outer_runs + 1):
            partial_rows = []
            
            for chunk_idx, chunk in enumerate(instruction_prompts_list, start=1):
                chunk_prompt = chunk["prompt"]
                allowed_numbers = set(chunk["numbers"])
                
                try:
                    # Call API with retry
                    rating_text = process_chunk_with_retry(
                        api_family, model_name, chunk_prompt, transcript,
                        temperature, run, txt_name, chunk_idx,
                        clients, schema_instruction, Psychopathologies, psychopathologies_schema
                    )
                    
                    # Save raw response
                    save_raw_response(rating_text, "./raw_responses/", api_family, 
                                    model_name, run, txt_name, chunk_idx)
                    
                    # Parse JSON and convert to dataframe
                    data = json.loads(rating_text)
                    df_row = json_to_dataframe(
                        data_list=data,
                        video_id=meta["video_id"],
                        video_type=meta["video_type"],
                        site=f"{api_family}_{number_of_definitions_in_one_run}",
                        api=api_provider,
                        model_name=model_name.replace('/', '_'),
                        rater_id=run,
                        allowed_numbers=allowed_numbers,
                        only_allowed_columns=True,
                        reference_df=reference_df
                    )
                    df_row["run"] = run
                    df_row["txt_name"] = txt_name
                    df_row["chunk_index"] = chunk_idx
                    partial_rows.append(df_row)
                
                except Exception as e:
                    os.makedirs("./parsing_errors/", exist_ok=True)
                    print(f"Parse error run {run} {api_family} {model_name.replace('/', '_')} {txt_name}: {e}")
                    err_file = os.path.join(
                        "./parsing_errors/",
                        f"error_answer_{api_family}_{model_name.replace('/', '_')}_run{run}_{txt_name}_chunk{chunk_idx}.txt"
                    )
                    with open(err_file, "w", encoding="utf-8") as f:
                        f.write(rating_text if rating_text else "No rating text")
                    model_parse_failed = True
                    break
            
            if model_parse_failed:
                print(f"Skipping remaining chunks/runs for model {model_name} due to parse error.")
                break
            
            # Merge chunks for this run
            merged = merge_partial_rows(partial_rows)
            if merged is not None:
                merged = finalize_merged_row(merged, reference_df=reference_df)
            
            print(f"Finalized merged row for run {run} {api_family} {model_name} {txt_name}. "
                  f"Columns: {merged.shape[1]}, NaNs: {merged.isna().sum().sum()}")
            rows.append(merged)
        
        if rows:
            result_df = pd.concat(rows, ignore_index=True)
            all_model_results.append(result_df)
    
    return all_model_results, model_parse_failed


def run_psychopathology_rating(models: list, transkripts: dict, instruction_prompts_list: list,
                              outer_runs: int, temperature: float, number_of_definitions_in_one_run: int,
                              clients: dict, schema_instruction: str, Psychopathologies, psychopathologies_schema: Dict[str, Any],
                              reference_df, rating_folder: str) -> pd.DataFrame:
    """
    Main orchestration function. Runs all models on all transcripts.
    Returns combined results dataframe.
    """
    print(f"Instruction prompt example (with {number_of_definitions_in_one_run} definitions):\n"
          f"{instruction_prompts_list[0]['prompt'][:1000]}...\n")
    print(f"STARTING WITH {outer_runs} runs, {number_of_definitions_in_one_run} definitions per run, "
          f"temperature={temperature}")
    
    all_results = []
    time_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    for api_family, model_name in models:
        results, _ = process_single_model(
            api_family, model_name, transkripts,
            instruction_prompts_list, outer_runs, temperature,
            number_of_definitions_in_one_run, clients,
            schema_instruction, Psychopathologies, psychopathologies_schema, reference_df
        )
        all_results.extend(results)
    
    all_results_df = pd.concat(all_results, ignore_index=True)
    output_path = os.path.join(rating_folder, 
                              f"all_runs_combined_definitions_{number_of_definitions_in_one_run}_{time_tag}_{temperature}.csv")
    all_results_df.to_csv(output_path, index=False)
    print(f"Saved combined results to {output_path}")
    
    return all_results_df