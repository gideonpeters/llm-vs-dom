import os

import pandas as pd
import json
from dotenv import load_dotenv
import time
import argparse
from openai import OpenAI

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request


from utils import (
    LLMS,
    PROMPTS,
    TASKS,
    batch_format_prompt,
    format_prompt,
    batch_parse_json,
    parse_html,
    completion,
    setup_logger,
    formatted_audits,
    estimate_tokens,
    is_past_5pm_edt
)

logger = setup_logger(__name__)

def is_valid_audit(audit):
  if((audit['scoreDisplayMode'] == 'notApplicable') or
    (audit['scoreDisplayMode'] == 'binary' and audit['score'] == 1) or
    (audit['scoreDisplayMode'] == 'informative') or
    (audit['scoreDisplayMode'] == 'manual') or
    (audit['scoreDisplayMode'] == 'error') or
    (audit['scoreDisplayMode'] == 'metricSavings' and audit['score'] == 1) or
    (audit['scoreDisplayMode'] == 'numeric' and audit['score'] == 1)):
    return False

  return True

def get_audits(dom_name: str, with_location = False):
    """
    Get the audits for a specific domain.

    Args:
        dom_name (str): The name of the domain.
        with_location (bool): Whether to include location in the audits.

    Returns:
        list: A list of audit data.
    """
    dom_path = os.path.join("./../dataset/lh-original-reports", f"{dom_name}.json")

    with open(dom_path, 'r') as file:
        audits = json.load(file)

    audits = [audit for key, audit in audits['audits'].items() if is_valid_audit(audit)]

    print(type(audits))

    return audits

def get_chunks_audits(dom_name: str, prompt_name: str = "eval-html"):
    audits = get_audits(dom_name=dom_name)

    audit_text = formatted_audits(audits)

    logger.info("Loaded %s audits for inference on task.", len(audits))

    dom_path = os.path.join("./../dataset/chunks", f"{dom_name}.json") 

    chunks = []
    with open(dom_path, 'r') as file:
        chunks = json.load(file)

    chunks_df = pd.DataFrame(chunks)
    # chunks_df = chunks_df[~chunks_df['id'].isin(["script_store", "style_store"])]

    for ix, chunk in chunks_df.iterrows():
        chunks_df.loc[ix, 'no_of_issues'] = len(audits)
        chunks_df.loc[ix, 'audit_issues'] = audit_text
        chunks_df.loc[ix, 'start_time'] = time.time()
        chunk_dict = chunks_df.loc[ix].to_dict()
        messages = format_prompt(PROMPTS[prompt_name], chunk_dict)
        chunks_df.loc[ix, 'prompt'] = json.dumps(messages)
        chunks_df.loc[ix, 'message_tokens'] = estimate_tokens(messages)

    return chunks_df, audits, audit_text

all_batch_results = {}
batch_summary = {}
def main_with_anthropic(prompt_name: str, dom_name: str):
    chunks_df, audits, audit_text = get_chunks_audits(dom_name=dom_name)

    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"),
    )

    def retrieve_results():
        with open(os.path.join(f"./../results/batches/{model_name}", "_summary.json"), 'r') as file:
            batch_summary = json.load(file)

        MESSAGE_BATCH_ID = batch_summary[dom_name]
    
        if MESSAGE_BATCH_ID:
            message_batch = message_batch = client.messages.batches.retrieve(
                MESSAGE_BATCH_ID
            ) 

            # if message_batch.processing_status != "ended" or is_past_5pm_edt(message_batch.created_at) is False:
            #     continue

            if message_batch.processing_status == "ended":            
                for result in client.messages.batches.results(
                    MESSAGE_BATCH_ID,
                ):
                    # print(f"result: {result}")
                    result_custom_id = result.custom_id
                    end_time_float = message_batch.ended_at.timestamp() if hasattr(message_batch.ended_at, 'timestamp') else time.mktime(message_batch.ended_at.timetuple())
                    chunks_df.loc[chunks_df['id'] == result_custom_id, 'end_time'] = end_time_float
                    chunks_df.loc[chunks_df['id'] == result_custom_id, 'time_taken'] = chunks_df.loc[chunks_df['id'] == result_custom_id, 'end_time'] - chunks_df.loc[chunks_df['id'] == result_custom_id, 'start_time']
                    
                    contents = result.result.message.content if hasattr(result, 'result') and hasattr(result.result, 'message') else ""
                    # join all the contents
                    contents = [content.text if hasattr(content, 'text') and content.text is not None  else "" for content in contents]
                    # print(contents)
                    contents = "".join(contents)
                    contents = parse_html(contents)
                    chunks_df.loc[chunks_df['id'] == result_custom_id, 'completion_tokens'] = estimate_tokens(contents)
                    chunks_df.loc[chunks_df['id'] == result_custom_id, 'completion'] = contents

                # store as csv
                output_path = os.path.join("./../results/evaluations/claude-3-7-sonnet-20250219-non-reasoning")
                if not os.path.exists(output_path):
                    os.makedirs(output_path)

                chunks_df.to_csv(os.path.join(output_path, f"{dom_name}.csv"), index=False)
                logger.info("Saved evaluation for %s to %s", dom_name, output_path)
                # store as jsonl
                with open(os.path.join(output_path, f"{dom_name}.jsonl"), 'w') as file:
                    for ix, chunk in chunks_df.iterrows():
                        file.write(json.dumps(chunk.to_dict()) + "\n")
                # store as json
                with open(os.path.join(output_path, f"{dom_name}.json"), 'w') as file:
                    json.dump(chunks_df.to_dict(orient='records'), file, indent=4)

                logger.info("Saved evaluation for %s to %s", dom_name, output_path)
            else:
                logger.info("Batch job for %s is %s", dom_name, message_batch.processing_status)

    retrieve_results()

    def run_results():
        requests = []
        for ix, chunk in chunks_df.iterrows():
            if(chunks_df.loc[ix, 'id'] == "script_store" or chunks_df.loc[ix, 'id'] == "style_store"):
                logger.info("Skipping %s chunk %s of %s", chunks_df.loc[ix, 'id'], (ix + 1), len(chunks_df))
                continue

            messages = json.loads(chunks_df.loc[ix, 'prompt'])
            request = Request(
                custom_id=chunks_df.loc[ix, 'id'],
                params=MessageCreateParamsNonStreaming(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=128000,
                    # thinking={
                    #     "type": "enabled",
                    #     "budget_tokens": 32000
                    # },
                    messages=messages,
                )
            )
            requests.append(request)

        message_batch = client.messages.batches.create(requests=requests)
        batch_summary[dom_name] = message_batch.id
        path_to_batch_summary = os.path.join("./../results/batches/claude-3.7")
        if not os.path.exists(path_to_batch_summary):
            os.makedirs(path_to_batch_summary)

        with open(os.path.join(path_to_batch_summary, "_summary.json"), 'w') as file:
            json.dump(batch_summary, file, indent=4)
        logger.info("Created %s batch for %s with %s chunks", "claude-3-7-sonnet-20250219", dom_name, len(chunks_df))

    # run_results()


def main_with_gpt(prompt_name: str, dom_name: str, model_name: str):
    chunks_df, audits, audit_text = get_chunks_audits(dom_name=dom_name)

    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )

    def run_results():
        def create_batch():
            messages = []
            for ix, chunk in chunks_df.iterrows():
                if (chunks_df.loc[ix, 'id'] == "script_store" or chunks_df.loc[ix, 'id'] == "style_store"):
                    logger.info("Skipping %s chunk %s of %s", chunks_df.loc[ix, 'id'], (ix + 1), len(chunks_df))
                    continue

                messages.append({
                    "custom_id": f"{dom_name}_{chunks_df.loc[ix, 'id']}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_name,
                        # "temperature": 0.0,
                        "messages": json.loads(chunks_df.loc[ix, 'prompt']),
                        # "max_tokens": 32000,
                        "max_completion_tokens": 32000,
                    },
                })
            return messages
        
        # create a jsonl file with the messages
        batch_path = f"./../results/batches/{model_name}"

        if not os.path.exists(batch_path):
            os.makedirs(batch_path)

        with open(os.path.join(batch_path, f"{dom_name}.jsonl"), 'w') as file:
            for message in create_batch():
                file.write(json.dumps(message) + "\n")
        logger.info("Saved %s batch for %s to %s", model_name, dom_name, batch_path)

        batch_input_file = client.files.create(
            file=open(os.path.join(batch_path, f"{dom_name}.jsonl"), "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id
        logger.info("Created batch input file %s for %s", batch_input_file_id, dom_name)

        batch_job = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": f"Batch for {dom_name} with {len(chunks_df)} chunks",
                "webpage": dom_name,
                "model": model_name,
            }
        )
        batch_summary[dom_name] = batch_job.id
        with open(os.path.join(batch_path, "_summary.json"), 'w') as file:
            json.dump(batch_summary, file, indent=4)

        logger.info("Created %s batch for %s with %s chunks", model_name, dom_name, len(chunks_df))

    # run_results()

    def retrieve_results():
        with open(os.path.join(f"./../results/batches/{model_name}", "_summary.json"), 'r') as file:
            batch_summary = json.load(file)

        batch_job = client.batches.retrieve(
            batch_summary[dom_name]
        )

        if batch_job is not None:
            logger.info("Batch job %s for %s is %s", batch_job.id, dom_name, batch_job.status)

            if batch_job.status == "completed":
                result_file = batch_job.output_file_id
                result_file_content = client.files.content(
                    result_file
                )
                result_file_content = result_file_content.text

                batch_output_path = f"./../results/batches/{model_name}/output"
                if not os.path.exists(batch_output_path):
                    os.makedirs(batch_output_path)

                with open(os.path.join(batch_output_path, f"{dom_name}.jsonl"), 'w') as file:
                    file.write(result_file_content)
                logger.info("Saved %s batch output for %s to %s", model_name, dom_name, batch_output_path)

    retrieve_results()



def main(prompt_name: str, evaluator_model: str, dom_name: str):
    chunks_df, audits, audit_text = get_chunks_audits(dom_name=dom_name)
    
    model = LLMS[evaluator_model]
    for ix, chunk in chunks_df.iterrows():
        if(chunks_df.loc[ix, 'id'] == "script_store" or chunks_df.loc[ix, 'id'] == "style_store"):
            logger.info("Skipping %s chunk %s of %s", chunks_df.loc[ix, 'id'], (ix + 1), len(chunks_df))
            continue

        logger.info("Processing chunk %s of %s with %s tokens", (ix + 1), len(chunks_df), chunks_df.loc[ix, 'message_tokens'])

        completions = completion(
            messages=json.loads(chunks_df.loc[ix, 'prompt']),
            custom_llm_provider=model['custom_llm_provider'],
            **model['model_parameters'], 
            **model['sample_parameters'],
            num_retries=3,
            timeout=300
        )
        chunks_df.loc[ix, 'end_time'] = time.time()
        chunks_df.loc[ix, 'time_taken'] = chunks_df.loc[ix, 'end_time'] - chunks_df.loc[ix, 'start_time']
        chunks_df.loc[ix, 'completion_tokens'] = estimate_tokens(completions)
        chunks_df.loc[ix, 'completion'] = parse_html(completions)

        output_path = os.path.join(f"./../results/evaluations/{evaluator_model}")
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # store as csv
        chunks_df.to_csv(os.path.join(output_path, f"{dom_name}.csv"), index=False)
        logger.info("Saved evaluation for chunk %s to %s", (ix + 1), output_path)

        # store as json
        with open(os.path.join(output_path, f"{dom_name}.json"), 'w') as file:
            json.dump(chunks_df.to_dict(orient='records'), file, indent=4)

if __name__ == "__main__":

    # get the model name from the command-line arguments
    # python scripts/dom_modifier.py --model_name <model_name>
    parser = argparse.ArgumentParser(description="Run the DOM modifier with a specified model.")
    parser.add_argument('--model_name', required=True, help='Name of the model to use')
    
    args = parser.parse_args()

    model_name = args.model_name
    print(f"Running DOM modifier with model: {model_name}")

    # MODELS = {
    #     "reasoning": {
    #         "gpt" : ["gpt-4o-mini", "gpt-4o"],
    #         "claude": ["claude-3.7"],
    #         "deepseek": ["deepseek-R1"],
    #         "gemini": ["gemini-2.5"]
    #     },
    #     "nonreasoning": {
    #         "gpt" : [gpt-4.1, o4-mini],
    #         "claude": ["claude-3.7-reasoning"],
    #         "deepseek": ["deepseek-V3"],
    #         "gemini": ["gemini-2.5"]
    #     },
    #     "small": {

    #     }
    # }

    html_pages = [
        # "airbnb",
        # "ebay", 
        # "github", 
        # "medium", 
        # "netflix", 
        # "pinterest", 
        # "quora", 
        # "reddit", 
        # "twitch", 
        # "walmart", 
        # "youtube",
        # "facebook", 
        # "twitter", 
        # "linkedin", 
        "aliexpress"
    ]

    for html_page in html_pages:
        try: 
            logger.info("Processing %s", html_page)
            if model_name == "claude-3.7":
                main_with_anthropic(prompt_name="eval-html", dom_name=html_page)
                # break
            elif model_name == "gpt-4.1":
                main_with_gpt(prompt_name="eval-html", dom_name=html_page, model_name=model_name)
            elif model_name == "o4-mini":
                main_with_gpt(prompt_name="eval-html", dom_name=html_page, model_name=model_name)
            else:
                main(prompt_name="eval-html", evaluator_model=model_name, dom_name=html_page)
            logger.info("Completed processing %s", html_page)
        except Exception as e:
            logger.error("%s => Error processing %s: %s", model_name, html_page, e)
            continue
