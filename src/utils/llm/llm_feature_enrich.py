
# This script is used to enrich subgraph: feature enrich

import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from datasets import load_dataset
from llm.prompt_builder import *
from llm.llm_client import *
from tqdm import tqdm
import re
from collections import defaultdict
import pandas as pd
from datasets import Dataset, DatasetDict
import asyncio
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime

def llm_feature_enrich_main(dataset_path=None,llm=None,openai_api_key=None,base_url=None,resume_path=None):
    return asyncio.run(llm_feature_enrich(dataset_path,llm,openai_api_key,base_url,resume_path))

async def llm_feature_enrich(dataset_path=None,llm=None,openai_api_key=None,base_url=None,resume_path=None):
    dataset_name = "cwq"
    if "cwq" in dataset_path:
        dataset_name = "cwq"
    elif "webqsp" in dataset_path:
        dataset_name = "webqsp"

    dataset = load_dataset("parquet", data_files={'test': dataset_path})

    # Print dataset information
    print(dataset)

    # Get current time
    current_time = datetime.now()

    # Format time as string (e.g., "2023-10-10_14-30-00"), replace this with the unfinished parquet name to continue
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Save file, final output file name format: {dataset_name}_{llm}_{qa}
    if resume_path == None:
        write_data_dir = f"preprocess_datasets/feature_enrich_datasets/{dataset_name}_{llm}_feature_enrich_{time_str}.parquet"
    else:
        write_data_dir = resume_path


    # Open the file, create if it doesn't exist
    # Ensure directory exists
    os.makedirs(os.path.dirname(write_data_dir), exist_ok=True)

    # Store results after question decomposition
    feature_enrich_dataset = None
    finished_id = []

    # Check if file exists, create if it doesn't
    if not os.path.exists(write_data_dir):
        # If file doesn't exist, create an empty DataFrame and save as parquet file
        df = pd.DataFrame()  # Create empty DataFrame
        df.to_parquet(write_data_dir)
        print(f"File doesn't exist, created new empty file: {write_data_dir}")
        # Initialize dataset
        feature_enrich_dataset = DatasetDict({
                "test": Dataset.from_dict({
                "id": "",
                "question": "",
                "user_queries":[],
                "answer": [],
                "q_entity": [],
                "a_entity": [],
                "graph": [],
                "pruned_graph": [],
                "choices": [],
                "feature_enrich_triples":[],
                "filter_triples":[]
            })
        })
    else:
        print(f"File already exists: {write_data_dir}, will continue LLM feature enrich task from this file")
        feature_enrich_dataset = load_dataset("parquet", data_files={'test': write_data_dir})
        # Check existing question_ids
        for sample in feature_enrich_dataset["test"]:
            finished_id.append(sample["id"])
    
    ###############################################################################################################
    # Build prompt
    ###############################################################################################################

    mode = "feature_enrich"
    feature_enrich_prompt = {}
    # Convert dataset["test"] to a fast lookup dictionary with id as key
    id_to_example_map = {example["id"]: example for example in dataset["test"]}

    # List for temporarily storing results
    batch_results = []
    filter_batch_size = 5  # Set batch size

    # Build prompt
    with tqdm(dataset['test'], desc="Building feature enrich prompts") as pbar:
        for each_sample in pbar:
            # Skip if already exists
            if each_sample["id"] in finished_id:
                continue
            question_id = each_sample["id"]
            filter_triples = each_sample["filter_triples"]
            pruned_graph = each_sample["pruned_graph"]

            # Extract related entities from filter_triples, construct corresponding data structure: key(entity)-value:"user_queries":[question list],"triples":[triple list]
            # Function get_related_questions_triples
            def get_related_questions_triples(filter_triples,pruned_graph):
                entity_question_triple_dict = {}
                for ind,question_list in enumerate(filter_triples):
                    if question_list != None and len(question_list) > 0:
                        corresponding_triple = pruned_graph[ind]
                        s,p,o = corresponding_triple
                        if s in entity_question_triple_dict:
                            entity_question_triple_dict[s]["triples"].append(corresponding_triple)
                            # Iterate through question_list and only add non-existing user_query
                            for user_query in question_list:
                                if user_query not in entity_question_triple_dict[s]["user_queries"]:
                                    entity_question_triple_dict[s]["user_queries"].append(user_query)
                        else:
                            entity_question_triple_dict[s] = {"triples": [],"user_queries":[]}
                            entity_question_triple_dict[s]["triples"].append((s,p,o))
                            for user_query in question_list:
                                entity_question_triple_dict[s]["user_queries"].append(user_query)
                        if o in entity_question_triple_dict:
                            entity_question_triple_dict[o]["triples"].append(corresponding_triple)
                            # Iterate through question_list and only add non-existing user_query
                            for user_query in question_list:
                                if user_query not in entity_question_triple_dict[o]["user_queries"]:
                                    entity_question_triple_dict[o]["user_queries"].append(user_query)
                        else:
                            entity_question_triple_dict[o] = {"triples": [],"user_queries":[]}
                            entity_question_triple_dict[o]["triples"].append((s,p,o))
                            for user_query in question_list:
                                entity_question_triple_dict[o]["user_queries"].append(user_query)
                return entity_question_triple_dict

            # Fill prompt based on data structure
            entity_question_triple_dict = {}
            entity_question_triple_dict = get_related_questions_triples(filter_triples,pruned_graph)
            
            # If filter triples is empty [], skip this sample
            if len(entity_question_triple_dict) == 0:
                # Fast lookup corresponding record by question_id
                example = id_to_example_map.get(each_sample["id"])
                if example:
                    # Add prediction result to example
                    example_with_prediction = {**example, "feature_enrich_triples": []}
                    batch_results.append(example_with_prediction)

                # If there are remaining results, write to file
                if len(batch_results) >= filter_batch_size:
                    if len(feature_enrich_dataset["test"]) == 0:  # If feature_enrich_dataset["test"] is empty
                        feature_enrich_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
                    else:
                        feature_enrich_dataset["test"] = Dataset.from_dict({
                            key: feature_enrich_dataset["test"][key] + [ex[key] for ex in batch_results]
                            for key in feature_enrich_dataset["test"].column_names
                        })
                    feature_enrich_dataset["test"].to_parquet(write_data_dir)
                    # Write to file
                    feature_enrich_dataset["test"].to_parquet(write_data_dir)
                    batch_results = []  # Clear temporary storage
                continue

            # Assemble prompt
            def generate_prompt(entity_question_triple_dict):
                # Initialize result string
                prompt_result = "Input:\nentity List:\n"
                
                # Iterate through each entity in the dictionary
                for entity, value in entity_question_triple_dict.items():
                    # Get current entity's questions and triples
                    user_queries = value.get("user_queries", [])
                    triples = value.get("triples", [])
                    
                    # Fill with None if list is empty
                    user_queries_str = "-".join(user_queries) if user_queries else "None"
                    triples_str = "-".join(["-".join(triple) for triple in triples]) if triples else "None"

                    # Fill template
                    entity_block = f"[${entity}$ context]\nrelavent triple(s):{triples_str}\nrelavent user query(ies):{user_queries_str}\n[/${entity}$ context]\n"
                    prompt_result = prompt_result + entity_block
                return prompt_result
            each_prompt = generate_prompt(entity_question_triple_dict)
            # Call PromptBuilder, pass in question
            prompt_builder = PromptBuilder(each_prompt,mode)
            feature_enrich_prompt[each_sample["id"]] = prompt_builder.build_prompt()

    # If there are remaining results, write to file
    if batch_results:
        if len(feature_enrich_dataset["test"]) == 0:  # If feature_enrich_dataset["test"] is empty
            feature_enrich_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
        else:
            feature_enrich_dataset["test"] = Dataset.from_dict({
                key: feature_enrich_dataset["test"][key] + [ex[key] for ex in batch_results]
                for key in feature_enrich_dataset["test"].column_names
            })
        feature_enrich_dataset["test"].to_parquet(write_data_dir)
        # Write to file
        feature_enrich_dataset["test"].to_parquet(write_data_dir)
        batch_results = []  # Clear temporary storage

    # Call LLM to translate triples
    llm_chat = llm_client(base_url=base_url,openai_api_keys=[openai_api_key],model=llm) 

    # Define async function to process single request
    async def process_single_query(llm_chat, question_id, each_triples_trans_prompt):
        response = await llm_chat.response(each_triples_trans_prompt)
        return question_id, response

    # Define async function to process multiple requests and handle results in completion order
    tasks = [
        process_single_query(llm_chat, question_id, each_feature_enrich_prompt)
        for question_id, each_feature_enrich_prompt in feature_enrich_prompt.items()
    ]
    
    # Use tqdm_asyncio to show progress bar
    with tqdm_asyncio(desc=f"Call {llm} for feature_enrich", total=len(feature_enrich_prompt)) as pbar:
        for future in asyncio.as_completed(tasks):
            result = await future  # Wait for a task to complete
            question_id, response = result  # Destructure from return value
            pbar.update(1)  # Update progress bar

    # with tqdm(feature_enrich_prompt.items(),desc=f"Call {llm} for feature_enrich") as pbar:
    #     for question_id,each_feature_enrich_prompt in pbar:
    #         pbar.set_postfix(current_question=question_id)
    #         response = llm_chat.response(each_feature_enrich_prompt,mode)

            def extract_triples_from_response(response):
                """
                Extract content between {result} and {/result} from response string and parse as a triple list.
                """
                # Use regex to extract content between {result} and {/result}
                match = re.search(r"\{result\}([\s\S]*?)\{/result\}", response)
                if not match:
                    # If no {result} block is matched, return empty list
                    return []

                # Extract content part (i.e., content between {result} and {/result})
                content = match.group(1).strip()

                # Parse content, convert each triple string to a list of length 3
                triples = []
                for line in content.splitlines():
                    # Remove extra whitespace
                    line = line.strip()
                    # Skip empty lines
                    if not line:
                        continue
                    # Ensure line starts and ends with parentheses (this is triple format)
                    if line.startswith("(") and line.endswith(")"):
                        # Remove parentheses and split by comma
                        triple_parts = line[1:-1].split(", ")
                        if len(triple_parts) == 3:
                            triples.append(triple_parts)  # Ensure triple consists of three parts

                return triples

            processed_answer = extract_triples_from_response(response)
            # Fast lookup corresponding record by question_id
            example = id_to_example_map.get(question_id)
            if example:
                # Add prediction result to example
                example_with_prediction = {**example, "feature_enrich_triples": processed_answer}
                batch_results.append(example_with_prediction)

            # When batch results reach filter_batch_size, perform one write
            if len(batch_results) >= filter_batch_size:
                if len(feature_enrich_dataset["test"]) == 0:  # If feature_enrich_dataset["test"] is empty
                    feature_enrich_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
                else:
                    # Batch merge to existing Dataset
                    feature_enrich_dataset["test"] = Dataset.from_dict({
                        key: feature_enrich_dataset["test"][key] + [ex[key] for ex in batch_results]
                        for key in feature_enrich_dataset["test"].column_names
                    })

                # Write to file
                feature_enrich_dataset["test"].to_parquet(write_data_dir)
                batch_results = []  # Clear temporary storage
                
    # If there are remaining results, write to file
    if batch_results:
        if len(feature_enrich_dataset["test"]) == 0:  # If feature_enrich_dataset["test"] is empty
            feature_enrich_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
        else:
            feature_enrich_dataset["test"] = Dataset.from_dict({
                key: feature_enrich_dataset["test"][key] + [ex[key] for ex in batch_results]
                for key in feature_enrich_dataset["test"].column_names
            })
        feature_enrich_dataset["test"].to_parquet(write_data_dir)
    
    print(f"Completed feature enrich task for {dataset_name} dataset!")
    print("Example after feature enrich:",feature_enrich_dataset["test"])
