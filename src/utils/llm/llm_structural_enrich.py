# This script is used for structural enrichment of pruned_graph
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
from src.graph.graph import Graph
import asyncio
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime


def llm_structural_enrich_main(dataset_path=None,llm=None,openai_api_key=None,base_url=None,resume_path=None):
    return asyncio.run(llm_structural_enrich(dataset_path,llm,openai_api_key,base_url,resume_path))

async def llm_structural_enrich(dataset_path=None,llm=None,openai_api_key=None,base_url=None,resume_path=None):
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

    # Format time as string (e.g., "2023-10-10_14-30-00"), replace this with the name of the unfinished parquet from last time to continue
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Save file, final output file name is {dataset_name}_{llm}_{qa}
    if resume_path == None:
        write_data_dir = f"preprocess_datasets/structural_enrich_datasets/{dataset_name}_{llm}_structural_enrich_{time_str}.parquet"
    else:
        write_data_dir = resume_path
    # Open the file, if it doesn't exist, create it
    # Ensure directory exists
    os.makedirs(os.path.dirname(write_data_dir), exist_ok=True)

    # Store results after question decomposition
    structural_enrich_dataset = None
    finished_id = []

    # Check if file exists, if not, create file
    if not os.path.exists(write_data_dir):
        # If file doesn't exist, create an empty DataFrame and save as parquet file
        df = pd.DataFrame()  # Create empty DataFrame
        df.to_parquet(write_data_dir)
        print(f"File doesn't exist, created new empty file: {write_data_dir}")
        # Initialize dataset
        structural_enrich_dataset = DatasetDict({
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
                "structural_enrich_triples":[],
                "filter_triples":[]
            })
        })
    else:
        print(f"File already exists: {write_data_dir}, will continue LLM structural enrich task from this file")
        structural_enrich_dataset = load_dataset("parquet", data_files={'test': write_data_dir})
        # Check existing question_id
        # if len(structural_enrich_dataset["test"]) != 0:
        for sample in structural_enrich_dataset["test"]:
            finished_id.append(sample["id"])

    ###############################################################################################################
    # For each question in the dataset, filter out the index of triples related to it
    ###############################################################################################################
    mode = 'structural_enrich'
    structural_enrich_prompt = {}
    my_graph = {}
    # Convert dataset["test"] to a fast retrieval dictionary with id as key
    id_to_example_map = {example["id"]: example for example in dataset["test"]}

    # List for temporarily storing results
    batch_results = []
    filter_batch_size = 10  # Set batch size

    # Build prompt
    with tqdm(dataset['test'], desc="Building structural_enrich prompts") as pbar:
        for each_sample in pbar:
            pbar.set_postfix(current_question=each_sample["question"])
            # Skip existing samples
            if each_sample["id"] in finished_id:
                continue

            question_id = each_sample["id"]
            filter_triples = each_sample["filter_triples"]
            pruned_graph = each_sample["pruned_graph"]
            triple_question_dict= {}

            # Graph data structure
            this_graph = Graph()

            def get_triple_query(filter_triples,pruned_graph):
                triple_question_dict = {}
                for ind,question_list in enumerate(filter_triples):
                    if question_list != None and len(question_list) > 0:
                        # Maintain graph structure
                        s,p,o = pruned_graph[ind]
                        this_graph.add_triplet((s,p,o))
                        # Maintain triple-question dictionary structure
                        if (s,p,o) in triple_question_dict:
                            for question in question_list:
                                if question not in triple_question_dict[(s,p,o)]:
                                    triple_question_dict[(s,p,o)].append(question)
                        else:
                            triple_question_dict[(s,p,o)] = []
                            triple_question_dict[(s,p,o)] = question_list
                return triple_question_dict
            
            triple_question_dict = get_triple_query(filter_triples,pruned_graph)

            # No content needs to be enriched
            if len(triple_question_dict) == 0:
                # Quickly retrieve corresponding record by question_id
                example = id_to_example_map.get(each_sample["id"])
                if example:
                    # Add prediction result to example
                    example_with_prediction = {**example, "structural_enrich_triples": []}
                    batch_results.append(example_with_prediction)
                # When batch results reach filter_batch_size, perform one write
                if len(batch_results) >= filter_batch_size:
                    if len(structural_enrich_dataset["test"]) == 0:  # If structural_enrich_dataset["test"] is empty
                        structural_enrich_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
                    else:
                        # Batch merge to existing Dataset
                        structural_enrich_dataset["test"] = Dataset.from_dict({
                            key: structural_enrich_dataset["test"][key] + [ex[key] for ex in batch_results]
                            for key in structural_enrich_dataset["test"].column_names
                        })

                    # Write to file
                    structural_enrich_dataset["test"].to_parquet(write_data_dir)
                    batch_results = []  # Clear temporary storage
                continue

            def assemble_prompt(triple_question_dict, my_graph, topics):
                # Helper function to format a triple into a string
                def format_triple(triple):
                    return f"({triple[0]},{triple[1]},{triple[2]})"

                # Helper function to format a two-hop path into a string
                def format_two_hop_path(two_hop_path):
                    return "->".join([format_triple(triple) for triple in two_hop_path])

                # Start building the prompt
                prompt = "Input:\n"

                # Add the triple-question mappings
                for triple, questions in triple_question_dict.items():
                    triple_str = format_triple(triple)
                    question_str = "-".join(questions)
                    prompt += f"{triple_str}-{question_str}\n"

                # Add 1-hop paths
                one_hop_paths = my_graph[question_id].get_one_hop_paths(topics)
                prompt += "1-hop:\n"
                for path in one_hop_paths:
                    prompt += f"{format_triple(path)}\n"

                # Add 2-hop paths
                two_hop_paths = my_graph[question_id].get_two_hop_paths(topics)
                prompt += "2-hop:\n"
                for path in two_hop_paths:
                    prompt += f"{format_two_hop_path(path)}\n"

                return prompt
            
            # Each question corresponds to a graph
            my_graph[each_sample["id"]] = this_graph
            # Get topics
            topics = each_sample["q_entity"]
            each_prompt = assemble_prompt(triple_question_dict, my_graph,topics)
            # Call PromptBuilder, pass in question
            prompt_builder = PromptBuilder(each_prompt,mode)
            structural_enrich_prompt[each_sample["id"]] = prompt_builder.build_prompt()

    # Call llm to translate triples
    llm_chat = llm_client(base_url=base_url,openai_api_keys=[openai_api_key],model=llm) 

    if batch_results:
        if len(structural_enrich_dataset["test"]) == 0:  # If structural_enrich_dataset["test"] is empty
            structural_enrich_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
        else:
            # Batch merge to existing Dataset
            structural_enrich_dataset["test"] = Dataset.from_dict({
                key: structural_enrich_dataset["test"][key] + [ex[key] for ex in batch_results]
                for key in structural_enrich_dataset["test"].column_names
            })

        # Write to file
        structural_enrich_dataset["test"].to_parquet(write_data_dir)
        batch_results = []  # Clear temporary storage
        
    # Define async function to process single request
    async def process_single_query(llm_chat, question_id, each_triples_trans_prompt):
        response = await llm_chat.response(each_triples_trans_prompt)
        return question_id, response

    # Define async function to process multiple requests and handle results in completion order
    tasks = [
        process_single_query(llm_chat, question_id, each_structural_enrich_prompt)
        for question_id, each_structural_enrich_prompt in structural_enrich_prompt.items()
    ]
    
    # Use tqdm_asyncio to show progress bar
    with tqdm_asyncio(desc=f"Call {llm} for structural_enrich", total=len(structural_enrich_prompt)) as pbar:
        for future in asyncio.as_completed(tasks):
            result = await future  # Wait for a task to complete
            question_id, response = result  # Destructure from return value
            pbar.update(1)  # Update progress bar

    # with tqdm(structural_enrich_prompt.items(),desc=f"Call {llm} for structural_enrich") as pbar:
    #     for question_id,each_structural_enrich_prompt in pbar:
    #         pbar.set_postfix(current_question=question_id)
    #         response = llm_chat.response(each_structural_enrich_prompt,mode)

            def extract_step_4_triples(response):
                # Find the part after "step 4:"
                # 1. Extract content after "Final output:"
                final_output_match = re.search(r"Final output:\n(.*)", response, re.DOTALL)
                # step_4_content = response.split("step 4:")[1].strip()

                if not final_output_match:
                    step_4_content = response.strip()
                else:
                    step_4_content = final_output_match.group(1).strip()

                step_4_content = step_4_content.replace('{/thought}', '').replace('{thought}', '')
                step_4_content = step_4_content.replace('{demonstrations}', '').replace('{/demonstrations}', '')
                step_4_content = step_4_content.replace('```', '')

                # Use a regular expression to extract all the triples
                # Match triples of the form (x, y, z)
                # triples = re.findall(r'$ ([^,]+),\s*([^,]+),\s*([^)]+) $', step_4_content)
                triples = step_4_content.split("\n")
                cleaned_triples = [
                    item.strip('()').split(',')  # Remove parentheses and split string by ", "
                    for item in triples
                ]

                cleaned_triples = [item for item in cleaned_triples if item != '' and len(item) >= 3]
                return cleaned_triples
            
            if response == None:
                new_added_triples = []
            else:
                new_added_triples = extract_step_4_triples(response)
            
            # Quickly retrieve corresponding record by question_id
            example = id_to_example_map.get(question_id)
            if example:
                # Add prediction result to example
                example_with_prediction = {**example, "structural_enrich_triples": new_added_triples}
                batch_results.append(example_with_prediction)

            # When batch results reach filter_batch_size, perform one write
            if len(batch_results) >= filter_batch_size:
                if len(structural_enrich_dataset["test"]) == 0:  # If structural_enrich_dataset["test"] is empty
                    structural_enrich_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
                else:
                    # Batch merge to existing Dataset
                    structural_enrich_dataset["test"] = Dataset.from_dict({
                        key: structural_enrich_dataset["test"][key] + [ex[key] for ex in batch_results]
                        for key in structural_enrich_dataset["test"].column_names
                    })

                # Write to file
                structural_enrich_dataset["test"].to_parquet(write_data_dir)
                batch_results = []  # Clear temporary storage
    
    # If there are remaining results, write to file
    if batch_results:
        if len(structural_enrich_dataset["test"]) == 0:  # If structural_enrich_dataset["test"] is empty
            structural_enrich_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
        else:
            structural_enrich_dataset["test"] = Dataset.from_dict({
                key: structural_enrich_dataset["test"][key] + [ex[key] for ex in batch_results]
                for key in structural_enrich_dataset["test"].column_names
            })
        structural_enrich_dataset["test"].to_parquet(write_data_dir)
    
    print(f"Completed structural enrich task for {dataset_name} dataset!")
    print("Sample after structural enrich:",structural_enrich_dataset["test"])