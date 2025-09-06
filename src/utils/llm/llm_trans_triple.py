import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from datasets import load_dataset
from llm.prompt_builder import *
from llm.llm_client import *
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, DatasetDict
import asyncio
from tqdm.asyncio import tqdm_asyncio
from datetime import datetime



def llm_trans_triple_main(dataset_path=None,llm=None,openai_api_key=None,base_url=None,resume_path=None):
    return asyncio.run(llm_trans_triple(dataset_path,llm,openai_api_key,base_url,resume_path))

async def llm_trans_triple(dataset_path=None,llm=None,openai_api_key=None,base_url=None,resume_path=None):
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
        write_data_dir = f"preprocess_datasets/triple_trans_datasets/{dataset_name}_{llm}_triple_trans_{time_str}.parquet"
    else:
        write_data_dir = resume_path

    # Open the file, if it doesn't exist, create it
    # Ensure directory exists
    os.makedirs(os.path.dirname(write_data_dir), exist_ok=True)

    triple_trans_dataset = None
    finished_id = []

    # Check if file exists, if not, create file
    if not os.path.exists(write_data_dir):
        # If file doesn't exist, create an empty DataFrame and save as parquet file
        df = pd.DataFrame()  # Create empty DataFrame
        df.to_parquet(write_data_dir)
        print(f"File doesn't exist, created new empty file: {write_data_dir}")
        # Initialize dataset
        triple_trans_dataset = DatasetDict({
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
                "triple_unit_queries":[]
            })
        })
    else:
        print(f"File already exists: {write_data_dir}, will continue triple translation task from this file")
        triple_trans_dataset = load_dataset("parquet", data_files={'test': write_data_dir})
        # Check existing question_id
        for sample in triple_trans_dataset["test"]:
            finished_id.append(sample["id"])

    # Build triple translation prompt
    mode = "triples_trans"
    triples_trans_prompt = {}
    with tqdm(dataset['test'], desc="Building triple transaction prompts") as pbar:
        for each_sample in pbar:
            # Filter out completed samples
            if each_sample["id"] in finished_id:
                continue

            pbar.set_postfix(current_question=each_sample["question"])
            sub_graph = each_sample["pruned_graph"]
            
            triple_text_list = []
            for triple in sub_graph:
                if isinstance(triple, list) and len(triple) == 3:  # Ensure each triple is a list of length 3
                    triple_text_list.append(f"({triple[0]}, {triple[1]}, {triple[2]})")
            
            # Use list comprehension to add sequence numbers to each element
            triple_text_list = [f"{i+1}.{text}" for i, text in enumerate(triple_text_list)]

            # Connect each triple text with newlines
            sub_graph_text = "\n".join(triple_text_list)
            # Add input prefix
            pre_text = "Input:\n" \
            "Triple(s):\n"

            sub_graph_text = pre_text + sub_graph_text
            # Call PromptBuilder, pass in question
            prompt_builder = PromptBuilder(sub_graph_text,mode)
            triples_trans_prompt[each_sample["id"]] = prompt_builder.build_prompt()

    
    # Call llm to translate triples
    llm_chat = llm_client(base_url=base_url,openai_api_keys=[openai_api_key],model=llm) 
    triple_queries = {}

    # await asyncio.gather(*(llm_chat.response(each_triples_trans_prompt,mode) for question_id,each_triples_trans_prompt in triples_trans_prompt.items()))
    
    # Convert dataset["test"] to a fast retrieval dictionary with id as key
    id_to_example_map = {example["id"]: example for example in dataset["test"]}

    # List for temporarily storing results
    batch_results = []
    filter_batch_size = 20  # Set batch size

    # Define async function to process single request
    async def process_single_query(llm_chat, question_id, each_triples_trans_prompt):
        response = await llm_chat.response(each_triples_trans_prompt)
        return question_id, response

    # Define async function to process multiple requests and handle results in completion order
    tasks = [
        process_single_query(llm_chat, question_id, each_triples_trans_prompt)
        for question_id, each_triples_trans_prompt in triples_trans_prompt.items()
    ]
    
    # Use tqdm_asyncio to show progress bar
    with tqdm_asyncio(desc=f"Call {llm} for triple transaction", total=len(triples_trans_prompt)) as pbar:
        for future in asyncio.as_completed(tasks):
            result = await future  # Wait for a task to complete
            question_id, response = result  # Destructure from return value
            pbar.update(1)  # Update progress bar
            # print(f"Processed {question_id}: {response}")  # Process each task result

    # with tqdm(triples_trans_prompt.items(),desc=f"Call {llm} for triple transaction") as pbar:
    #     for question_id,each_triples_trans_prompt in pbar:
    #         pbar.set_postfix(current_question=question_id)
    #         response = await llm_chat.response(each_triples_trans_prompt,mode)

            # Example response text
            # response = """Natural Language Question: 
            # (Beijing,located in,?):Which country does Beijing locate? (?,located in,China):What cities or places are located in China?
            # (Eiffel Tower, located in, ?):In which city is the Eiffel Tower located? (?, located in, Paris):What landmarks or places are located in Paris?
            # (Apple, founded by, ?):Who founded Apple? (?, founded by, Steve Jobs):Which companies or organizations were founded by Steve Jobs?
            # (Python, created by, ?):Who created Python? (?, created by, Guido van Rossum):What programming languages or projects were created by Guido van Rossum?
            # (Tesla, CEO of, ?):Who is the CEO of Tesla? (?, CEO of, Elon Musk):Which companies or organizations have Elon Musk as their CEO?
            # """
            def extract_triple_queries(response):
                # Initialize final result list
                triple_queries = []
                
                # First check if there is "Natural Language Question: "
                start_index = response.find("Natural Language Question:")
                if start_index != -1:
                    # If exists, extract content starting from "Natural Language Question: "
                    response = response[start_index + len("Natural Language Question:"):]
                
                # Split content by lines
                lines = response.splitlines()
                
                for line in lines:
                    # Skip empty lines
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Find the separator ":" between triple part and corresponding question
                    part = line.split("<SEP>", 2)
                    if len(part) == 3:
                        left,right1,right2 = part
                        triple_queries.append([right1,right2])
                    elif len(part) == 2:
                        left,right = part
                        triple_queries.append([right])
                    else: 
                        # Exception, can be detected when checking subgraph length
                        continue
                        
                return triple_queries
            
            processed_answer = extract_triple_queries(response)
            triple_queries[question_id] = processed_answer
            
            # Quickly retrieve corresponding record by question_id
            example = id_to_example_map.get(question_id)
            pruned_graph_len = len(example["pruned_graph"])
            print("graph length",len(example["pruned_graph"]))
            print("generated list length",len(processed_answer))
            if len(processed_answer) != pruned_graph_len:
                # Discard this result
                continue
            if example:
                # Add prediction result to example
                example_with_prediction = {**example, "triple_unit_queries": processed_answer}
                batch_results.append(example_with_prediction)

            # When batch results reach filter_batch_size, perform one write
            if len(batch_results) >= filter_batch_size:
                if len(triple_trans_dataset["test"]) == 0:  # If triple_trans_dataset["test"] is empty
                    triple_trans_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
                else:
                    # Batch merge to existing Dataset
                    triple_trans_dataset["test"] = Dataset.from_dict({
                        key: triple_trans_dataset["test"][key] + [ex[key] for ex in batch_results]
                        for key in triple_trans_dataset["test"].column_names
                    })

                # Write to file
                triple_trans_dataset["test"].to_parquet(write_data_dir)
                batch_results = []  # Clear temporary storage

    # If there are remaining results, write to file
    if batch_results:
        if len(triple_trans_dataset["test"]) == 0:  # If triple_trans_dataset["test"] is empty
            triple_trans_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
        else:
            triple_trans_dataset["test"] = Dataset.from_dict({
                key: triple_trans_dataset["test"][key] + [ex[key] for ex in batch_results]
                for key in triple_trans_dataset["test"].column_names
            })
        triple_trans_dataset["test"].to_parquet(write_data_dir)
    
    print(f"Completed triple translation task for {dataset_name} dataset!")
    print("Sample after triple translation:",triple_trans_dataset["test"])
