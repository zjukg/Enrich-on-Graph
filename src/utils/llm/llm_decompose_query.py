
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from datasets import load_dataset
from llm.prompt_builder import *
from llm.llm_client import *
from tqdm import tqdm
import pandas as pd
from datasets import Dataset, DatasetDict
from datetime import datetime
import asyncio
from tqdm.asyncio import tqdm_asyncio

def llm_decompose_query_main(dataset_path=None,llm=None,openai_api_key=None,base_url=None,resume_path=None):
    return asyncio.run(llm_decompose_query(dataset_path,llm,openai_api_key,base_url,resume_path))

async def llm_decompose_query(dataset_path=None,llm=None,openai_api_key=None,base_url=None,resume_path=None):

    data_dir = 'datasets'
    dataset_name = "cwq"
    if "cwq" in dataset_path:
        dataset_name = "cwq"
    elif "webqsp" in dataset_path:
        dataset_name = "webqsp"
    dataset = load_dataset("parquet", data_files={'test': f'{data_dir}/RoG-{dataset_name}/data/test*.parquet'})

    # Print dataset information
    print(dataset)

    # Get current time
    current_time = datetime.now()

    # Format time as string (e.g., "2023-10-10_14-30-00"), replace this with the unfinished parquet name to continue
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Save file, final output file name format: {dataset_name}_{llm}_{qa}
    if resume_path == None:
        write_data_dir = f"preprocess_datasets/question_decompose_datasets/{dataset_name}_{llm}_question_decompose_{time_str}.parquet"
    else:
        write_data_dir = resume_path

    # Save file, final output file name format: {dataset_name}_{llm}_{qa}
    # write_data_dir = f"preprocess_datasets/question_decompose_datasets/{dataset_name}_{llm}_question_decompose.parquet"
    
    # Open the file, create if it doesn't exist
    # Ensure directory exists
    os.makedirs(os.path.dirname(write_data_dir), exist_ok=True)

    # Store results after question decomposition
    decompose_question_dataset = None
    finished_id = []

    # Check if file exists, create if it doesn't
    if not os.path.exists(write_data_dir):
        # If file doesn't exist, create an empty DataFrame and save as parquet file
        df = pd.DataFrame()  # Create empty DataFrame
        df.to_parquet(write_data_dir)
        print(f"File doesn't exist, created new empty file: {write_data_dir}")
        # Initialize dataset
        decompose_question_dataset = DatasetDict({
            "test": Dataset.from_dict({
                "id": "",
                "question": "",
                "answer": [],
                "q_entity": [],
                "a_entity": [],
                "graph": [],
                "choices": [],
                "user_queries":[]
            })
        })
    else:
        print(f"File already exists: {write_data_dir}, will continue decomposing user query task from this file")
        decompose_question_dataset = load_dataset("parquet", data_files={'test': write_data_dir})
        # Check existing question_ids
        for sample in decompose_question_dataset["test"]:
            finished_id.append(sample["id"])

    # Used to extract question and generate result_prompt
    def process_dataset(dataset):
        # Store results
        result_prompts = {}
        
        # Iterate through each row in dataset
        for example in tqdm(dataset['test'],desc="Building question decompose prompts"):
            # Extract question field
            question_id = example['id']
            if question_id in finished_id:
                continue

            question = example['question']
            mode = "question_decompose"
            # Call PromptBuilder, pass in question
            prompt_builder = PromptBuilder(question,mode)
            
            # Get generated result_prompt
            result_prompts[question_id] = prompt_builder.build_prompt()

            # print(result_prompts)
            # return result_prompts

        return result_prompts

    # Call function to process dataset
    question_decompose_result_prompts = process_dataset(dataset)
    llm_chat = llm_client(base_url=base_url,openai_api_keys=[openai_api_key],model=llm) 

    # Response preprocessing
    def extract_questions(sub_queries):
        """
        Extract all questions from the given sub_queries text and return a formatted question list.

        :param sub_queries: Text containing tree structure of questions and sub-questions
        :return: Formatted question list
        """
        # Remove all "-", "--", "---" and newlines
        cleaned_text = sub_queries.replace("-", "").replace("--", "").replace("---", "").replace("\n", " ")
        
        # Split text to extract all questions
        questions = [question.strip() for question in cleaned_text.split("?") if question.strip()]
        
        return questions

    # Convert dataset["test"] to a fast lookup dictionary with id as key
    id_to_example_map = {example["id"]: example for example in dataset["test"]}

    # List for temporarily storing results
    batch_results = []
    filter_batch_size = 10  # Set batch size

    # Define async function to process single request
    async def process_single_query(llm_chat, question_id, each_prompt):
        response = await llm_chat.response(each_prompt)
        return question_id, response

    # Define async function to process multiple requests and handle results in completion order
    tasks = [
        process_single_query(llm_chat, question_id, each_prompt)
        for question_id, each_prompt in question_decompose_result_prompts.items()
    ]
    
    # Use tqdm_asyncio to show progress bar
    with tqdm_asyncio(desc=f"Call LLM for question decomposing", total=len(question_decompose_result_prompts)) as pbar:
        for future in asyncio.as_completed(tasks):
            result = await future  # Wait for a task to complete
            question_id, sub_queries = result  # Destructure from return value
            pbar.update(1)  # Update progress bar

    # with tqdm(question_decompose_result_prompts.items(), desc="Call LLM for question decomposing and Mapping to datasets") as pbar:
    #     for question_id, question in pbar:
    #         sub_queries = llm_chat.response(question)
    #         pbar.set_postfix(current_question_id=question_id)

            # Exception handling
            if sub_queries == None:
                question_list = list(id_to_example_map.get(question_id)["question"])
            else:
                # Preprocess response
                question_list = extract_questions(sub_queries)
            
            processed_answer = question_list

            # Fast lookup corresponding record by question_id
            example = id_to_example_map.get(question_id)
            if example:
                # Add prediction result to example
                example_with_prediction = {**example, "user_queries": processed_answer}
                batch_results.append(example_with_prediction)

            # When batch results reach filter_batch_size, perform one write
            if len(batch_results) >= filter_batch_size:
                if len(decompose_question_dataset["test"]) == 0:  # If decompose_question_dataset["test"] is empty
                    decompose_question_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
                else:
                    # Batch merge to existing Dataset
                    decompose_question_dataset["test"] = Dataset.from_dict({
                        key: decompose_question_dataset["test"][key] + [ex[key] for ex in batch_results]
                        for key in decompose_question_dataset["test"].column_names
                    })

                # Write to file
                decompose_question_dataset["test"].to_parquet(write_data_dir)
                batch_results = []  # Clear temporary storage

            # # Ensure each record initializes 'user_queries' column
            # if "user_queries" not in decompose_question_dataset["test"].column_names:
            #     decompose_question_dataset["test"] = decompose_question_dataset["test"].add_column("user_queries", [None] * len(decompose_question_dataset["test"]))


            # # Define function to modify data for specific id
            # def add_user_queries(example):
            #     if example["id"] == question_id:
            #         example["user_queries"] = question_list
            #     return example
            # decompose_question_dataset["test"] = decompose_question_dataset["test"].map(add_user_queries)

    # If there are remaining results, write to file
    if batch_results:
        if len(decompose_question_dataset["test"]) == 0:  # If decompose_question_dataset["test"] is empty
            decompose_question_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
        else:
            decompose_question_dataset["test"] = Dataset.from_dict({
                key: decompose_question_dataset["test"][key] + [ex[key] for ex in batch_results]
                for key in decompose_question_dataset["test"].column_names
            })
        decompose_question_dataset["test"].to_parquet(write_data_dir)
    
    print(f"Completed decomposing user query task for {dataset_name} dataset!")

    print("Dataset example after question decomposition:",decompose_question_dataset['test'])
