import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from datasets import load_dataset
from llm.prompt_builder import *
from llm.llm_client import *
from tqdm import tqdm
from datetime import datetime
import pandas as pd
from datasets import Dataset, DatasetDict
import asyncio
from tqdm.asyncio import tqdm_asyncio
from llm.opensource_llm_client import *


def llm_qa_main(task="eog_qa",dataset_name=None,llm=None,openai_api_key=None,base_url=None,resume_path=None):
    return asyncio.run(llm_qa(task,dataset_name,llm,openai_api_key,base_url,resume_path))

async def llm_qa(task="eog_qa",dataset_path=None,llm=None,openai_api_key=None,base_url=None,resume_path=None):
    
    dataset_name = "cwq"
    if "cwq" in dataset_path:
        dataset_name = "cwq"
    elif "webqsp" in dataset_path:
        dataset_name = "webqsp"

    dataset = load_dataset("parquet", data_files={'test': dataset_path})

    mode = "direct_qa"

    # Print dataset information
    print(dataset)

    # Path to the file to be written
    # Get current time
    current_time = datetime.now()

    # Format time as string (e.g., "2023-10-10_14-30-00"), replace this with the name of the unfinished parquet from last time to continue
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Save file, final output file name is {dataset_name}_{llm}_{qa}
    if resume_path == None:
        write_data_dir = f"preprocess_datasets/qa_datasets/{dataset_name}_{llm}_{task}_{time_str}.parquet"
    else:
        write_data_dir = resume_path
    # Open the file, if it doesn't exist, create it
    # Ensure directory exists
    os.makedirs(os.path.dirname(write_data_dir), exist_ok=True)

    qa_dataset = None
    finished_id = []

    # Check if file exists, if not, create file
    if not os.path.exists(write_data_dir):
        # If file doesn't exist, create an empty DataFrame and save as parquet file
        df = pd.DataFrame()  # Create empty DataFrame
        df.to_parquet(write_data_dir)
        print(f"File doesn't exist, created new empty file: {write_data_dir}")
        # Initialize qa_dataset
        if task == "direct_qa":
            qa_dataset = DatasetDict({
                    "test": Dataset.from_dict({
                    "id": "",
                    "question": "",
                    "answer": [],
                    "q_entity": [],
                    "a_entity": [],
                    "graph": [],
                    "pruned_graph": [],
                    "choices": []
                })
            })
        else:
            qa_dataset = DatasetDict({
                    "test": Dataset.from_dict({
                    "id": "",
                    "question": "",
                    "answer": [],
                    "q_entity": [],
                    "a_entity": [],
                    "graph": [],
                    "choices": [],
                    "structural_enrich_triples":[],
                    "feature_enrich_triples":[],
                    "pruned_graph":[]
                })
            })
    else:
        print(f"File already exists: {write_data_dir}, will continue {task} task from this file")
        qa_dataset = load_dataset("parquet", data_files={'test': write_data_dir})
        # Check existing question_id
        for sample in qa_dataset["test"]:
            finished_id.append(sample["id"])



    ###############################################################################################################

    ###############################################################################################################
    # qa task
    ###############################################################################################################
    def assemble_prompt(question, subgraph):
        """
        Assemble and return a prompt string based on the given template and input question and subgraph.
        
        :param question: str, input question
        :param subgraph: list, list containing triples, each triple is a list of length 3
        :return: str, assembled prompt
        """
        # Initialize template
        template = "Input:\nquestion:\n{input_question}\ninformation:\n{triples}"
        
        # Convert each triple in subgraph to string
        triple_strings = ["({}, {}, {})".format(triple[0], triple[1], triple[2]) for triple in subgraph]
        
        # Join triple strings with newlines
        triples_section = "\n".join(triple_strings)
        
        # Assemble result using template
        prompt = template.format(input_question=question, triples=triples_section)
        
        return prompt
    
    def assemble_eog_prompt(question, subgraph,eog_triples):
        # Initialize template
        template = "Input:\nquestion:\n{input_question}\ninformation:\n{triples}"
        
        triple_strings1 = [
            "({})".format(", ".join(triple))  # Dynamically format as "(h, r, t, ...)" based on triple length
            for triple in eog_triples
            if len(triple) >= 3  # Ensure triple length is at least 3
        ]

        # Convert each triple in subgraph to string
        triple_strings2 = [
            "({})".format(", ".join(triple))  # Dynamically format as "(h, r, t, ...)" based on triple length
            for triple in subgraph
            if len(triple) >= 3  # Ensure triple length is at least 3
        ]
        # triple_strings2 = ["({}, {}, {})".format(triple[0], triple[1], triple[2]) for triple in subgraph]
        
        triple_strings = triple_strings1 + triple_strings2
        # Join triple strings with newlines
        triples_section = "\n".join(triple_strings)
        
        # Assemble result using template
        prompt = template.format(input_question=question, triples=triples_section)
        
        return prompt
    
    
    # Used to extract question and generate result_prompt
    def process_dataset(dataset):
        # Store results
        result_prompts = {}
        # Iterate through each row in the dataset
        for example in tqdm(dataset['test'],desc="Building qa prompts"):
            subgraph = []
            
            # Extract question field
            question_id = example['id']
            # Filter out completed samples
            if question_id in finished_id:
                continue
            question = example['question']
            # choose subgraph based on task
            if task == "eog_qa":
                for each_triple in example["feature_enrich_triples"]:
                    subgraph.append(each_triple)
                for each_triple in example["pruned_graph"]:
                    subgraph.append(each_triple)
            else:
                subgraph = example["graph"]

            if task == "eog_qa":
                each_prompt = assemble_eog_prompt(question,subgraph,example["structural_enrich_triples"])
            else:
                each_prompt = assemble_prompt(question,subgraph)
            # Call PromptBuilder, pass in question
            prompt_builder = PromptBuilder(each_prompt,mode)
            # print(each_prompt)
            # Get generated result_prompt
            result_prompts[question_id] = prompt_builder.build_prompt()
            # print("prompt content:",result_prompts[question_id])

        
        return result_prompts

    # Call function to process dataset
    qa_result_prompts = process_dataset(dataset)

    # Call openai
    llm_chat = llm_client(base_url=base_url,openai_api_keys=[openai_api_key],model=llm) 
    # Call open source small model
    # llm_chat = opensource_llm_client(model_path="/disk0/qiaoshuofei/PLMs/qwen2-7b-instruct/")

    def process_llm_answer(llm_answer=""):
        """
        Extract content after "Final answer:" in llm_answer, perform splitting processing, and return an answer list.
        If llm_answer is empty or doesn't contain "Final answer:", return empty list.
        
        :param llm_answer: str, string containing answer
        :return: list, processed answer list
        """
        # Check if llm_answer is empty string
        if not llm_answer:
            return []

        # Look for "Final answer:" or "Final Answer:"
        final_answer_keywords = ["Final answer:", "Final Answer:"]

        # Find the last occurrence of keywords
        last_occurrence_index = -1
        last_keyword = None
        for keyword in final_answer_keywords:
            index = llm_answer.rfind(keyword)  # Use rfind to find the last matching index
            if index > last_occurrence_index:  # Keep the last occurrence of "Final answer:" or "Final Answer:"
                last_occurrence_index = index
                last_keyword = keyword

        # Extract content after the last matching keyword
        if last_occurrence_index != -1:
            content = llm_answer[last_occurrence_index + len(last_keyword):].strip()
        else:
            content = llm_answer.strip()
        content = content.replace('```', '')
        content = content.replace('{thoughts & reason}', '').replace('{/thoughts & reason}', '')
        content = content.replace('{demonstrations}', '').replace('{/demonstrations}', '')
        content = content.replace('{/instruction}','').replace('{instruction}','')
        content = content.strip()
        # Split extracted content by <SEP> and remove possible spaces on both sides
        answers = [answer.strip() for answer in content.split("<SEP>")]
        
        # Return answer list
        return answers

    qa_result = {}

    # Convert dataset["test"] to a fast retrieval dictionary with id as key
    id_to_example_map = {example["id"]: example for example in dataset["test"]}

    # List for temporarily storing results
    batch_results = []
    filter_batch_size = 10  # Set batch size

    # Define async function to process single request
    async def process_single_query(llm_chat, question_id, each_qa_prompt):
        response = await llm_chat.response(each_qa_prompt)
        return question_id, response

    # Define async function to process multiple requests and handle results in completion order
    tasks = [
        process_single_query(llm_chat, question_id, each_qa_prompt)
        for question_id, each_qa_prompt in qa_result_prompts.items()
    ]

    # Use tqdm_asyncio to show progress bar
    with tqdm_asyncio(desc=f"Call {llm} for QA", total=len(qa_result_prompts)) as pbar:
        for future in asyncio.as_completed(tasks):
            result = await future  # Wait for a task to complete
            question_id, llm_answer = result  # Destructure from return value
            pbar.update(1)  # Update progress bar
            

    # Use tqdm to iterate and process
    # with tqdm(qa_result_prompts.items(), desc=f"{dataset_name} using {llm} QA") as pbar:
    #     for question_id, qa_prompt in pbar:
    #         # Call LLM to get answer
    #         llm_answer = llm_chat.response(qa_prompt, mode)
    #         all_tokens.append(count_tokens(llm_answer))

            # Process LLM's answer
            processed_answer = process_llm_answer(llm_answer=llm_answer)
            qa_result[question_id] = processed_answer

            # Quickly retrieve corresponding record by question_id
            example = id_to_example_map.get(question_id)

            # print("This sample's prompt is as follows:",qa_result_prompts[example["id"]])
            # print("This sample's LLM Answer is as follows:",llm_answer)
            
            if example:
                # Add prediction result to example
                example_with_prediction = {**example, "predictions": processed_answer}
                batch_results.append(example_with_prediction)


            # When batch results reach filter_batch_size, perform one write
            if len(batch_results) >= filter_batch_size:
                if len(qa_dataset["test"]) == 0:  # If qa_dataset["test"] is empty
                    qa_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
                else:
                    # Batch merge to existing Dataset
                    qa_dataset["test"] = Dataset.from_dict({
                        key: qa_dataset["test"][key] + [ex[key] for ex in batch_results]
                        for key in qa_dataset["test"].column_names
                    })

                # Write to file
                qa_dataset["test"].to_parquet(write_data_dir)
                batch_results = []  # Clear temporary storage
                

    # If there are remaining results, write to file
    if batch_results:
        if len(qa_dataset["test"]) == 0:  # If qa_dataset["test"] is empty
            qa_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
        else:
            qa_dataset["test"] = Dataset.from_dict({
                key: qa_dataset["test"][key] + [ex[key] for ex in batch_results]
                for key in qa_dataset["test"].column_names
            })
        qa_dataset["test"].to_parquet(write_data_dir)
