# This script is used to filter triples that are strongly related to the question
# Input:
# dataset, where dataset needs to contain graph, user_queries, and triple_unit_queries fields, involving fields
# graph subgraph, is a list
# user_queries, is a list
# triple_unit_queries, query corresponding to each triple, is a list
# Output:
# Output the index of triples related to this question

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

def llm_filter_triples_main(dataset_path=None,llm=None,openai_api_key=None,base_url=None,resume_path=None):
    return asyncio.run(llm_filter_triples(dataset_path,llm,openai_api_key,base_url,resume_path))


async def llm_filter_triples(dataset_path=None,llm=None,openai_api_key=None,base_url=None,resume_path=None):
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
        write_data_dir = f"preprocess_datasets/filter_triple_datasets/{dataset_name}_{llm}_filter_triple_{time_str}.parquet"
    else:
        write_data_dir = resume_path

    # Save file, final output file name is {dataset_name}_{llm}_{qa}
    # write_data_dir = f"preprocess_datasets/filter_triple_datasets/{dataset_name}_{llm}_filter_triple.parquet"
    
    # Open the file, if it doesn't exist, create it
    # Ensure directory exists
    os.makedirs(os.path.dirname(write_data_dir), exist_ok=True)

    # Store results after question decomposition
    filter_triple_dataset = None
    finished_id = []

    # Check if file exists, if not, create file
    if not os.path.exists(write_data_dir):
        # If file doesn't exist, create an empty DataFrame and save as parquet file
        df = pd.DataFrame()  # Create empty DataFrame
        df.to_parquet(write_data_dir)
        print(f"File doesn't exist, created new empty file: {write_data_dir}")
        # Initialize dataset
        filter_triple_dataset = DatasetDict({
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
                "triple_unit_queries":[],
                "filter_triples":[]
            })
        })
    else:
        print(f"File already exists: {write_data_dir}, will continue LLM triple filtering task from this file")
        filter_triple_dataset = load_dataset("parquet", data_files={'test': write_data_dir})
        # Check existing question_id
        for sample in filter_triple_dataset["test"]:
            finished_id.append(sample["id"])

    ###############################################################################################################
    # For each question in the dataset, filter out the index of triples related to it
    ###############################################################################################################
    mode = 'filter_triples'
    filter_triples_prompt = {}

    # Build prompt
    with tqdm(dataset['test'], desc="Building filter triples prompts") as pbar:
        for each_sample in pbar:
            pbar.set_postfix(current_question=each_sample["question"])
            # Skip existing samples
            if each_sample["id"] in finished_id:
                continue
            user_queries = each_sample["user_queries"]
            sub_graph = each_sample["pruned_graph"]
            triple_unit_queries = each_sample["triple_unit_queries"]
            
            # # The following is test data
            # user_queries = [
            #     "What is the location that appointed Michelle Bachelet to a governmental position?",
            #     "Who is Michelle Bachelet?",
            #     "What governmental position was Michelle Bachelet appointed to?",
            #     "Where was Michelle Bachelet appointed to this position?",
            #     "What language is spoken in this location?"
            # ]

            # sub_graph = [
            #     ["Michelle Bachelet", "people.person.nationality", "Chile"],
            #     ["Chile", "language.human_language.countries_spoken_in", "Spanish Language"]
            # ]

            # triple_unit_queries = [
            #     ["What is Michelle Bachelet's nationality?", "Which people have Chilean nationality?"],
            #     ["What language is spoken in Chile?", "Which countries speak the Spanish Language?"]
            # ]

            def assemble_prompt(user_queries, sub_graph, triple_unit_queries):
                # Initialize template
                template = "Input:\nuser unit queries:\n"
                
                # Fill user_queries
                user_queries_section = "\n".join(user_queries)
                template += user_queries_section + "\ntriple unit queries:\n"
                
                # Fill triple_unit_queries
                triple_unit_queries_section = ""
                for idx, triple in enumerate(sub_graph):
                    triple_str = f"({triple[0]},{triple[1]},{triple[2]})"
                    if triple_unit_queries[idx]:  # If this triple has corresponding query
                        queries = "<SEP>".join(triple_unit_queries[idx])
                        triple_unit_queries_section += f"{triple_str}<SEP>{queries}\n"
                    else:  # If this triple has no corresponding query
                        triple_unit_queries_section += f"{triple_str}<SEP>\n"
                
                # Merge to generate final result
                template += triple_unit_queries_section.strip()
                return template

            each_input = assemble_prompt(user_queries,sub_graph,triple_unit_queries)
            
            # Call PromptBuilder, pass in question
            prompt_builder = PromptBuilder(each_input,mode)
            filter_triples_prompt[each_sample["id"]] = prompt_builder.build_prompt()

    # Call llm to filter triples
    llm_chat = llm_client(base_url=base_url,openai_api_keys=[openai_api_key],model=llm) 
    
    # Convert dataset["test"] to a fast retrieval dictionary with id as key
    id_to_example_map = {example["id"]: example for example in dataset["test"]}

    # List for temporarily storing results
    batch_results = []
    filter_batch_size = 5  # Set batch size

    # Define async function to process single request
    async def process_single_query(llm_chat, question_id, each_triples_trans_prompt):
        response = await llm_chat.response(each_triples_trans_prompt)
        return question_id, response

    # Define async function to process multiple requests and handle results in completion order
    tasks = [
        process_single_query(llm_chat, question_id, each_filter_triples_prompt)
        for question_id, each_filter_triples_prompt in filter_triples_prompt.items()
    ]
    
    # Use tqdm_asyncio to show progress bar
    with tqdm_asyncio(desc=f"Call {llm} for filter triples", total=len(filter_triples_prompt)) as pbar:
        for future in asyncio.as_completed(tasks):
            result = await future  # Wait for a task to complete
            question_id, response = result  # Destructure from return value
            pbar.update(1)  # Update progress bar

    # with tqdm(filter_triples_prompt.items(),desc=f"Call {llm} for filter triples") as pbar:
    #     for question_id,each_filetr_triples_prompt in pbar:
    #         pbar.set_postfix(current_question=question_id)
    #         response = llm_chat.response(each_filetr_triples_prompt,mode)

            ###############################################################################################################
            # Response format
            # question<SEP>None
            # question<SEP>triple1<SEP>triple2
            ###############################################################################################################

            # Filter content after Final output:

            def extract_and_process_response(response):
                # 1. Extract content after "Final output:"
                final_output_match = re.search(r"Final output:\n(.*)", response, re.DOTALL)
                if not final_output_match:
                    content = response.strip()
                else:
                    content = final_output_match.group(1).strip()
                    
                content = content.replace('```', '')
                content = content.replace('{thought}', '').replace('{/thought}', '')
                content = content.replace('{demonstrations}', '').replace('{/demonstrations}', '')

                # 2. Initialize result dictionary
                triple_to_questions = defaultdict(list)
                
                # 3. Parse content
                for line in content.splitlines():
                    line = line.strip()
                    if line:
                        # question part
                        question, triples_str = line.split("<SEP>", 1)
                        question = question.strip()
                        
                        # triples part
                        triples = []
                        if triples_str.strip() != "None":
                            triple_strings = triples_str.split('<SEP>')
                            for triple in triple_strings:
                                # Remove leading/trailing parentheses and spaces, then split by comma and clean each element
                                triple = triple.strip().strip('()')
                                elements = [e.strip() for e in triple.split(',')]
                                triples.append(elements)
                        # Add question to each triple's list
                        if triples:
                            for triple in triples:
                                triple_to_questions[tuple(triple)].append(question)
                        else:
                            # If no triples, add to empty list corresponding key
                            triple_to_questions[()].append(question)
                
                # 4. Convert to regular dict and return
                return dict(triple_to_questions)
            
            # Test code
            # response = """
            # Final output:
            # Who was the champion of the 1931 World Series Championship?<SEP>(1931 World Series, sports.sports_team.championships, St. Louis Cardinals)
            # What is the World Series Championship?<SEP>None
            # Who won the World Series in 1931?<SEP>(1931 World Series, sports.sports_team.championships, St. Louis Cardinals)
            # Where does this champion team play?<SEP>(St. Louis Cardinals, plays at, Busch Stadium)<SEP>(St. Louis Cardinals, has arena, Busch Stadium)
            # What is the name of the stadium associated with this team?<SEP>(St. Louis Cardinals, sports.sports_team.arena_stadium, Busch Stadium)<SEP>(St. Louis Cardinals, has arena, Busch Stadium)
            # Was this stadium their home stadium in 1931?<SEP>(St. Louis Cardinals, sports.sports_team.arena_stadium, Roger Dean Stadium)<SEP>(St. Louis Cardinals, home ground of, Busch Stadium)<SEP>(St. Louis Cardinals, home ground of, Roger Dean Stadium)
            # """

            # Data structure: key(tuple[s,p,o])-value(list[question]), s,p,o and question are all str, list[question] may be empty []
            triple_question_dict = extract_and_process_response(response)

            # Match keys in triple_question_dict with pruned_graph field in dataset, find matching index, then add question list to corresponding index in filter_triples
            # Function write_triple_question
            def write_triple_question(triple_question_dict, example):
                """
                This function matches triples in triple_question_dict with triples in example["pruned_graph"],
                and generates match_index and no_match_dict.

                :param triple_question_dict: dict, data structure is key(tuple[s, p, o]) - value(list[question])
                :param example: dict, contains "pruned_graph" key, its value is list, each element is a list of length 3 representing a triple
                :return: match_index: list, same length as example["pruned_graph"]
                        no_match_dict: dict, key(tuple[s, p, o]) - value(list[question])
                """
                # Get pruned_graph
                pruned_graph = example["pruned_graph"]

                # Initialize match_index, length equal to pruned_graph, initial values all None
                match_index = [None] * len(pruned_graph)

                # Initialize no_match_dict
                no_match_dict = {}

                # Iterate through each triple and its corresponding question list in triple_question_dict
                for triple, questions in triple_question_dict.items():
                    # Initialize match flag
                    matched = False

                    # Iterate through pruned_graph and its index, look for matching triples
                    for index, graph_triple in enumerate(pruned_graph):
                        # If triple matches
                        if triple == tuple(graph_triple):  # Convert graph_triple to tuple for comparison
                            # Record questions at corresponding position in match_index
                            match_index[index] = questions
                            matched = True
                            break  # Stop matching current triple if successful

                    # If no triple matched, add to no_match_dict
                    if not matched:
                        no_match_dict[triple] = questions

                # Return match_index and no_match_dict
                return match_index, no_match_dict

            # Quickly retrieve corresponding record by question_id
            example = id_to_example_map.get(question_id)
            no_match_dict = {}
            # match_index is a list, each element is a triple, each triple is a list of length 3,
            processed_answer,no_match_dict = write_triple_question(triple_question_dict,example)
            
            ###############################################################################################################
            # Assume it can always match, no exception handling for now
            if len(no_match_dict) > 0:
                if len(no_match_dict) == 1:
                    for key,value in no_match_dict.items():
                        if key == ():
                            pass
                        else:
                            print("There are unmatched triples, please note!!!")
                else:
                    print("There are unmatched triples, please note!!!")
                pass
            ###############################################################################################################

            
            if example:
                # Add prediction result to example
                example_with_prediction = {**example, "filter_triples": processed_answer}
                batch_results.append(example_with_prediction)
            # When batch results reach filter_batch_size, perform one write
            if len(batch_results) >= filter_batch_size:
                if len(filter_triple_dataset["test"]) == 0:  # If filter_triple_dataset["test"] is empty
                    filter_triple_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
                else:
                    # Batch merge to existing Dataset
                    filter_triple_dataset["test"] = Dataset.from_dict({
                        key: filter_triple_dataset["test"][key] + [ex[key] for ex in batch_results]
                        for key in filter_triple_dataset["test"].column_names
                    })

                # Write to file
                filter_triple_dataset["test"].to_parquet(write_data_dir)
                batch_results = []  # Clear temporary storage
                
    # If there are remaining results, write to file
    if batch_results:
        if len(filter_triple_dataset["test"]) == 0:  # If filter_triple_dataset["test"] is empty
            filter_triple_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
        else:
            filter_triple_dataset["test"] = Dataset.from_dict({
                key: filter_triple_dataset["test"][key] + [ex[key] for ex in batch_results]
                for key in filter_triple_dataset["test"].column_names
            })
        filter_triple_dataset["test"].to_parquet(write_data_dir)
    
    print(f"Completed LLM triple filtering task for {dataset_name} dataset!")
    print("Sample after triple translation:",filter_triple_dataset["test"])
