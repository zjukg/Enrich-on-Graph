
import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from utils.get_answer_entity_coverage import check_answer_in_graph_main
import torch
from datasets import load_dataset
import pandas as pd
from datasets import Dataset, DatasetDict
from datetime import datetime


def vanilla_prune(dataset_path=None,initial_pruning_llm="sentence-transformers",embedding_model_path=None,initial_pruning_topk=750,resume_path=None):
    # Load path
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

    # Format time as string (e.g., "2023-10-10_14-30-00"), replace this with the name of the unfinished parquet from last time to continue
    time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")

    # Save file, final output file name is {dataset_name}_{llm}_{qa}
    if resume_path == None:
        write_data_dir = f"preprocess_datasets/vanilla_pruning_datasets/{dataset_name}_{initial_pruning_llm}_{initial_pruning_topk}_vanilla_pruning_{time_str}.parquet"
    else:
        write_data_dir = resume_path

    # Ensure directory exists
    os.makedirs(os.path.dirname(write_data_dir), exist_ok=True)

    # Store results after question decomposition
    llm_pruning_dataset = None
    finished_id = []

    # Check if file exists, if not, create file
    if not os.path.exists(write_data_dir):
        # If file doesn't exist, create an empty DataFrame and save as parquet file
        df = pd.DataFrame()  # Create empty DataFrame
        df.to_parquet(write_data_dir)
        print(f"File doesn't exist, created new empty file: {write_data_dir}")
        # Initialize dataset
        llm_pruning_dataset = DatasetDict({
            "test": Dataset.from_dict({
                "id": "",
                "question": "",
                "user_queries":[],
                "answer": [],
                "q_entity": [],
                "a_entity": [], 
                "graph": [],
                "pruned_graph": [],
                "choices": []
            })
        })
    else:
        print(f"File already exists: {write_data_dir}, will continue llm pruning task from this file")
        llm_pruning_dataset = load_dataset("parquet", data_files={'test': write_data_dir})
        # Check existing question_id
        for sample in llm_pruning_dataset["test"]:
            finished_id.append(sample["id"])

    ###############################################################################################################

    # MPS > CUDA > CPU
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    # Initialize Sentence-BERT model
    model = SentenceTransformer(embedding_model_path,device=device)

    print(f"Coverage information for {dataset_name} dataset before pruning:")
    check_answer_in_graph_main(dataset=dataset,graph_name="graph")

    filtered_graph = {}
    subgraph_total_length = 0
    # Convert dataset["test"] to a fast retrieval dictionary with id as key
    id_to_example_map = {example["id"]: example for example in dataset["test"]}

    # List for temporarily storing results
    batch_results = []
    filter_batch_size = 30  # Set batch size

    for sample in tqdm(dataset["test"], desc="vanilla pruning"):
        corpus = []
        subgraph_total_length = subgraph_total_length + len(sample["graph"])
        # Skip if already pruned
        if sample["id"] in finished_id:
            continue
        if initial_pruning_topk >= len(sample["graph"]):
            filtered_graph[sample["id"]] = sample["graph"]
            question_id = sample["id"]
            # Quickly retrieve corresponding record by question_id
            example = id_to_example_map.get(question_id)
            if example:
                # Add prediction result to example
                example_with_prediction = {**example, "pruned_graph": sample["graph"]}
                batch_results.append(example_with_prediction)
        else:
            # Convert each element to string and store in corpus
            for sublist in sample["graph"]:
                corpus.append(" ".join(map(str, sublist)))
            # Embed the candidate set
            corpus_embeddings = model.encode(corpus)

            # Normalize embedding vectors (cosine similarity requires normalizing vectors to unit norm)
            corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

            # Build FAISS index (using inner product)
            embedding_dim = corpus_embeddings.shape[1]  # Dimension of embedding vectors
            index = faiss.IndexFlatIP(embedding_dim)    # Use inner product
            index.add(corpus_embeddings)               # Add embedding vectors to index

            # Calculate scores for each query in sample["user_queries"]
            total_scores = np.zeros(len(corpus))  # Store total score for each corpus

            all_queries = []
            all_queries.append(sample["question"])
            if "user_queries" in sample and sample["user_queries"] is not None:
                for query in sample["user_queries"]:
                    all_queries.append(query)
            for query in all_queries:
                query_embedding = model.encode([query])  # Embed query

                # Normalize query vector
                query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

                # Calculate similarity (score) with corpus
                distances, indices = index.search(query_embedding, len(corpus))

                for i, idx in enumerate(indices[0]):  # Iterate through returned indices
                    total_scores[idx] += distances[0][i]

                # Accumulate scores to total_scores
                # total_scores += distances[0]

            # Sort by total score in descending order and select top k, using llm_pruning_topk
            top_k_indices = np.argsort(total_scores)[::-1][:initial_pruning_topk]

            # Filtered triple list
            triple_filtered_graph = [sample["graph"][idx] for idx in top_k_indices]

            filtered_graph[sample["id"]] = triple_filtered_graph

            question_id = sample["id"]
            processed_answer = triple_filtered_graph
            # Quickly retrieve corresponding record by question_id
            example = id_to_example_map.get(question_id)
            if example:
                # Add prediction result to example
                example_with_prediction = {**example, "pruned_graph": processed_answer}
                batch_results.append(example_with_prediction)

            # When batch results reach filter_batch_size, perform one write
            if len(batch_results) >= filter_batch_size:
                if len(llm_pruning_dataset["test"]) == 0:  # If llm_pruning_dataset["test"] is empty
                    llm_pruning_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
                else:
                    # Batch merge to existing Dataset
                    llm_pruning_dataset["test"] = Dataset.from_dict({
                        key: llm_pruning_dataset["test"][key] + [ex[key] for ex in batch_results]
                        for key in llm_pruning_dataset["test"].column_names
                    })

                # Write to file
                llm_pruning_dataset["test"].to_parquet(write_data_dir)
                batch_results = []  # Clear temporary storage

    # If there are remaining results, write to file
    if batch_results:
        if len(llm_pruning_dataset["test"]) == 0:  # If llm_pruning_dataset["test"] is empty
            llm_pruning_dataset["test"] = Dataset.from_dict({key: [ex[key] for ex in batch_results] for key in batch_results[0]})
        else:
            llm_pruning_dataset["test"] = Dataset.from_dict({
                key: llm_pruning_dataset["test"][key] + [ex[key] for ex in batch_results]
                for key in llm_pruning_dataset["test"].column_names
            })
        llm_pruning_dataset["test"].to_parquet(write_data_dir)
    
    print(f"Completed vanilla pruning task for {dataset_name} dataset!")
    

    # Check the length of subgraph after coverage and answer coverage rate
    pruned_subgraph_total_length = 0
    for sample in dataset["test"]:
        pruned_subgraph_total_length = pruned_subgraph_total_length + len(sample["graph"])

    print("Total number of triples before vanilla pruning of dataset:",subgraph_total_length)
    print("Total number of triples after vanilla pruning:",pruned_subgraph_total_length)
    print("Ratio of the two:",pruned_subgraph_total_length/subgraph_total_length)

    ###############################################################################################################
