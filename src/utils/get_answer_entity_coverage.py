
# This script is used to get the answer coverage rate of graph in the current dataset, i.e., whether answer entities are in some triples of the graph
# Input: dataset
from datasets import load_dataset

def check_answer_in_graph(sample,graph_name="graph"):
    # Get answer list and graph
    answers = sample["answer"]  # Assume sample["answer"] is a list
    subgraph = []
    if isinstance(graph_name,list):
        for name in graph_name:
            # Here we don't limit triples to length 3
            for triple in sample[name]:
                subgraph.append(triple)
            # subgraph.append(triple for triple in sample[name])
    else:
        subgraph = sample[graph_name]

    # Counter to count the number of existing answers
    match_count = 0

    # Iterate through each answer in answers
    for answer in answers:
        # Iterate through each triple in graph
        for triplet in subgraph:
            # Check if current answer exists in current triple
            if answer in triplet:
                match_count += 1
                break  # Answer matched, no need to check other triples
    if match_count == 0:
        # print(sample["id"])
        pass
    return match_count

def check_answer_in_graph_main(dataset,graph_name="graph"):
    exist_number = 0
    total_number = 0
    for sample in dataset["test"]:
        is_exist = check_answer_in_graph(sample,graph_name=graph_name)
        exist_number = exist_number + is_exist
        total_number = total_number + len(sample["answer"])
    
    coverage = exist_number / total_number
    print("Total number of answers in dataset samples:",total_number)
    print("Number of covered sample answers:",exist_number)
    print("Coverage rate:",coverage)


if __name__ == "__main__":
    # Build queries to be retrieved
    # Load path
    # cwq dataset
    # webqsp dataset
    data_dir = 'datasets/RoG-webqsp/data/'

    # Use wildcards to match all parquet files starting with "test"
    # dataset = load_dataset("parquet", data_files={'test': f'{data_dir}test*.parquet'})

    dataset = load_dataset(
        "parquet", 
        data_files={'test': 'preprocess_datasets/temp_datasets/cwq_gpt4o-mini_sentence-transformers_750_llm_filter_gt_triples_04_11.parquet'}
    )
    # Access test set
    test_dataset = dataset['test']

    print("Dataset field overview:",test_dataset)
    exist_number = 0
    total_number = 0

    for sample in test_dataset:
        # is_exist = check_answer_in_graph(sample,graph_name=["gt_triples","graph"])
        is_exist = check_answer_in_graph(sample,graph_name="gt_triples")
        exist_number = exist_number + is_exist
        total_number = total_number + len(sample["answer"])

    coverage = exist_number / total_number
    print("Total number of answers in dataset samples:",total_number)
    print("Number of covered sample answers:",exist_number)
    print("Coverage rate:",coverage)
