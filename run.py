import sys
import os
from src.utils.vanilla_prune import vanilla_prune
from src.utils.llm.llm_decompose_query import llm_decompose_query_main
from src.utils.llm.llm_trans_triple import llm_trans_triple_main
from src.utils.llm.llm_qa import llm_qa_main
from src.utils.evaluate_result import eval_result_main
from src.utils.llm.llm_filter_triples import llm_filter_triples_main
from src.utils.llm.llm_structural_enrich import llm_structural_enrich_main
from src.utils.llm.llm_feature_enrich import llm_feature_enrich_main
from src.utils.three_channel_pruning import three_channel_pruning

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
import argparse

def run(args):
    task = args.task
    if task == "vanilla_pruning":
        vanilla_prune(args.d,args.embedding_model,args.embedding_model_path,args.pruning_top_k,args.resume_path)
    elif task == "llm_pruning_three_channels":
        three_channel_pruning(args.d,args.embedding_model,args.embedding_model_path,args.pruning_top_k,args.resume_path)
    elif task == "query_decompose":
        llm_decompose_query_main(args.d,args.llm,args.openai_api_key,args.base_url,args.resume_path)
    elif task == "triple_trans":
        llm_trans_triple_main(args.d,args.llm,args.openai_api_key,args.base_url,args.resume_path)
    elif task == "filter_triples":
        llm_filter_triples_main(args.d,args.llm,args.openai_api_key,args.base_url,args.resume_path)
    elif task == "structral_enrich":
        llm_structural_enrich_main(args.d,args.llm,args.openai_api_key,args.base_url,args.resume_path)
    elif task == "feature_enrich":
        llm_feature_enrich_main(args.d,args.llm,args.openai_api_key,args.base_url,args.resume_path)
    elif task == "direct_qa" or task == "eog_qa":
        llm_qa_main(args.task,args.d,args.llm,args.openai_api_key,args.base_url,args.resume_path)
    elif task == "qa_evaluate":
        eval_result_main(args.d,args.cal_f1,args.qa_eval_top_k)
    else:
        print("error task name!")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    # datasets
    argparser.add_argument('-d', type=str,help="dataset path", default="cwq")
    argparser.add_argument('--resume_path',type=str,help="resume file path",default=None)

    # pruning settings
    argparser.add_argument("--embedding_model",type=str,help="embedding model",choices=["sentence-transformers"])
    argparser.add_argument('--embedding_model_path',type=str,help="embedding model path",default=None)
    argparser.add_argument('--pruning_top_k',help="pruning topk",type=int, default=750)

    # eval settings
    argparser.add_argument('--qa_eval_top_k',help="qa eval topk",type=int, default=-1)
    argparser.add_argument('--cal_f1', action='store_true', help="qa eval use f1",default=True)
    
    # llm api settings
    argparser.add_argument("--llm",type=str,help="llm name")
    argparser.add_argument('--openai_api_key',type=str,help="openai api key",default=None)
    argparser.add_argument('--base_url',type=str,help="base url for openai api",default=None)
    # task
    argparser.add_argument("--task",type=str,help="task name",choices=["vanilla_pruning","llm_pruning_three_channels","query_decompose","triple_trans","filter_triples","structral_enrich","feature_enrich","direct_qa","eog_qa","qa_evaluate"])
    

    args = argparser.parse_args()
    run(args=args)
