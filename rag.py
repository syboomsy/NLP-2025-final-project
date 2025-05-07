import os
from typing import List, Dict
import time
import faiss
import numpy as np
import random
from tqdm import tqdm
from ipdb import set_trace
import pandas as pd
import langchain

langchain.verbose = False
langchain.debug = False
langchain.llm_cache = False

from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from concurrent.futures import ThreadPoolExecutor, Future
from sentence_transformers import SentenceTransformer

import json
import jsonlines
import torch

SEED = 3
random.seed(SEED)
VEC_INDEX_PATH = r"index\medQA_vecidx_colab.faiss"
MAX_WORKERS = 16
TESTSET_RATIO = 1.0
TOPK = 0.0
IS_TESTING = False
medQA_train_path = r"data\medQA\phrases_no_exclude_train.jsonl"
medQA_test_path = r"data\medQA\phrases_no_exclude_test.jsonl"
logiQA_train_path = r"data\logiQA\logiqa_train.jsonl"
logiQA_test_path = r"data\logiQA\logiqa_test.jsonl"

device = "cuda" if torch.cuda.is_available() else "cpu"
if not IS_TESTING:
    embed_model = SentenceTransformer('all-MiniLM-L6-v2',device=device)
else:
    embed_model = None
dim = 384  # e.g., 384 for MiniLM
index = faiss.IndexFlatL2(dim)  # L2 = Euclidean distance

train_corpus:List[Dict] = []
test_corpus:List[Dict] = []

os.environ["DEEPSEEK_API_KEY"] = "sk-ab64082b8e0d4c03824fa017cee237be"
os.environ["DEEPSEEK_BASE_URL"] = "https://api.deepseek.com"
prompt_templ = ChatPromptTemplate.from_messages(
    [
        ("system", 
         
        #  """
        # You are an expert in medical knowledge QA, 
        # now you need to answer some questions. 
        # You can refer to example QAs if provided(not none). 
        # Make sure to wrap your answer-index in [].
        #  """
        """
        You are an expert in doing QA exam, now you need to answer some questions.
        You can refer to any example, hints and facts if provided.
        Make sure to ONLY wrap your final answer-index in [], and dont wrap any other answer-index you excluded.
        """
        # """
        # You are an expert in medical knowledge, now you need to do some tasks.Keep conciseness
        #  and preciseness.
        # """

         ),
        ("user", "{query}")
    ]
)

fact_gen_templ = """
    {qa_example}\n For this QA example, you need to give netural fact info for each
    option according to question context, so that later user can infer out the 
    correct answer from these facts. The template of facts is [option1_name:fact| option2_name:fact|...],e.g: [Ampicillin:High resistance, not first-line. | Ceftriaxone:Overkill (for pyelonephritis/severe UTI).]
"""

fact_test_templ = """
    Question:\n{question}\n\n Option:\n{option} \n\n Hint:\n{hint} \n\n Answer:
"""

full_qa_templ = "nQuestion:\n{question}\n\n Option:\n{option}\n\n Answer:\n{answer}"

prepare_templ = "Example:\n{example}\n\nQuestion:\n{question}\n\n Option:\n{option}"

example_templ = "ExmapleQuestion:\n{question}\n\n Option:\n{option}\n\n Answer:[{answer}]"

dpsk_model = ChatDeepSeek(model="deepseek-chat")

chain = prompt_templ | dpsk_model | StrOutputParser()

def process_item(item):
    global chain
    item["model_answer"] =chain.invoke({"query": item["query"]})
    return item

def process_nohinted(item):
    global chain, fact_test_templ
    query = fact_test_templ.format(
       question=item["question"],
        option=get_option_part(item),
        hint="None"
    )
    item["model_answer"] =chain.invoke({"query": query})
    item["hinted_gen_query"] = query
    return item
    pass

def process_hinted(item):
    global chain, fact_test_templ
    query = fact_test_templ.format(
       question=item["question"],
        option=get_option_part(item),
        hint=item["model_answer"]
    )
    item["model_answer"] =chain.invoke({"query": query})
    item["hinted_gen_query"] = query
    return item

def process_facto_item(item):
    global chain
    query = fact_gen_templ.format(
        qa_example = full_qa_templ.format(
            question=item["question"],
            option=get_option_part_noindex(item),
            answer=item["answer"]
        )
    )
    item["model_answer"] =chain.invoke({"query": query})
    item["fact_gen_query"] = query
    return item

def get_option_part_noindex(each:Dict):
    option_part = ""
    try:
        for key, val in each["options"].items():
            option_part += val + ";\n"
    except:
        set_trace()
    return option_part

def get_option_part(each:Dict):
    option_part = ""
    try:
        for key, val in each["options"].items():
            option_part += key + ":" + val + ";\n"
    except:
        set_trace()
    return option_part

def prepare_question(corpus:List[Dict], sample_ratio:float=TESTSET_RATIO, example_k:int = TOPK):
    assert sample_ratio >0 and sample_ratio <= 1
    assert example_k >= 0
    
    # corpus = random.sample(corpus, int(sample_ratio * len(corpus)))
    random.shuffle(corpus)
    corpus = corpus[: int(sample_ratio * len(corpus))]

    prepared_corpus:List[Dict] = []
    global prepare_templ, train_corpus, test_corpus, example_templ
    
    for each in tqdm(corpus, desc="question preparing..."):
        
        example_part = "none."
        if example_k > 0:
            example_part = ""
            example_dicts = index_retrieve(each["question"], train_corpus, example_k)
                
            for each_retri in example_dicts:
                assert each_retri["question"] != each["question"]
                example_part += example_templ.format(
                    question=each_retri["question"],
                    option=get_option_part(each_retri),
                    answer=each_retri["answer_idx"]
                ) + "\n\n"
            
        question = prepare_templ.format(
            example = example_part,
            question = each["question"],
            option = get_option_part(each),
        )
        # set_trace()
        prepared_corpus.append(
            {
                "query": question, 
                "answer":each["answer"], 
                "answer_idx":each["answer_idx"],

            }
        )
    return prepared_corpus


def read_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # skip empty lines
                data.append(json.loads(line))
    return data

def save_json(j_dicts, path:str):
    with open(path, "w", encoding="utf-8") as f:
        for item in j_dicts:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def read_parquet(path:str):
    return pd.read_parquet(path)

def load_index(index_path):
    index = faiss.read_index(index_path)
    if index:
        print(f"Index has {index.ntotal} vectors")
    return index

def save_index(index, index_path):
    global VEC_INDEX_PATH
    if not os.path.exists("index"):
        os.mkdir("index")
    faiss.write_index(index, VEC_INDEX_PATH)

def build_index(dict_corpus:List[Dict]):
    corpus = [x["question"] for x in dict_corpus]
    start_time = time.perf_counter()
    # Step 1: Initialize an embedding model (you can use any)
    global embed_model, index
    loaded_time = time.perf_counter()
    # Step 3: Get embeddings (shape: [num_texts, embedding_dim])
    embeddings = embed_model.encode(corpus, convert_to_numpy=True)
    encoded_time = time.perf_counter()
    # Step 4: Create FAISS index
    
    print("load time ", loaded_time - start_time)
    print("encoded time ", encoded_time - loaded_time)
    # # Step 5: Add vectors to the index
    # index.add(embeddings) 
    return embeddings

def concurrent_build_index(corpus:List[str]):
    global MAX_WORKERS
    assert MAX_WORKERS <= len(corpus)
    global VEC_INDEX_PATH, index
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = []
            slice_size = len(corpus) // MAX_WORKERS
            for i in range(MAX_WORKERS):
                if i < MAX_WORKERS - 1:
                    subcorpus = corpus[i* slice_size : (i+1)*slice_size]
                else:
                    subcorpus = corpus[i*slice_size:]
                futures.append(executor.submit(build_index, subcorpus))
           
            # 使用tqdm显示进度
            for future in tqdm(
                futures, total=len(futures), desc="处理项目中"
            ):
                assert isinstance(future, Future)
                index.add(future.result())
    save_index(index, VEC_INDEX_PATH)

def index_retrieve(a_str:str, corpus:List[Dict], topk:int = 2):
    global embed_model, index
    key_embed = embed_model.encode([a_str], convert_to_numpy=True)
    distances, indices = index.search(key_embed, topk)
    distances = distances.squeeze().tolist()
    indices = indices.squeeze().tolist()
    ret_dicts = []
    # print(distances)
    if not isinstance(indices, list):
        indices = [indices]
    for i in indices:
        ret_dicts.append(corpus[i])
    return ret_dicts


def get_answers(items_to_process, output_path, item_func, max_workers=8):
    with jsonlines.open(
        output_path, "w"
    ) as writer:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(item_func, item): item for item in items_to_process
            }
            
            # 使用tqdm显示进度
            for future in tqdm(
                futures, total=len(items_to_process), desc="处理项目中"
            ):
                try:
                    writer.write(future.result())
                except Exception as e:
                    print(
                        f"处理项目时出错: {futures[future]['query']}. 错误: {e}"
                    )

def evaluation(output_dicts:List[Dict]):
    # assert len(output_dicts)
    # unit:float = 100 / len(output_dicts) 
    valid_count = 0
    correct_count = 0
    valid_answer_idxs = "ABCDEFGHI"
    for each in output_dicts:
        label = each["answer_idx"]
        pred:str = each["model_answer"]
        if not "[" in pred:
            continue
        pred_idx = pred.find("[")
        pred_label = pred[pred_idx+1]
        if not pred_label in valid_answer_idxs:
            continue
        correct_count += label in pred_label
        valid_count += 1
    test_num = len(output_dicts)
    return {
        "valid_cnt": valid_count,
        "invalid_cnt": test_num - valid_count,
        "correct_cnt": correct_count,
        "test_num": test_num,
        "score": (correct_count / test_num) * 100,
    }

def prepare_QASC_data(qasc_df:pd.DataFrame):
    # Step 1: Drop rows with null or blank answerKey
    df_clean = qasc_df[qasc_df['answerKey'].notna() & (qasc_df['answerKey'].str.strip() != '')]

    # Step 2: Select and rename the desired columns
    columns = ['formatted_question', 'fact1', 'fact2', 'answerKey']
    jsonl_data = df_clean[columns].to_dict(orient='records')
    # save_json(jsonl_data, saved_path)
    return jsonl_data


def build_fact_corpus(list_of_dicts:List):
    fact_corpus = {}
    def process_one_qa(qa_dict):
        facts_str = qa_dict["model_answer"].strip()
        assert isinstance(facts_str, str)
        start = facts_str.find("[")
        end = facts_str.find("]")
        if start  == -1 or end == -1:
            return {}
        facts_str = facts_str[start+1: end-1].strip()
        facts = facts_str.split("|")
        for each in facts:
            split_index = each.find(":")
            # try:
            #     key, val = each.split(":")
            # except:
            #     set_trace()
            key = each[:split_index]
            val = each[split_index+1:]
            key = key.strip()
            val = val.strip()
            if fact_corpus.get(key, None) is None:
                fact_corpus[key] = [val]
            else:
                fact_corpus[key].append(val)
    for each_dict in tqdm(list_of_dicts, desc="building facto corpus..."):
        process_one_qa(each_dict)
    return fact_corpus


if __name__ == "__main__":
    # VEC_INDEX_PATH = r"index\logiQA_vecidx_colab.faiss"
    # index = load_index(VEC_INDEX_PATH)
    # train_corpus = read_jsonl(logiQA_train_path)
    # test_corpus = read_jsonl(logiQA_test_path)

    VEC_INDEX_PATH = r"index\medQA_vecidx_colab.faiss"
    index = load_index(VEC_INDEX_PATH)
    train_corpus = read_jsonl(medQA_train_path)
    test_corpus = read_jsonl(medQA_test_path)
    test_corpus = test_corpus[:100]
    # set_trace()
    # test_str = test_corpus[3]["question"]
    # print(test_str, end="\n\n")
    # ret_dicts = index_retrieve(test_str, train_corpus, 3)
    # for each in ret_dicts:
    #     print(each["question"],end="\n\n")

    # TESTSET_RATIO = 0.15
    # TOPK = 1
    # prepared_qa_saved_path = "data\\prepare_medqa_test{}_topk{}.jsonl".format(TESTSET_RATIO, TOPK)
    # prepared_dicts = prepare_question(test_corpus,sample_ratio=TESTSET_RATIO, example_k=TOPK)
    # save_json(prepared_dicts, prepared_qa_saved_path)

    IS_TESTING = True

    # test_dicts = read_jsonl(r"data\prepare_medqa_test0.15_topk0.jsonl")
    # output_path = r"data\output\output_medqa_test0.15_topk0.jsonl"
    output_path =  r"data\output\output_medqa_testfacto100.jsonl"
    # output_path_pred =  r"data\output\output_medqa_testfacto100_pred.jsonl"
    output_path_pred =  r"data\output\output_medqa_testfacto100_prednohint.jsonl"
    get_answers(
        read_jsonl(output_path),
        output_path_pred,
        # process_hinted,
        process_nohinted,
        2
    )
    print(
        evaluation(read_jsonl(output_path_pred) )
    )

    # output_medqa_testfacto = read_jsonl(output_path)
    # fact_corpus = build_fact_corpus(output_medqa_testfacto)
    # set_trace()

    #############################
   
    # set_trace()
    pass