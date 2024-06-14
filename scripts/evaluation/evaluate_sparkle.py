import os
import random
import argparse
import pickle
import json
import re
import sys
from tqdm import tqdm
from SPARQLWrapper.SPARQLExceptions import QueryBadFormed, EndPointInternalError
from sparkle.dataset.webqsp import WebQSP
from sparkle.dataset.lcquad1 import LCQuAD1
from sparkle.dataset.simplequestions import SimpleQuestions

# arguments
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        default="lcquad1",
        help="Dataset specification",
        choices=["simplequestions", "lcquad1", "webqsp"]
    )
    parser.add_argument(
        "--predict_file",
        type=str,
        default="data/output/dataset/facebook/bart-large/lr5e-05_batch4_beam4/generated/generate_batch8_beam10.json",
        help="A generated file which includes draft candidate sparqls",
    )
    parser.add_argument(
        "--dataset_preprocess_dir",
        type=str,
        default="data/preprocessed/dataset",
        help="A generated file which includes draft candidate sparqls",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="https://endpoint",
        help="An endpoint to query",
    )
    parser.add_argument(
        "--cold_start",
        action="store_true"
    )
    parser.add_argument(
        "--warm_start",
        action="store_true"
    )

    args = parser.parse_args()
    return args

def get_f1(pred: set, gt: set):
    precision = len(pred & gt) / len(pred)
    recall = len(pred & gt) / len(gt)

    if precision == 0 and recall == 0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)

    return f1

# https://github1s.com/microsoft/KC/blob/main/papers/TIARA/src/utils/statistics/webqsp_legacy_eval.py
def get_h1(pred: set, gt: set):
    pred = list(pred)
    num_random = 100
    random_hit = 0
    for _ in range(num_random):
        random_ans = random.choice(pred)
        if random_ans in gt:
            random_hit += 1
    random_hit /= num_random

    return random_hit

def main():
    random.seed(2023)
    args = parse_arguments()

    model_output_dir = os.path.dirname(os.path.dirname(args.predict_file))
    result_dir = os.path.join(model_output_dir, "results")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)
    
    log_file = os.path.basename(args.predict_file).replace(".json", ".txt")
    if args.cold_start:
        log_file = "cold_" + log_file
    elif args.warm_start:
        log_file = "warm_" + log_file
    
    log_path = os.path.join(result_dir, log_file)
    sys.stdout = open(log_path, 'w')


    # dataset select
    if args.dataset == "webqsp":
        dataset = WebQSP
    elif args.dataset == "lcquad1":
        dataset = LCQuAD1
    elif args.dataset == "simplequestions":
        dataset = SimpleQuestions
    
    # load kg related files
    with open(os.path.join(args.dataset_preprocess_dir, "entity_relation_token_dict.pkl"), "rb") as f:
        entity_token_dict, relation_token_dict = pickle.load(f)

    token_entity_dict = {v["decoded"]: k for k, v in entity_token_dict.items()}
    token_relation_dict = {
        v["decoded"]: k for k, v in relation_token_dict.items()
    }

    with open(args.predict_file, "r") as f:
        predict_data = json.load(f)
    predict_data = predict_data["result"]

    # one question - multiple (sparql - answer) sets (e.g webqsp)
    if all(isinstance(a, list) for a in predict_data[0]["answers"]):
        is_answer_list_of_list = True
    else:
        is_answer_list_of_list = False

    # cold-start
    assert not (args.cold_start and args.warm_start), "Choose one mode between cold_start and warm_start"
    if args.cold_start or args.warm_start:
        with open(os.path.join(args.dataset_preprocess_dir, "train_data.json"), "r") as f:
            train_data = json.load(f)
        components = set()
        pattern = r'\[.*?\]'
        for data in train_data:
            matches = re.findall(pattern, data["sparql_proc"])
            for match in matches:
                components.add(match)

    ranks = []
    inexecutables = []
    invalids = []
    f1s = []
    h1s = []
    ems = []

    zero_answer_questions = 0

    for num_p, predict in tqdm(enumerate(predict_data)):
        text = predict["text"]
        answers = predict["answers"]
        gt_sparql = predict["gt_sparql"]

        if args.cold_start or args.warm_start:
            matches = re.findall(pattern, gt_sparql)
            if args.cold_start and not any([match not in components for match in matches]):
                continue
            if args.warm_start and any([match not in components for match in matches]):
                continue

        draft_sparqls = predict["candidates"]
        predicted_queries = dataset.postprocess(
            token_entity_dict,
            token_relation_dict,
            draft_sparqls,
        )

        if is_answer_list_of_list:
            if sum([len(answer_set) for answer_set in answers]) == 0:
                zero_answer_questions += 1
                continue
        else:
            if len(answers) == 0:
                zero_answer_questions += 1
                continue
        
        predicted_answers = None
        inexecutables.append(False)
        invalids.append(False)
        first_syntax_correct_idx = -1

        print("===============================")
        print(text)
        print("---")
        print("TRUE: ", gt_sparql)
        print("---")

        # Answer retrieve
        for i in range(len(draft_sparqls)):
            draft_sparql = draft_sparqls[i]
            query = predicted_queries[i]

            if draft_sparql == gt_sparql:
                print("PRED: ", draft_sparql)
                if is_answer_list_of_list:
                    predicted_answers = answers[0]
                else:
                    predicted_answers = answers
                ranks.append(i + 1)
                break

            try:
                predicted_answers = dataset.query(query, args.endpoint)
            except QueryBadFormed:
                print(f"* Bad Formed ({i}): {draft_sparql}")
                continue
            except AssertionError:
                print(f"* Assertion Error ({i}): {draft_sparql}")
                continue
            except EndPointInternalError:
                print(f"* EndPointInternal Error ({i}): {draft_sparql}")
                continue
            except json.decoder.JSONDecodeError:
                print(f"* JSONDecodeError ({i}): {draft_sparql}")
                continue
            except Exception as e:
                print(f"* Exception ({i}): {draft_sparql}")
                print(e)
                continue
            
            if len(predicted_answers) == 0:
                # if all num_beams query is invalid or inexecutable, first syntax correct query is used
                if first_syntax_correct_idx == -1:
                    first_syntax_correct_idx = i
            else:
                # first non-empty answers are retrieved
                print("PRED: ", draft_sparql)
                ranks.append(i + 1)
                break

        # all generated queries are inexecutable (syntax error)
        if predicted_answers is None:
            inexecutables[-1] = True
            f1s.append(0)
            h1s.append(0)
            ems.append(0)
            print("FINISHED: INEXECUTABLE")
            continue
        # all generated queries are invalid (no syntax error)
        elif len(predicted_answers) == 0:
            invalids[-1] = True
            f1s.append(0)
            h1s.append(0)
            ems.append(0)
            print("PRED (INVALID): ", draft_sparqls[first_syntax_correct_idx])
            print("FINISHED: INVALID")
            continue

        pred = set(predicted_answers)
        highest_score_index = 0
        if is_answer_list_of_list:
            best_f1 = 0
            best_h1 = 0
            for index_answer, answer_set in enumerate(answers):
                gt = set(answer_set)
                f1 = get_f1(pred, gt)
                h1 = get_h1(pred, gt)
                if f1 > best_f1:
                    highest_score_index = index_answer
                    best_f1 = f1
                    best_h1 = h1
            f1 = best_f1
            h1 = best_h1
        else:
            gt = set(answers)
            f1 = get_f1(pred, gt)
            h1 = get_h1(pred, gt)

        f1s.append(f1)
        h1s.append(h1)
        if f1 == 1:
            ems.append(1)
        else:
            ems.append(0)

        print("---")
        if is_answer_list_of_list:
            print("TRUE: ", f"{answers[highest_score_index][:5]} ..." if len(answers[highest_score_index]) > 5 else answers[highest_score_index])
        else:
            print("TRUE: ", f"{answers[:5]} ..." if len(answers) > 5 else answers)
        print("---")
        print(
            "PRED: ",
            f"{predicted_answers[:5]} ..."
            if len(predicted_answers) > 5
            else predicted_answers,
        )

        if num_p % 100 == 0:
            print()
            print("======CURRENT RESULT======")
            print("INEXECUTABLES: ", sum(inexecutables) / len(inexecutables))
            print("INVALIDS: ", sum(invalids) / len(invalids))
            print("F1 score: ", sum(f1s) / len(f1s))
            print("Hits@1 score: ", sum(h1s) / len(h1s))
            print("EM score: ", sum(ems) / len(ems))
            print("Average ranks for valid queries: ", sum(ranks) / len(ranks))
            print()

    print()
    print("======FINAL RESULT======")
    print("INEXECUTABLES: ", sum(inexecutables) / len(inexecutables))
    print("INVALIDS: ", sum(invalids) / len(invalids))
    print("F1 score: ", sum(f1s) / len(f1s))
    print("Hits@1 score: ", sum(h1s) / len(h1s))
    print("EM score: ", sum(ems) / len(ems))
    print("Average ranks for valid queries: ", sum(ranks) / len(ranks))
    print("Number of zero-answer questions: ", zero_answer_questions)
    print()
    sys.stdout.close()

if __name__ == "__main__":
    main()