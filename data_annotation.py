import re
import json
import transformers
import tqdm
import multiprocessing
import os
import argparse

DEFAULT_PAD_TOKEN = "<pad>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_UNK_TOKEN = "<unk>"


def load_tokenizer(model_name_or_path):
    """
    Initialize a tokenizer with specified configurations.

    Args:
        model_name_or_path (str): Path or name of the pre-trained model

    Returns:
        transformers.PreTrainedTokenizer: Initialized tokenizer
    """
    print(f"+ [Model] Initializing Tokenizer: {model_name_or_path}")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side="right",
        use_fast=False,
    )

    if "phi" in model_name_or_path:
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens(
                {
                    "eos_token": DEFAULT_EOS_TOKEN,
                    "bos_token": DEFAULT_BOS_TOKEN,
                    "unk_token": DEFAULT_UNK_TOKEN,
                }
            )

    return tokenizer


def split_string_list(text_list, delimiter="\n"):
    """
    Split a list of characters into sublists based on a delimiter.

    Args:
        text_list (list): List of characters to split
        delimiter (str): Character to split on, defaults to newline

    Returns:
        list: List of joined character strings
    """
    sublists = []
    current_sublist = []

    for item in text_list:
        current_sublist.append(item)
        if item == delimiter:
            if current_sublist:
                sublists.append("".join(current_sublist))
                current_sublist = []

    if current_sublist:
        sublists.append("".join(current_sublist))

    return sublists


def split_token_list(token_list, delimiter=13):
    """
    Split a list of tokens into sublists based on a delimiter token.

    Args:
        token_list (list): List of tokens to split
        delimiter (int): Token ID to split on, defaults to 13

    Returns:
        list: List of token sublists
    """
    sublists = []
    current_sublist = []

    for item in token_list:
        current_sublist.append(item)
        if item == delimiter:
            if current_sublist:
                sublists.append(current_sublist)
                current_sublist = []

    if current_sublist:
        sublists.append(current_sublist)

    return sublists


def evaluate_expression_para(response_all, v_score, tokenizer, is_true):
    """
    Evaluate expressions in parallel, checking for calculation errors.

    Args:
        response_all (str): Complete response text
        v_score (list): Verification scores
        tokenizer: Initialized tokenizer
        is_true (bool): Truth value flag

    Returns:
        tuple: Dict of evaluation results and processed scores
    """
    labels = []
    predictions = []
    sol_tokens = tokenizer(response_all).input_ids
    process_v_score = [0] * len(sol_tokens)
    calc_error = False
    error_detection = False

    response_list = split_string_list(response_all)
    token_list = split_token_list(sol_tokens)
    previous_len = 0

    for idx, string in enumerate(response_list):
        para_token = token_list[idx]
        para_token_location = sum([len(item) for item in token_list[:idx]])

        if error_detection:
            break

        if abs(v_score[para_token_location]) < 1e-5:
            error_detection = True

        elif (
            v_score[para_token_location + len(para_token) - 1]
            - v_score[para_token_location]
        ) / v_score[para_token_location] < -0.5:
            error_detection = True

        else:
            if not error_detection:
                process_v_score[
                    para_token_location : para_token_location + len(para_token)
                ] = [1] * len(para_token)

        previous_len += len(string)

    return {
        "label": labels,
        "prediction": predictions,
        "calc_error": calc_error,
    }, process_v_score


def process_chunk(tokenizer, chunk, wf_path):
    """
    Process a chunk of data and calculate metrics.

    Args:
        tokenizer: Initialized tokenizer
        chunk (list): Chunk of data to process
        wf_path (str): Path to write results

    Returns:
        dict: Metrics including accuracy, recall, and calculation error rates
    """
    acc = []
    recall_count = [0, 0]  # [correct positives, total positives]
    calc_errors = []

    with open(wf_path, "w", encoding="utf-8") as wf:
        for line in tqdm.tqdm(chunk):
            for output in line["outputs"]:
                v_scores = output.get("vscores", [])
                response = output.get("response", "")
                is_true = output.get("label", "")
                evaluation_results, process_v_scores = evaluate_expression_para(
                    response, v_scores, tokenizer, is_true
                )
                output["process_vscores"] = process_v_scores

                if evaluation_results["calc_error"]:
                    calc_errors.append(1)
                else:
                    calc_errors.append(0)

                for label, prediction in zip(
                    evaluation_results["label"], evaluation_results["prediction"]
                ):
                    acc.append((label, prediction))

                for idx, prediction in enumerate(evaluation_results["prediction"]):
                    label = evaluation_results["label"][idx]
                    if label == "positive":
                        recall_count[1] += 1
                        if prediction == "positive":
                            recall_count[0] += 1

            wf.writelines(json.dumps(line, ensure_ascii=False) + "\n")

    return {
        "accuracy_sum": sum(1 for label, prediction in acc if label == prediction),
        "total": len(acc),
        "recall_correct": recall_count[0],
        "recall_total": recall_count[1],
        "calc_error_sum": sum(calc_errors),
        "calc_error_total": len(calc_errors),
    }


def parallel_process_line(tokenizer, lines, wf_path, num_processes=32):
    """
    Process lines in parallel using multiple CPU cores.

    Args:
        tokenizer: Initialized tokenizer
        lines (list): Lines to process
        wf_path (str): Path to write results
        num_processes (int): Number of parallel processes to use
    """
    if num_processes is None:
        num_processes = multiprocessing.cpu_count()

    chunk_size = int(len(lines) / num_processes)
    chunks = [lines[i : i + chunk_size] for i in range(0, len(lines), chunk_size)]
    temp_files = [f"multirun/{wf_path}_temp_{i}.json" for i in range(len(chunks))]

    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(
            process_chunk,
            [
                (tokenizer, chunk, temp_file)
                for chunk, temp_file in zip(chunks, temp_files)
            ],
        )

    with open(f"multirun2/{wf_path}.json", "w", encoding="utf-8") as wf:
        for temp_file in temp_files:
            with open(temp_file, "r", encoding="utf-8") as tf:
                wf.write(tf.read())
            os.remove(temp_file)

    total_acc = sum(result["accuracy_sum"] for result in results)
    total = sum(result["total"] for result in results)
    total_recall_correct = sum(result["recall_correct"] for result in results)
    total_recall = sum(result["recall_total"] for result in results)
    total_calc_errors = sum(result["calc_error_sum"] for result in results)
    total_calc_error_counts = sum(result["calc_error_total"] for result in results)

    overall_accuracy = total_acc / total if total else 0
    overall_recall = total_recall_correct / total_recall if total_recall else 0
    overall_calc_error_rate = (
        total_calc_errors / total_calc_error_counts if total_calc_error_counts else 0
    )

    print(f"Overall accuracy: {overall_accuracy}")
    print(f"Overall recall: {overall_recall}")
    print(f"Overall calculation error rate: {overall_calc_error_rate}")


def main():
    """
    Main function to process JSONL file and perform evaluation.
    """
    parser = argparse.ArgumentParser(
        description="Process JSONL file and evaluate expressions."
    )
    parser.add_argument("file_path", type=str, help="Path to the JSONL file")
    parser.add_argument("model_path", type=str, help="Path to the model")
    args = parser.parse_args()

    # Load and filter data
    lines = [
        json.loads(line)
        for line in open(args.file_path, "r", encoding="utf-8").readlines()
    ]

    for example in lines:
        dedup_outputs = [
            output for output in example["outputs"] if len(output["tokens"]) <= 2048
        ]
        example["outputs"] = dedup_outputs

    tokenizer = load_tokenizer(args.model_path)
    parallel_process_line(tokenizer, lines, "test.json")


if __name__ == "__main__":
    main()
