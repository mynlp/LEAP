"""check_answer_in_pred, compute_edit_quality, build_qa_probes and calculate_averages below are
adapted from CaKE's (https://github.com/zjunlp/CaKE, MIT License,
Copyright (c) 2025 ZJUNLP) edit_utils.py/test_cake.py: same prompt
templates, same dataset field names, same output shapes.
"""


def check_answer_in_pred(pred, answers):
    pred = pred.lower()
    return any(a.lower() in pred for a in answers)


def cloze_and_qa_questions(cloze_text, question_text, answers):
    """Build the (cloze, qa) prompt pair used to probe a single fact, both
    scored against the same answer list."""
    questions = [
        "Answer: " + cloze_text,
        "Question: " + question_text + "\nAnswer: The answer is",
    ]
    return questions, [answers, answers]


def check_answers_batch(
    model, questions, tokenizer, answers, device, max_new_tokens=50, return_texts=False
):
    """Check multiple prompts with a single batched generation call.

    If return_texts is True, also return the raw generated text for each
    prompt alongside the correctness list, as (results, generated_texts).
    """
    if not questions:
        return ([], []) if return_texts else []

    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_attention_mask=True,
        ).to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    finally:
        tokenizer.padding_side = original_padding_side

    generated_ids = outputs[:, inputs["input_ids"].shape[1] :]
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    results = []
    for answer, generated_text in zip(answers, generated_texts):
        results.append(check_answer_in_pred(generated_text, answer))
    if return_texts:
        return results, generated_texts
    return results


def compute_edit_quality(model, tokenizer, edit_item, device):
    metrics = {
        "hop_wise": [],
        "hop_wise_generated": [],
        "accuracy": [],
        "accuracy_generated": [],
    }
    questions = []
    answers = []
    hop_count = 0

    for hop in edit_item["new_single_hops"]:
        ans = list(hop["answer_alias"]) + [hop["answer"]]
        qs, ans_list = cloze_and_qa_questions(hop["cloze"], hop["question"], ans)
        questions.extend(qs)
        answers.extend(ans_list)
        hop_count += 1

    answer = list(edit_item["new_answer_alias"]) + [edit_item["new_answer"]]
    qs, ans_list = cloze_and_qa_questions(
        edit_item["cloze_question"], edit_item["questions"][0], answer
    )
    questions.extend(qs)
    answers.extend(ans_list)

    batch_results, batch_generated_texts = check_answers_batch(
        model,
        questions,
        tokenizer,
        answers,
        device,
        max_new_tokens=10,
        return_texts=True,
    )

    for idx in range(hop_count):
        start = idx * 2
        metrics["hop_wise"].append(batch_results[start : start + 2])
        metrics["hop_wise_generated"].append(batch_generated_texts[start : start + 2])

    accuracy_start = hop_count * 2
    metrics["accuracy"].extend(batch_results[accuracy_start : accuracy_start + 2])
    metrics["accuracy_generated"].extend(
        batch_generated_texts[accuracy_start : accuracy_start + 2]
    )
    return metrics


def build_qa_probes(item):
    answer = list(item["answer_alias"]) + [item["answer"]]
    return cloze_and_qa_questions(item["cloze_question"], item["questions"][0], answer)


def build_single_hop_probes(item):
    questions = []
    answers = []
    for hop in item["single_hops"]:
        ans = list(hop["answer_alias"]) + [hop["answer"]]
        qs, ans_list = cloze_and_qa_questions(hop["cloze"], hop["question"], ans)
        questions.extend(qs)
        answers.extend(ans_list)
    return questions, answers


def compute_locality_quality(model, tokenizer, probe_items, device):
    results = []
    for probe_item in probe_items:
        questions, answers = build_qa_probes(probe_item)
        correct, generated = check_answers_batch(
            model,
            questions,
            tokenizer,
            answers,
            device,
            max_new_tokens=10,
            return_texts=True,
        )

        hop_questions, hop_answers = build_single_hop_probes(probe_item)
        hop_correct, hop_generated = check_answers_batch(
            model,
            hop_questions,
            tokenizer,
            hop_answers,
            device,
            max_new_tokens=10,
            return_texts=True,
        )
        hop_wise = [hop_correct[i : i + 2] for i in range(0, len(hop_correct), 2)]
        hop_wise_generated = [
            hop_generated[i : i + 2] for i in range(0, len(hop_generated), 2)
        ]

        results.append(
            {
                "probe_case_id": probe_item["case_id"],
                "correct": correct,
                "generated": generated,
                "hop_wise": hop_wise,
                "hop_wise_generated": hop_wise_generated,
            }
        )
    return results


def calculate_averages(data):
    total_cases = len(data)
    accuracy_true_count = [0, 0]
    hop_wise_case_averages = []

    for case in data:
        hop_wise = case["post"]["hop_wise"]
        accuracy = case["post"]["accuracy"]
        case_hop_true_count = 0
        total_hops = 0
        for hop_pair in hop_wise:
            case_hop_true_count += sum(1 for x in hop_pair if x)
            total_hops += len(hop_pair)

        hop_wise_case_averages.append(case_hop_true_count / total_hops)
        accuracy_true_count[0] += 1 if accuracy[0] else 0
        accuracy_true_count[1] += 1 if accuracy[1] else 0

    overall_hop_wise_avg = sum(hop_wise_case_averages) / total_cases
    accuracy_avg = [count / total_cases for count in accuracy_true_count]

    return {
        "total_cases": total_cases,
        "overall_hop_wise_average": f"{overall_hop_wise_avg:.3f}",
        "accuracy_averages": {
            "cloze": f"{accuracy_avg[0]:.3f}",
            "qa": f"{accuracy_avg[1]:.3f}",
        },
    }


def calculate_locality_average(data):
    final_total, final_correct = 0, 0
    hop_total, hop_correct = 0, 0
    for case in data:
        for probe in case["post"].get("locality", []):
            for is_correct in probe.get("correct", []):
                final_total += 1
                final_correct += 1 if is_correct else 0
            for hop_pair in probe.get("hop_wise", []):
                for is_correct in hop_pair:
                    hop_total += 1
                    hop_correct += 1 if is_correct else 0
    if final_total == 0 and hop_total == 0:
        return None
    result = {}
    if final_total > 0:
        result["final_answer"] = {
            "total_probes": final_total,
            "locality_accuracy": f"{final_correct / final_total:.3f}",
        }
    if hop_total > 0:
        result["single_hop"] = {
            "total_probes": hop_total,
            "locality_accuracy": f"{hop_correct / hop_total:.3f}",
        }
    return result
