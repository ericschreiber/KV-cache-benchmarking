#!/usr/bin/env python3
"""
LooGLE Benchmark - A comprehensive benchmark for measuring total time and TPS
Oriented at tokenomics but with customizable parameters for flexible testing.

Input Parameters:
- json output file
- number of input words
- number of output tokens
- number of prompts
- number of repetitions of the prompt
- number of concurrent requests (-1 means all)
- order of repetitions (random, sequential, randomSequential)
  Random: random order of repetitions (default)
  Sequential: Completely sequential. Ignores the number concurrent requests.
  RandomSequential: Random order of repetitions, but for each repetition it is made sure that previous one has finished.
"""

import argparse
import concurrent.futures
import json
import random
import statistics
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
from openai import OpenAI
from datasets import load_dataset


class DatasetHandler:
    @staticmethod
    def loogle_handler(num_unique_prompts, num_input_words, seed, offset):
        ds = load_dataset("bigai-nlco/LooGLE", "longdep_qa")
        dataset_size = len(ds["test"])
        assert (
            num_unique_prompts + offset <= dataset_size
        ), f"Requested {num_unique_prompts} unique prompts, with offset {offset}, dataset has {dataset_size} unique prompts. Offset is due to the number of prompts already run."

        sampled_dataset = (
            ds["test"]
            .shuffle(seed=seed)
            .select(range(offset, offset + num_unique_prompts))
        )

        conversations = []
        for item in sampled_dataset:
            # Combine context and question, then limit to num_input_words
            num_words_context = len(item["context"].split())
            num_words_question = len(item["question"].split())

            # make the combined length of context and question equal to num_input_words
            if num_words_context + num_words_question > num_input_words:
                context = item["context"].split()[
                    : num_input_words - num_words_question
                ]
                question = item["question"].split()
            else:  # repeat the context to fill up
                context = item["context"].split()
                question = item["question"].split()
                context = context * (num_input_words // len(context) + 1)
                context = context[: num_input_words - len(question)]

            limited_content = " ".join(context + question)

            messages = [
                {"role": "user", "content": limited_content},
            ]
            conversations.append(messages)

        return conversations


DATASET_HANDLERS = {"LooGLE": DatasetHandler.loogle_handler}


class ExecutionOrder:
    @staticmethod
    def execute_sequential(
        client,
        model,
        prompts,
        num_repetitions,
        num_output_tokens,
        num_concurrent_requests,
        temperature,
    ):
        assert (
            num_concurrent_requests == 1
        ), "Sequential execution does not support concurrent requests"
        results = []
        for prompt in prompts:
            for _ in range(num_repetitions):
                results.append(
                    call_server_completion(
                        client,
                        model,
                        prompt,
                        temperature,
                        num_output_tokens,
                    )
                )
        return results

    @staticmethod
    def execute_random_concurrent(
        client,
        model,
        prompts,
        num_repetitions,
        num_output_tokens,
        num_concurrent_requests,
        temperature,
    ):
        if num_concurrent_requests == -1:
            num_concurrent_requests = len(prompts) * num_repetitions
        if num_concurrent_requests > len(prompts) * num_repetitions:
            raise ValueError(
                f"Number of concurrent requests ({num_concurrent_requests}) is greater than the number of prompts ({len(prompts) * num_repetitions})"
            )

        # Extend the list of prompts num_repetitions times and shuffle them
        extended_prompts = []
        for _ in range(num_repetitions):
            extended_prompts.extend(prompts)

        # Shuffle the extended list
        random.shuffle(extended_prompts)
        prompts = extended_prompts

        results = []
        # Execute the tasks concurrently
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_concurrent_requests
        ) as executor:
            futures = [
                executor.submit(
                    call_server_completion,
                    client,
                    model,
                    prompt,
                    temperature,
                    num_output_tokens,
                )
                for prompt in prompts
            ]
            for future in concurrent.futures.as_completed(futures):
                prompt_tokens, completion_tokens, tps, start_time, end_time = (
                    future.result()
                )
                results.append(
                    (prompt_tokens, completion_tokens, tps, start_time, end_time)
                )

        return results

    @staticmethod
    def execute_randomSequential(
        client,
        model,
        prompts,
        num_repetitions,
        num_output_tokens,
        num_concurrent_requests,
        temperature,
    ):
        # Execute the tasks concurrently but make sure that the previous one has finished
        # Therefore, make min(num_concurrent_requests, len(prompts)) requests at a time and each thread executes the
        # prompt num_repetitions times.

        if num_concurrent_requests == -1:
            num_concurrent_requests = len(prompts)
        if num_concurrent_requests > len(prompts):
            raise ValueError(
                f"Number of concurrent requests ({num_concurrent_requests}) is greater than the number of prompts ({len(prompts)})"
            )

        results = []  # Initialize results list
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=num_concurrent_requests
        ) as executor:
            futures = [
                executor.submit(
                    ExecutionOrder.execute_sequential,  # Use correct class name
                    client,
                    model,
                    [prompt],
                    num_repetitions,
                    num_output_tokens,
                    1,  # Sequential execution needs 1 concurrent request
                    temperature,
                )
                for prompt in prompts
            ]
            for future in concurrent.futures.as_completed(futures):
                results.extend(future.result())  # Use extend instead of +=
        return results


EXECUTION_ORDERS = {
    "sequential": ExecutionOrder.execute_sequential,
    "random": ExecutionOrder.execute_random_concurrent,
    "randomSequential": ExecutionOrder.execute_randomSequential,
}


def create_sample_conversations(
    dataset_key: str,
    num_unique_prompts: int,
    num_input_words: int,
    seed: int = 42,
    offset: int = 0,
):
    handler = DATASET_HANDLERS.get(dataset_key)
    if not handler:
        raise ValueError(f"Unknown dataset key: {dataset_key}")
    return handler(num_unique_prompts, num_input_words, seed, offset)


def call_server_completion(client, model, messages, temperature, max_tokens):
    try:
        start_time = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        prompt_tokens = response.usage.prompt_tokens
        completion_tokens = response.usage.completion_tokens
        tokens_per_second = completion_tokens / elapsed if elapsed > 0 else 0
        return prompt_tokens, completion_tokens, tokens_per_second, start_time, end_time
    except Exception as e:
        print(f"Error during API call: {e}")
        return 0, 0, 0, 0, 0


def calculate_metrics(
    results: List[Tuple], start_time: float, end_time: float
) -> Dict[str, Any]:
    """Calculate comprehensive metrics from results"""
    if not results:
        return {}

    prompt_tokens_list = [r[0] for r in results]
    completion_tokens_list = [r[1] for r in results]
    tps_list = [r[2] for r in results if r[2] > 0]  # Filter out zero TPS
    duration_list = [
        r[4] - r[3] for r in results
    ]  # Calculate duration for each request

    total_time = end_time - start_time
    total_input_tokens = sum(prompt_tokens_list)
    total_output_tokens = sum(completion_tokens_list)
    total_tokens = total_input_tokens + total_output_tokens

    metrics = {
        "summary": {
            "total_requests": len(results),
            "total_time_seconds": total_time,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
        },
        "throughput": {
            "overall_tps": (total_output_tokens / total_time if total_time > 0 else 0),
            "input_tokens_per_second": (
                total_input_tokens / total_time if total_time > 0 else 0
            ),
            "requests_per_second": (len(results) / total_time if total_time > 0 else 0),
            "request_tps_mean": statistics.mean(tps_list) if tps_list else 0,
            "request_tps_std": (statistics.stdev(tps_list) if len(tps_list) > 1 else 0),
            "request_tps_min": min(tps_list) if tps_list else 0,
            "request_tps_max": max(tps_list) if tps_list else 0,
        },
        "timing": {
            "request_duration_mean": (
                statistics.mean(duration_list) if duration_list else 0
            ),
            "request_duration_std": (
                statistics.stdev(duration_list) if len(duration_list) > 1 else 0
            ),
            "request_duration_min": min(duration_list) if duration_list else 0,
            "request_duration_max": max(duration_list) if duration_list else 0,
            "request_duration_list": duration_list,
        },
        "tokens": {
            "input_tokens_mean": (
                statistics.mean(prompt_tokens_list) if prompt_tokens_list else 0
            ),
            "input_tokens_std": (
                statistics.stdev(prompt_tokens_list)
                if len(prompt_tokens_list) > 1
                else 0
            ),
            "output_tokens_mean": (
                statistics.mean(completion_tokens_list) if completion_tokens_list else 0
            ),
            "output_tokens_std": (
                statistics.stdev(completion_tokens_list)
                if len(completion_tokens_list) > 1
                else 0
            ),
            "input_tokens_per_request": prompt_tokens_list,
            "output_tokens_per_request": completion_tokens_list,
        },
    }

    return metrics


def run_benchmark(
    api_base: str = "http://localhost:8000/v1",
    model: str = "default",
    execution_order: str = "random",
    num_input_words: int = 100,
    num_output_tokens: int = 2,
    num_unique_prompts: int = 100,
    offset: int = 0,
    num_repetitions: int = 10,
    num_concurrent_requests: int = -1,
    seed: int = 42,
    dataset_key: str = "LooGLE",
    temperature: float = 0.7,
):
    client = OpenAI(api_key="sk-dummy", base_url=api_base)

    if execution_order not in EXECUTION_ORDERS:
        raise ValueError(f"Unknown execution order: {execution_order}")

    execution_func = EXECUTION_ORDERS[execution_order]
    prompts = create_sample_conversations(
        dataset_key, num_unique_prompts, num_input_words, seed, offset
    )

    print(f"=== LooGLE Benchmark Configuration ===")
    print(f"Input words per prompt: {num_input_words}")
    print(f"Output tokens per request: {num_output_tokens}")
    print(f"Number of unique prompts: {num_unique_prompts}")
    print(f"Repetitions per prompt: {num_repetitions}")
    print(
        f"Concurrent requests: {num_concurrent_requests if num_concurrent_requests != -1 else 'unlimited'}"
    )
    print(f"Execution order: {execution_order}")
    print(f"Dataset: {dataset_key}")
    print(f"Random seed: {seed}")
    print()

    # Time the benchmark execution
    benchmark_start = time.perf_counter()

    results = execution_func(
        client,
        model,
        prompts,
        num_repetitions,
        num_output_tokens,
        num_concurrent_requests,
        temperature,
    )

    benchmark_end = time.perf_counter()

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(results, benchmark_start, benchmark_end)

    return metrics


def save_results(results: Dict[str, Any], filename: str):
    """Save results to JSON file"""
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="LooGLE Benchmark - Comprehensive vLLM benchmarking tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic benchmark
  python LooGLE_benchmark.py --results-file results.json --num-input-words 100 --num-output-tokens 50 --num-unique-prompts 5 --num-repetitions 3 --num-concurrent-requests 1 --execution-order random --model default --api-base http://localhost:8000/v1 --seed 42 --temperature 0.7 --description "Basic benchmark"

  # High concurrency benchmark
  python LooGLE_benchmark.py --results-file results.json --num-input-words 200 --num-output-tokens 100 --num-unique-prompts 100 --num-repetitions 10 --num-concurrent-requests -1 --execution-order random --model default --api-base http://localhost:8000/v1 --seed 42 --temperature 0.7 --description "High concurrency benchmark"

  # Sequential execution
  python LooGLE_benchmark.py --results-file results.json --num-input-words 150 --num-output-tokens 75 --num-unique-prompts 100 --num-repetitions 10 --num-concurrent-requests 1 --execution-order sequential --model default --api-base http://localhost:8000/v1 --seed 42 --temperature 0.7 --description "Sequential execution"
        """,
    )

    parser.add_argument(
        "--results-file",
        type=str,
        default="server_benchmark_results.json",
        help="Path to JSON file for saving results.",
    )
    parser.add_argument(
        "--num-input-words",
        type=int,
        default=10000,
        help="Number of input words per prompt",
    )
    parser.add_argument(
        "--num-output-tokens",
        type=int,
        default=2,
        help="Number of output tokens to generate per request",
    )
    parser.add_argument(
        "--num-unique-prompts",
        type=str,
        default="1,10,50,100",
        help="Comma-separated list of number of unique prompts to create (e.g. 100,200,300)",
    )
    parser.add_argument(
        "--num-repetitions",
        type=str,
        default="1,1",
        help="Comma-separated list of number of repetitions to run for each prompt (e.g. 1,2,3 or 1,1 if you want to see the difference caching brings)",
    )
    parser.add_argument(
        "--num-concurrent-requests",
        type=int,
        default=-1,
        help="Number of concurrent requests (-1 means all) (default: 4)",
    )
    parser.add_argument(
        "--execution-order",
        type=str,
        choices=["random", "sequential", "randomSequential"],
        default="random",
        help="Order of repetitions (default: random)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model to use for benchmarking (default: default)",
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL of the vLLM server API (default: http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="",
        help="Optional description or notes for the experiment.",
    )

    args = parser.parse_args()
    num_unique_prompts = [
        int(bs.strip()) for bs in args.num_unique_prompts.split(",") if bs.strip()
    ]
    num_repetitions = [
        int(reps.strip()) for reps in args.num_repetitions.split(",") if reps.strip()
    ]

    # Create metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "parameters": {
            "num_input_words": args.num_input_words,
            "num_output_tokens": args.num_output_tokens,
            "num_unique_prompts": args.num_unique_prompts,
            "num_repetitions": args.num_repetitions,
            "concurrent_requests": args.num_concurrent_requests,
            "execution_order": args.execution_order,
            "model": args.model,
            "api_base": args.api_base,
            "seed": args.seed,
            "temperature": args.temperature,
            "description": args.description,
        },
    }

    # Run benchmark for each combination of unique prompts and repetitions
    sum_num_prompts = 0
    results_dict = {}
    for num_prompts in num_unique_prompts:
        results_dict[num_prompts] = {}
        for index, num_reps in enumerate(num_repetitions):
            print(
                f"\n=== Running benchmark: {num_prompts} prompts, {num_reps} repetitions ==="
            )
            results = run_benchmark(
                api_base=args.api_base,
                model=args.model,
                num_input_words=args.num_input_words,
                num_output_tokens=args.num_output_tokens,
                num_unique_prompts=num_prompts,
                offset=sum_num_prompts,
                num_repetitions=num_reps,
                num_concurrent_requests=args.num_concurrent_requests,
                execution_order=args.execution_order,
                seed=args.seed,
                temperature=args.temperature,
            )
            results_dict[num_prompts][f"repetition_{index}:{num_reps}"] = results

            # Display summary for this run
            print(
                f"\n=== Benchmark Results Summary: {num_prompts} prompts, {num_reps} repetitions ==="
            )
            print(f"Total requests: {results['summary']['total_requests']}")
            print(f"Total time: {results['summary']['total_time_seconds']:.2f} seconds")
            print(f"Total output tokens: {results['summary']['total_output_tokens']}")
            print(f"Overall TPS: {results['throughput']['overall_tps']:.2f}")
            print(
                f"Requests per second: {results['throughput']['requests_per_second']:.2f}"
            )
            print(
                f"Average request TPS: {results['throughput']['request_tps_mean']:.2f} ± {results['throughput']['request_tps_std']:.2f}"
            )

    # Add metadata to results
    final_results = {"metadata": metadata, "results": results_dict}

    # Save results
    save_results(final_results, args.results_file)
    print(f"\n=== All benchmarks completed! ===")
    print(f"Results saved to {args.results_file}")


def test_benchmark_output_input_conversations():
    # print out the created conversations
    prompts = create_sample_conversations("LooGLE", 100, 100, 42, 0)
    for prompt in prompts:
        print(prompt)

    # print out the length of the prompts
    print(len(prompts))


if __name__ == "__main__":
    main()
