# LooGLE Benchmark

A comprehensive benchmark tool for measuring total time and TPS (Tokens Per Second) performance of vLLM models using the LooGLE dataset. This benchmark provides flexible configuration options for various testing scenarios with real-world long-context data.

## Features

- **LooGLE Dataset Integration**: Uses the official bigai-nlco/LooGLE dataset for realistic long-context benchmarking
- **Total Time Measurement**: Measures the complete execution time of all requests
- **TPS Calculation**: Calculates tokens per second both per-request and overall
- **Intelligent Context Management**: Automatically handles context + question length to match target word counts
- **Flexible Execution Orders**: 
  - `random`: Concurrent execution with random order (default)
  - `sequential`: Completely sequential execution
  - `randomSequential`: Random order but waits for each request to complete
- **Batch Processing**: Test multiple prompt counts and repetition scenarios in one run
- **Comprehensive Metrics**: Detailed statistics and performance data
- **JSON Output**: Structured results for analysis and comparison

## Usage

### Basic Usage

```bash
python LooGLE_benchmark.py \
  --results-file results.json \
  --num-input-words 10000 \
  --num-output-tokens 2 \
  --num-unique-prompts "1,10,50" \
  --num-repetitions "1,1" \
  --model your-model-name
```

### High Concurrency Benchmark

```bash
python LooGLE_benchmark.py \
  --results-file high_concurrency_results.json \
  --num-input-words 20000 \
  --num-output-tokens 10 \
  --num-unique-prompts "100,200,500" \
  --num-repetitions "5,10" \
  --num-concurrent-requests -1 \
  --model your-model-name
```

### Sequential Execution Test

```bash
python LooGLE_benchmark.py \
  --results-file sequential_results.json \
  --num-input-words 15000 \
  --num-output-tokens 5 \
  --num-unique-prompts "50,100" \
  --num-repetitions "2,3" \
  --execution-order sequential \
  --model your-model-name
```

### Cache Performance Test

```bash
# Test caching benefits with repeated prompts
python LooGLE_benchmark.py \
  --results-file cache_test.json \
  --num-input-words 10000 \
  --num-output-tokens 2 \
  --num-unique-prompts "10" \
  --num-repetitions "1,1" \
  --model your-model-name \
  --description "Testing cache performance"
```

## Parameters

### Required Parameters

- `--model`: Model to use for benchmarking (required)

### Optional Parameters

- `--results-file`: Path to JSON file for saving results (default: `server_benchmark_results.json`)
- `--num-input-words`: Number of input words per prompt (default: 10000)
- `--num-output-tokens`: Number of output tokens to generate per request (default: 2)
- `--num-unique-prompts`: Comma-separated list of unique prompts to test (default: `"1,10,50,100"`)
- `--num-repetitions`: Comma-separated list of repetitions for each prompt (default: `"1,1"`)
- `--num-concurrent-requests`: Number of concurrent requests (-1 means unlimited, default: -1)
- `--execution-order`: Execution order (`random`, `sequential`, `randomSequential`, default: `random`)
- `--api-base`: Base URL of the vLLM server API (default: `http://localhost:8000/v1`)
- `--seed`: Random seed for reproducibility (default: 42)
- `--temperature`: Sampling temperature (default: 0.7)
- `--description`: Optional description for the experiment (default: empty)

## Output Format

The benchmark generates a JSON file with the following structure:

```json
{
  "metadata": {
    "timestamp": "2024-01-01T12:00:00",
    "parameters": {
      "num_input_words": 10000,
      "num_output_tokens": 2,
      "num_unique_prompts": "1,10,50,100",
      "num_repetitions": "1,1",
      "concurrent_requests": -1,
      "execution_order": "random",
      "model": "your-model-name",
      "api_base": "http://localhost:8000/v1",
      "seed": 42,
      "temperature": 0.7,
      "description": ""
    }
  },
  "results": {
    "1": {
      "summary": {
        "total_requests": 2,
        "total_time_seconds": 1.2,
        "total_input_tokens": 20000,
        "total_output_tokens": 4,
        "total_tokens": 20004
      },
      "throughput": {
        "overall_tps": 3.33,
        "input_tokens_per_second": 16666.67,
        "requests_per_second": 1.67,
        "request_tps_mean": 3.5,
        "request_tps_std": 0.2,
        "request_tps_min": 3.2,
        "request_tps_max": 3.8
      },
      "tokens": {
        "input_tokens_mean": 10000.0,
        "input_tokens_std": 0.0,
        "output_tokens_mean": 2.0,
        "output_tokens_std": 0.0,
        "input_tokens_per_request": [10000, 10000],
        "output_tokens_per_request": [2, 2]
      }
    },
    "10": { ... },
    "50": { ... },
    "100": { ... }
  }
}
```

## Execution Orders

### Random (Default)
Executes requests concurrently in random order. This is the most realistic scenario for testing server performance under load.

### Sequential
Executes all requests one after another, ignoring the concurrent requests parameter. Useful for measuring single-threaded performance.

### RandomSequential
Executes requests in random order but waits for each request to complete before starting the next. This ensures no overlap between requests while still providing randomization.

## Metrics Explained

- **Total Time**: Complete wall clock time for the entire benchmark
- **Overall TPS**: Total output tokens divided by total time
- **Request TPS**: Individual TPS for each request
- **Input/Output Tokens**: Detailed token counts for analysis
- **Detailed Results**: Complete per-request timing and token data

## Setup

Make sure you have a vLLM server running and install the required dependencies:

```bash
# Install dependencies
pip install vllm openai datasets

# Start vLLM server (example)
vllm serve your-model --port 8000
```

**Note**: The benchmark automatically downloads the LooGLE dataset from Hugging Face on first run.

## Example Scenarios

### Performance Testing
```bash
# Test high-load scenario with many prompts
python LooGLE_benchmark.py \
  --results-file perf_test.json \
  --num-input-words 15000 \
  --num-output-tokens 10 \
  --num-unique-prompts "100,500,1000" \
  --num-repetitions "5,10" \
  --num-concurrent-requests -1 \
  --model your-model-name

# Test sequential baseline for comparison
python LooGLE_benchmark.py \
  --results-file baseline.json \
  --num-input-words 15000 \
  --num-output-tokens 10 \
  --num-unique-prompts "100,500" \
  --num-repetitions "5,10" \
  --execution-order sequential \
  --model your-model-name
```

### Cache Performance Analysis
```bash
# Compare first vs repeated executions to measure caching benefits
python LooGLE_benchmark.py \
  --results-file cache_analysis.json \
  --num-input-words 20000 \
  --num-output-tokens 2 \
  --num-unique-prompts "10,50,100" \
  --num-repetitions "1,5,10" \
  --model your-model-name \
  --description "Cache performance analysis"
```

### Scaling Analysis
```bash
# Test different concurrency levels
for concurrent in 1 4 8 16 -1; do
  python LooGLE_benchmark.py \
    --results-file scaling_${concurrent}.json \
    --num-input-words 12000 \
    --num-output-tokens 5 \
    --num-unique-prompts "50,100" \
    --num-repetitions "3,5" \
    --num-concurrent-requests $concurrent \
    --model your-model-name \
    --description "Scaling test with ${concurrent} concurrent requests"
done
```

### Long Context Performance
```bash
# Test very long contexts (up to dataset limit)
python LooGLE_benchmark.py \
  --results-file long_context.json \
  --num-input-words 50000 \
  --num-output-tokens 2 \
  --num-unique-prompts "5,10,20" \
  --num-repetitions "2,3" \
  --model your-model-name \
  --description "Long context performance test"
``` 