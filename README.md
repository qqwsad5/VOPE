# VOPE: Voluntary-imagined Object Presence Evaluation

VOPE is a recheck-based evaluation protocol for assessing hallucinations in Large Vision-Language Models (LVLMs) on **voluntary imagination tasks** (e.g., story writing), where models are expected to generate novel content beyond the given image — a setting largely overlooked by existing hallucination benchmarks.

## Data Preparation

Download the [MSCOCO 2014](https://cocodataset.org/#download) validation images and place the `val2014/` folder under `./data/`:

```
VOPE/
└── data/
    ├── val2014/
    │   ├── COCO_val2014_000000000042.jpg
    │   └── ...
    ├── Generative/
    └── ReCheck/
```

Then update `dataset_root` in `evaluation.py` if needed (default is `./data`).

## Usage

### Step 1: Run the generative task

Set `dataset_name = 'generative'` in the `DefaultConfigs` of `evaluation.py` and specify the task type (`captioning`, `reasoning`, or `writing`):

```python
dataset_name = 'generative'
dataset_info = {
    'type': 'captioning',   # or 'reasoning' / 'writing'
}
```

Then run:

```bash
python evaluation.py
```

Results will be saved to `./results/generative_<type>/<model_name>_results.json`.

### Step 2: Generate the recheck dataset

```bash
cd data/ReCheck
python generate_reckeck_dataset.py
```

This reads the generative results and produces the recheck question file at `./data/ReCheck/generative_captioning/<model_name>.json`.

### Step 3: Run the recheck task

Set `dataset_name = 'recheck'` in `evaluation.py`:

```python
dataset_name = 'recheck'
dataset_info = {
    'target_dataset': 'generative_captioning',
}
```

Then run:

```bash
python evaluation.py
```

The final VOPE metrics (`hal_d`, `hal_i`, `exp`) will be saved to `./results/recheck_generative_captioning/<model_name>_metrics.json`.

### Step 4: Run the relevance evaluation (optional)

First, generate the label file from the recheck results:

```bash
cd rel_eval
python gen_eval_label.py
```

This produces `rel_eval/generative_captioning/<model_name>.json` containing the objects to be scored.

Then fill in your OpenAI API key and the required paths in `gpt_eval.py`, and run:

```bash
python gpt_eval.py
```

This uses GPT-4o to score the relevance between each object and the image. Results will be saved to `<save_root>/gpt_eval_<dataset>/<model_name>_results.json`.

## License

This benchmark is released under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
