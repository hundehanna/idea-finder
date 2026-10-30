# Idea Finder

Idea Finder reads bad product reviews and turns them into a ranked list of problems that people keep running into. The goal is to find gaps a new product could fill.

Many good products start with a problem the founder keeps hitting. This tool does the same search at scale: it reads thousands of 1–3 star reviews and finds the problems that show up again and again, across many different products.

## How it works

The script runs in six stages. Each stage saves its output, so you can rerun any step without starting over.

1. **collect** – Pulls 1–3 star reviews for a category from the [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset, or loads your own CSV/JSONL file. You can filter by keyword.
2. **extract** – Claude (Haiku) reads the reviews in batches and pulls out each real problem as a short, plain statement. It also notes the context, who has the problem, any workaround, and what they wished for. Complaints that are only about shipping, the seller, or price are skipped.
3. **cluster** – Embeds the problem statements and groups similar ones together.
4. **label** – Claude (Sonnet) names each group, describes the underlying need, explains why current products fail, and suggests product directions. It also marks each group as a design gap or a quality defect.
5. **report** – Scores and ranks the groups and writes `report.md` and `clusters.csv`.
6. **sample** – Writes a random sample of extractions to a CSV so you can grade them by hand.

## Setup

You need Python 3.10 or newer and an Anthropic API key.

```bash
pip install -r requirements.txt
```

Set your API key.

Windows (Command Prompt / Anaconda Prompt):

```
set ANTHROPIC_API_KEY=sk-ant-your-key-here
```

macOS / Linux:

```bash
export ANTHROPIC_API_KEY=sk-ant-your-key-here
```

`sentence-transformers` is optional but recommended. Without it, the script falls back to TF-IDF, which is worse at grouping problems that mean the same thing but use different words.

## Usage

Start with a small run to check the output and cost:

```bash
python idea_finder.py all --category Baby_Products --keyword "diaper bag" --n 200 --scan 300000
```

Then scale up:

```bash
python idea_finder.py all --category Home_and_Kitchen --keyword lunch --n 2000
```

Use your own data (any CSV or JSONL with a `text` column; `rating`, `title`, `product`, `timestamp` and `helpful_vote` are optional):

```bash
python idea_finder.py all --input my_reviews.csv --workdir runs/my_data
```

Rerun only later stages after changing settings:

```bash
python idea_finder.py cluster label report --workdir runs/Baby_Products_diaper-bag --threshold 0.5
```

Run `python idea_finder.py --help` to see all options.

## Output

Each run saves its files in `runs/<name>/`:

| File | What it is |
|---|---|
| `reviews.jsonl` | The reviews that were collected |
| `extractions.jsonl` | Problems extracted from each review |
| `problems_clustered.jsonl` | Each problem with its group number |
| `labels.json` | Name, need and product directions for each group |
| `report.md` | The ranked report, with example quotes |
| `clusters.csv` | The same results as a table |
| `eval_sample.csv` | Sample extractions to grade by hand |

## How the score works

Each group gets a score from 0 to 100 based on:

- **Frequency (35%)** – how many problems are in the group
- **Breadth (25%)** – how many different products share the problem. A problem shared by many products is a gap in the whole category, not one bad product.
- **Workarounds (20%)** – how often people built their own fix. This is a strong sign people would pay for a real solution.
- **Pain (20%)** – share of 1-star reviews

Groups labeled as quality defects get their score cut in half, because they are usually manufacturing issues and not new product opportunities. You can change the weights in `WEIGHTS` at the top of the report section.

The report is a list of leads. The next step is always to talk to real people who have the problem.

## Checking the extraction step

The whole report depends on the extraction step being right. To check it:

```bash
python idea_finder.py sample --workdir runs/<name>
```

Open `eval_sample.csv` and mark each row: are the extracted problems correct, did the model make anything up, and did it miss anything? Fix the prompt (`EXTRACT_SYSTEM`) where it goes wrong, then rerun.

## Tuning

- **Too many tiny groups:** raise `--threshold`.
- **One giant group:** lower `--threshold`.
- **Too few reviews collected:** raise `--scan` (or use `--scan 0` for the whole file) or use a broader keyword.
- **Failed or skipped batches:** run `extract` again. It only retries the missing reviews.
- **Different models:** set the `EXTRACT_MODEL` or `LABEL_MODEL` environment variables.

## Notes on data

- The Amazon Reviews 2023 dataset covers May 1996 to September 2023. It is good for finding problems that have lasted for years, not for spotting new trends.
- The script streams the dataset file, so it does not download the whole category.
- For newer data, collect reviews or comments yourself and pass them in with `--input`.

## Planned

- A step that merges groups describing the same need
- A YouTube comments collector for fresher data
- An option to run extraction on a local model with Ollama
- An agent that picks categories, reads the report, and decides where to dig deeper

## Data credit

Amazon Reviews 2023 is from McAuley Lab, UC San Diego ([dataset](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023), [paper](https://arxiv.org/abs/2403.03952)).
