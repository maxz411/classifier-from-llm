# Data and base-model provenance

Dataset loaders download from the original distribution sites. No dataset cache or full input corpus is included in this release. Third-party rights are not replaced by the code license.

The table records publisher-card metadata observed on 20 September 2026. An absent or `unknown` license is unresolved, not permission to redistribute. Dataset and base-model revisions were not pinned in the original experiments.

| Kind | Source | Publisher-card license |
|---|---|---|
| dataset | [ChilleD/StrategyQA](https://huggingface.co/datasets/ChilleD/StrategyQA) | mit |
| dataset | [Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag) | unspecified |
| dataset | [SetFit/sst5](https://huggingface.co/datasets/SetFit/sst5) | unspecified |
| dataset | [Yelp/yelp_review_full](https://huggingface.co/datasets/Yelp/yelp_review_full) | other |
| dataset | [allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc) | cc-by-sa-4.0 |
| dataset | [allenai/openbookqa](https://huggingface.co/datasets/allenai/openbookqa) | unknown |
| dataset | [allenai/qasc](https://huggingface.co/datasets/allenai/qasc) | cc-by-4.0 |
| dataset | [allenai/sciq](https://huggingface.co/datasets/allenai/sciq) | cc-by-nc-3.0 |
| dataset | [allenai/winogrande](https://huggingface.co/datasets/allenai/winogrande) | unspecified |
| dataset | [cais/mmlu](https://huggingface.co/datasets/cais/mmlu) | mit |
| dataset | [cardiffnlp/tweet_eval](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | unknown |
| dataset | [clinc/clinc_oos](https://huggingface.co/datasets/clinc/clinc_oos) | cc-by-3.0 |
| dataset | [community-datasets/yahoo_answers_topics](https://huggingface.co/datasets/community-datasets/yahoo_answers_topics) | unknown |
| dataset | [cornell-movie-review-data/rotten_tomatoes](https://huggingface.co/datasets/cornell-movie-review-data/rotten_tomatoes) | unknown |
| dataset | [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion) | other |
| dataset | [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news) | unknown |
| dataset | [fancyzhx/dbpedia_14](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | cc-by-sa-3.0 |
| dataset | [google-research-datasets/go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) | apache-2.0 |
| dataset | [google/boolq](https://huggingface.co/datasets/google/boolq) | cc-by-sa-3.0 |
| dataset | [mteb/amazon_counterfactual](https://huggingface.co/datasets/mteb/amazon_counterfactual) | cc-by-4.0 |
| dataset | [mteb/amazon_massive_intent](https://huggingface.co/datasets/mteb/amazon_massive_intent) | apache-2.0 |
| dataset | [mteb/banking77](https://huggingface.co/datasets/mteb/banking77) | mit |
| dataset | [stanfordnlp/imdb](https://huggingface.co/datasets/stanfordnlp/imdb) | other |
| dataset | [tau/commonsense_qa](https://huggingface.co/datasets/tau/commonsense_qa) | mit |
| dataset | [ucirvine/sms_spam](https://huggingface.co/datasets/ucirvine/sms_spam) | unknown |
| model | [HuggingFaceTB/SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) | apache-2.0 |
| model | [Qwen/Qwen2.5-32B](https://huggingface.co/Qwen/Qwen2.5-32B) | apache-2.0 |
| model | [Qwen/Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | apache-2.0 |
| model | [Qwen/Qwen3-1.7B-Base](https://huggingface.co/Qwen/Qwen3-1.7B-Base) | apache-2.0 |
| model | [Qwen/Qwen3-14B-Base](https://huggingface.co/Qwen/Qwen3-14B-Base) | apache-2.0 |
| model | [Qwen/Qwen3-4B-Base](https://huggingface.co/Qwen/Qwen3-4B-Base) | apache-2.0 |
| model | [Qwen/Qwen3-8B-Base](https://huggingface.co/Qwen/Qwen3-8B-Base) | apache-2.0 |
| model | [allenai/OLMo-2-0425-1B](https://huggingface.co/allenai/OLMo-2-0425-1B) | apache-2.0 |
| model | [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | apache-2.0 |

SciQ is marked CC BY-NC 3.0, and several other training sources have custom or
unspecified terms. The code's Apache-2.0 license does not grant rights to these
datasets or to model weights trained on them. No model weights are included.

Task configurations, split selection, truncation, label rendering and single-label filtering are defined in `src/packreadout/tasks/`.
