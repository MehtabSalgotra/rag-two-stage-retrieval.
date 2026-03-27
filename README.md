# rag-two-stage-retrieval.
Two-stage retrieval architecture for domain-specific question answering combining Hybrid RRF with cross-encoder reranking. Master's thesis implementation demonstrating consistent 10-13% improvement over single-stage baselines.

# Two-Stage Retrieval Architecture for Domain-Specific Question Answering

Implementation of a novel two-stage retrieval architecture for Retrieval-Augmented Generation systems, evaluated on Medical, Finance, and Politics domains. This work demonstrates that separating recall and precision optimization across sequential stages yields consistent performance improvements compared to single-stage retrieval methods.

## Overview

This repository contains the complete implementation and experimental framework for evaluating five retrieval methods:

- TF-IDF (traditional lexical baseline)
- BM25 (probabilistic lexical retrieval)
- Dense Vector (neural semantic retrieval using Sentence-BERT)
- Hybrid RRF (Reciprocal Rank Fusion combining BM25 and Dense Vector)
- Two-Stage (novel architecture combining Hybrid RRF with cross-encoder reranking)

## Key Results

The Two-Stage method achieves consistent improvements of 10 to 13 percent over the best single-stage baseline (Hybrid RRF) across all three evaluated domains. While baseline methods show variable performance ranging from minimal to moderate gains depending on domain characteristics, the Two-Stage architecture maintains stable effectiveness across diverse evaluation conditions.

Statistical testing confirms high significance for all comparisons with p-values below 0.001. Score distribution analysis reveals that Two-Stage produces 68.9 percent high-quality answers with only 4.4 percent failures, compared to 53.3 percent high-quality and 15.6 percent failures for Hybrid RRF baseline.

## Experimental Setup

**Datasets:**
- Medical domain: MedQuad Medical Question Answering Dataset (200 documents)
- Finance domain: Finance-Alpaca dataset (200 documents)
- Politics domain: XSUM news summarization dataset (200 documents)

**Evaluation Protocol:**
- 45 test questions (15 per domain) generated from corpus content
- 225 total experiments (45 questions × 5 methods)
- Top-5 document retrieval for all methods
- Answer generation using Gemini 2.5 Flash (temperature 0.0, max tokens 500)
- Independent evaluation using GPT-4o Mini as judge (0-10 scale normalized to 0-1)
- Statistical validation through paired t-tests

**Two-Stage Architecture:**
- Stage 1: Hybrid RRF retrieves 10 high-recall candidates from full corpus
- Stage 2: Cross-encoder (ms-marco-MiniLM-L-6-v2) reranks candidates and selects top 5


## Requirements

Python 3.8 or higher with the following dependencies:
```
numpy<2.0
langchain-openai
langchain-google-genai
datasets
chromadb
rank-bm25
sentence-transformers
scikit-learn
pandas
scipy
tqdm
matplotlib
```

## Installation and Usage

Clone the repository and install dependencies:
```bash
git clone https://github.com/[username]/two-stage-retrieval-rag.git
cd two-stage-retrieval-rag
pip install -r requirements.txt
```

Configure API credentials in the notebook before running:
```python
os.environ["OPENAI_API_KEY"] = "your-key-here"
os.environ["GOOGLE_API_KEY"] = "your-key-here"
```

Open and execute the Jupyter notebook:
```bash
jupyter notebook RAG_Two_Stage_Retrieval.ipynb
```

Expected runtime is approximately 20 to 25 minutes. Results are saved to experiment_results.json and results.csv.

## Methodology

The experimental methodology follows a controlled design comparing five retrieval methods under identical conditions. Each method retrieves the top 5 documents for each test question. Retrieved documents are concatenated to form context for answer generation. Generated answers are evaluated by an independent language model judge using strict grading standards that penalize incomplete or inaccurate responses.

Statistical analysis employs paired t-tests to assess significance of performance differences, with domain serving as blocking variable to examine consistency of improvements across evaluation conditions. Performance consistency is quantified through standard deviation and improvement range analysis across the three domains.

## Results Summary

**Overall Performance:**

Two-Stage method demonstrates consistent improvements of 10 to 13 percent over Hybrid RRF baseline across all domains, with narrow 2-percentage-point variation demonstrating exceptional consistency. Baseline methods exhibit substantially more variable performance depending on evaluation conditions.

**Domain-Specific Patterns:**

- Medical domain: Two-Stage achieves 12 percent improvement over Hybrid RRF
- Finance domain: Two-Stage achieves 11 percent improvement over Hybrid RRF  
- Politics domain: Two-Stage achieves 10 percent improvement over Hybrid RRF

**Statistical Significance:**

All comparisons show highly significant differences with p-values below 0.001, providing strong evidence that observed improvements are not due to random variation. Domain-specific tests confirm that Two-Stage advantages generalize consistently rather than reflecting domain-specific effects.



## Acknowledgments

This work builds upon several foundational contributions:

- MedQuad dataset: Abacha and Demner-Fushman (2019)
- Finance-Alpaca dataset: Taori et al. (2023)
- XSUM dataset: Narayan et al. (2018)
- Sentence-BERT framework: Reimers and Gurevych (2019)
- Cross-encoder reranking: Nogueira and Cho (2019)
- Reciprocal Rank Fusion: Cormack et al. (2009)

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Contact

For questions or collaboration opportunities, please contact [your email] or open an issue in this repository.

## Notes

This implementation is provided for research and educational purposes. API access to OpenAI (GPT-4o Mini) and Google (Gemini 2.5 Flash) is required to run the complete experimental pipeline. API keys should be kept confidential and not committed to version control.
