# Advanced NLP (SCIA / ANLP1 & ANLP2)

![Banner](static/github_anlp_banner.png)

## Objectives
1. Knowlegde: Understand how the main components of modern NLP pipelines work.
2. Hands-on Practice: Develop the ability to build NLP systems for most major NLP tasks (classification, question-answering, ChatGPT-like assistant...).
3. Critical Thinking: Understand the flaws and challenges of NLP, and come up with creative ideas to improve current systems.

## Course organization

Part 1: General NLP 

* **Session 1 (3/10, Célia)**: Recap
* **Session 2 (17/10, Francis)**: Tokenization
* **Session 3 (24/10, Francis)**: Language Modeling
* **Session 4 (31/10, Francis)**: Modern NLP with limited resources
* **Session 5 (7/11, Francis)**: Modern Interpretability 
* **Session 6 (14/11, Francis/Célia)**: Midterm project session

Part 2: Advanced NLP Applications

* **Session 7 (21/11, Célia)**: Safety, Ethics, and Alignment of LMs
* **Session 8 (28/11, Rian)**: Advanced NLP Tasks
* **Session 9 (5/12, Rian)**: Domain-specific NLP
* **Session 10 (12/12, Rian)**: Multilingual NLP 
* **Session 11 (19/12, Célia)**: Multimodal NLP
* **Session 12 (16/01, Francis/Célia/Rian)**: Final Presentations!

## Materials

1. Recap on Deep Learning & basic NLP ([slides](https://github.com/Madjakul/AdvancedNLP/blob/main/slides/pdf/course1_recap.pdf) / [lab session](https://colab.research.google.com/drive/1_QzQBdP289benS8Uo3yPQmtXoM-f80-n?usp=sharing) / [lab correction](https://colab.research.google.com/drive/1Kql7UC4Y5vXt3X8ptyMhHquZWBAehTfQ?usp=share_link) )
2. Tokenization ([slides](https://github.com/Madjakul/AdvancedNLP/raw/main/slides/pdf/course2_tokenization.pdf) / [lab session](https://colab.research.google.com/drive/1P26eLfcl3xVszeTtU5kOjnM4Cd-ZO5bf?usp=sharing))
3. Language Modeling ([slides](https://github.com/Madjakul/AdvancedNLP/raw/main/slides/pdf/course3_lm.pdf) / [lab session](https://arena-chapter1-transformer-interp.streamlit.app/[1.1]_Transformer_from_Scratch))
4. Efficient NLP ([slides](https://github.com/Madjakul/AdvancedNLP/raw/main/slides/pdf/course4_efficiency.pdf) / [lab session](https://arena-chapter1-transformer-interp.streamlit.app/[1.1]_Transformer_from_Scratch))
5. Modern Interpretability ([slides](https://github.com/Madjakul/AdvancedNLP/raw/main/slides/pdf/course5_interpret.pdf) / [lab session](https://arena-chapter1-transformer-interp.streamlit.app/[1.2]_Intro_to_Mech_Interp))
6. Safety, Ethics, and Alignment of LMs ([slides](https://drive.google.com/file/d/1nwDUvYEKng-wlrC140z_D0fIpyVwc6DZ/view?usp=share_link) / [lab session](https://colab.research.google.com/drive/1HQRQRRm_AJcwaEovHW4UU9k8h45wsfs5?usp=sharing))
7. Advanced NLP tasks ([slides](https://github.com/Madjakul/AdvancedNLP/raw/main/slides/pdf/course7_advanced.pdf) / [lab session](https://colab.research.google.com/drive/1Xs0c4YJYEiBuEm_M7oD8U4CQJcOoD0xK?usp=sharing))
8. Domain-specific NLP ([slides](https://github.com/Madjakul/AdvancedNLP/raw/main/slides/pdf/course8_specific.pdf) / [lab session](https://colab.research.google.com/drive/1ZmW0J9GIztl4J2F884LU4YvFp0i7B_Ok?usp=sharing))
9. Multilingual NLP ([slides](https://github.com/Madjakul/AdvancedNLP/blob/main/slides/pdf/Course%209%20-%20Multilingual%20NLP.pdf) / [lab session]())
10. Multimodal NLP ([slides](https://github.com/Madjakul/AdvancedNLP/blob/main/slides/pdf/cours10_Multimodal_NLP.pdf) / [lab session]())

## Evaluation

The evaluation consists in a team project (3-5 people). The choice of the subject is **free** but needs to follow some basic rules:

- Obviously, the project must be highly related with NLP and especially with the notions we will cover in the course
- You can only use open-source LLM that _you serve yourself_. In other words, no API / ChatGPT-like must be used, except for final comparison with your model.
- You must identify and address a <ins>challenging</ins> problem (e.g. not only _can a LLM do X?_, but _can a LLM <ins>that runs on a CPU</ins> do X?_, or _can I make a LLM <ins>better</ins> at X?_)
- It must be reasonably doable: you will not be able to fine-tune (even to use) a 405B parameters model, or to train a model from scratch. That's fine, there are a lot of smaller models that should be good enough, like [the Pythia models](https://huggingface.co/collections/EleutherAI/pythia-scaling-suite-64fb5dfa8c21ebb3db7ad2e1), [TinyLLama](https://huggingface.co/collections/TinyLlama/tinyllama-11b-v1-660bb5bfabd8bd25eebbb1ef), the 1B parameter [OLMo](https://huggingface.co/collections/allenai/olmo-2-674117b93ab84e98afc72edc), or the small models from the [Llama3.2 suite](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf).

:alarm_clock: The project follows 3 deadlines:

- **Project announcement (before 2025/10/17)**: send an email to `francis.kulumba@inria.fr` with cc's `celia.nouri@inria.fr` and `rian.touchent@inria.fr` explaining
  - The team members (also cc'ed)
  - A small description of the project (it can change later on)
- **Project proposal (25% of final grade, before 2025/11/21)**: following [this template](https://docs.google.com/document/d/1JxkbE73ti4vjbIFcBfQpTOT0RfxTInotKyFf4Rglzi8/edit?usp=sharing), produce a project proposal explaining first attempts (e.g. version alpha), how they failed/succeeded and what you want to do before the delivery.
- **Project delivery (75% of final grade, 2026/01/09)**: delivery of a short report detailing each person's contributions, a GitHub repo with an explanatory README + oral presentation on **January 16th**

## Inspiring articles

### Tokenization

- A Vocabulary-Free Multilingual Neural Tokenizer for End-to-End Task Learning (https://arxiv.org/abs/2204.10815)
- BPE-Dropout: Simple and Effective Subword Regularization (https://aclanthology.org/2020.acl-main.170/)
- FOCUS: Effective Embedding Initialization for Monolingual Specialization of Multilingual Models (https://aclanthology.org/2023.emnlp-main.829/)

### Fast inference

- Efficient Streaming Language Models with Attention Sinks (https://arxiv.org/abs/2309.17453)
- Lookahead decoding (https://lmsys.org/blog/2023-11-21-lookahead-decoding/)
- Efficient Memory Management for Large Language Model Serving with PagedAttention (https://arxiv.org/pdf/2309.06180.pdf)

### Inference-time scaling (OpenAI's o1 model)

- Chain-of-Thought Prompting Elicits Reasoning in Large Language Models (https://arxiv.org/abs/2201.11903)
- Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters (https://arxiv.org/abs/2408.03314v1)

### LLM detection

- Detecting Pretraining Data from Large Language Models (https://arxiv.org/abs/2310.16789)
- Proving Test Set Contamination in Black Box Language Models (https://arxiv.org/abs/2310.17623)

### SSMs (off-program)

- Mamba: Linear-Time Sequence Modeling with Selective State Spaces (https://arxiv.org/abs/2312.00752)

### Alignment & Safety

- Null It Out: Guarding Protected Attributes by Iterative Nullspace Projection (https://aclanthology.org/2020.acl-main.647/)
- Direct Preference Optimization: Your Language Model is Secretly a Reward Model (https://arxiv.org/abs/2305.18290)
- Text Embeddings Reveal (Almost) As Much As Text (https://arxiv.org/abs/2310.06816)
