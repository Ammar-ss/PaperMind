"""
Research Paper Q&A — Agentic AI Assistant
agent.py  |  Capstone Project  |  Agentic AI Course 2026
Domain  : Research Paper Q&A
User    : PhD students, researchers, academics
Tool    : Datetime + simple arithmetic calculator
"""

import re
from datetime import datetime
from typing import TypedDict, List

# ── LangGraph ────────────────────────────────────────────────────────────────
from langgraph.graph import StateGraph, END
try:
    from langgraph.checkpoint.memory import MemorySaver
except ImportError:
    from langgraph.checkpoint import MemorySaver          # older langgraph

# ── LangChain / Groq ─────────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# ── Embeddings + Vector DB ───────────────────────────────────────────────────
from sentence_transformers import SentenceTransformer
import chromadb

# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

FAITHFULNESS_THRESHOLD = 0.7
MAX_EVAL_RETRIES       = 2
GROQ_MODEL             = "llama-3.3-70b-versatile"
EMBED_MODEL            = "all-MiniLM-L6-v2"

# ═════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE  — 12 research-domain documents (100-500 words each)
# ═════════════════════════════════════════════════════════════════════════════

DOCUMENTS = [
    {
        "id": "doc_001",
        "topic": "How to Read a Research Paper Efficiently",
        "text": (
            "Reading a research paper efficiently is a skill every researcher must develop. "
            "The most widely recommended approach is the three-pass method. In the first pass, "
            "spend five to ten minutes reading only the title, abstract, introduction, section "
            "headings, and conclusion. This gives you a high-level understanding of what the "
            "paper is about and whether it is relevant to your work. Do not read any "
            "mathematical derivations or experimental details yet. At the end of the first pass "
            "you should be able to answer: What category of paper is this? What is the context? "
            "What are the main contributions? Is the paper well-written?\n\n"
            "In the second pass, read the paper more carefully but skip proofs and detailed "
            "derivations. Pay close attention to figures, graphs, and tables — they often "
            "contain the most important findings. Mark any terms you do not understand and "
            "identify the key references. This pass should take about one hour for an "
            "experienced reader. After this pass you should be able to summarize the paper's "
            "main ideas with supporting evidence.\n\n"
            "The third pass is for fully understanding the paper. Try to virtually re-implement "
            "the paper: make the same assumptions and reconstruct the work step by step. This "
            "deep reading can take four to five hours for beginners. It helps you identify "
            "implicit assumptions, missing citations, and potential issues with the methodology.\n\n"
            "Practical tips: always start with the abstract and conclusion together before "
            "committing to the full paper. Keep a reading log with one-paragraph summaries. "
            "Use reference managers like Zotero or Mendeley. Never read a paper without a "
            "specific question in mind — ask yourself what you hope to extract before you begin."
        )
    },
    {
        "id": "doc_002",
        "topic": "Understanding the Abstract of a Research Paper",
        "text": (
            "The abstract is the most important section of a research paper. It is a "
            "self-contained summary that appears before the main body and must convey the "
            "entire contribution in 150 to 300 words. A well-structured abstract follows five "
            "elements: motivation (why does this problem matter?), problem statement (what "
            "specific problem is being solved?), approach (what method or solution is proposed?), "
            "results (what did the experiments show?), and conclusions (what is the broader impact?).\n\n"
            "When reading an abstract, focus on identifying these five elements. Many researchers "
            "decide whether to read a full paper based solely on the abstract. If the abstract is "
            "vague or does not clearly state the contribution, that is often a sign of a weaker paper.\n\n"
            "There are two types of abstracts. A descriptive abstract tells what the paper covers "
            "without giving results — common in humanities. An informative abstract gives a complete "
            "mini-version of the paper including results and conclusions — standard in science and "
            "engineering, including AI and medical imaging.\n\n"
            "Key things to extract from an abstract: the specific research problem, the proposed "
            "method name if any, the dataset or benchmark used, the main evaluation metric and "
            "performance number, and the claimed improvement over prior work. For example, an "
            "abstract might state that a proposed transformer-based model achieves 92.3 percent "
            "Dice score on a standard benchmark, outperforming the previous state of the art by "
            "2.1 percent. From this single sentence you can extract the method type, task, dataset, "
            "metric, and improvement.\n\n"
            "Abstracts also reveal the writing style and clarity of the authors, which often "
            "reflects the paper's overall quality. A strong abstract uses precise language and "
            "avoids vague terms like 'promising' or 'good results' without quantification."
        )
    },
    {
        "id": "doc_003",
        "topic": "Introduction and Problem Statement in Research Papers",
        "text": (
            "The introduction section convinces the reader that the problem is important and that "
            "the authors' approach is novel. A well-structured introduction follows the CARS model "
            "— Create a Research Space. This means: first establishing that the field is important "
            "(claiming centrality), then identifying a gap or problem in existing work (identifying "
            "a niche), and finally presenting the paper as filling that gap (occupying the niche).\n\n"
            "The problem statement is the core of the introduction. It should be specific, "
            "measurable, and clearly differentiated from prior work. A weak problem statement says "
            "'medical image segmentation is difficult.' A strong problem statement says 'existing "
            "CNN-based methods for fetal ultrasound segmentation fail when images are acquired with "
            "different ultrasound protocols, because domain shift between protocols reduces Dice "
            "score by up to 15%.'\n\n"
            "The introduction also previews the contributions. Most papers list contributions as "
            "bullet points near the end of the introduction, using phrases like 'our main "
            "contributions are as follows.' These contribution claims are what reviewers check "
            "against the experimental results when evaluating the paper.\n\n"
            "Research gaps take several forms: a problem no one has solved, an existing solution "
            "that works on one dataset but not others, an efficient method that sacrifices accuracy, "
            "or an accurate method too slow for real-world deployment. Understanding which type of "
            "gap the paper addresses helps you evaluate whether the proposed solution is appropriate.\n\n"
            "The introduction usually ends with a paragraph describing the paper's organization — "
            "listing what each subsequent section covers. This roadmap is useful for navigating "
            "directly to the most relevant parts of a long paper."
        )
    },
    {
        "id": "doc_004",
        "topic": "Research Methodology Types in Computer Science Papers",
        "text": (
            "Research methodology describes how a study is conducted. Understanding the methodology "
            "section is crucial for evaluating whether a paper's conclusions are valid. In computer "
            "science, there are three broad categories of research papers: experimental papers, "
            "theoretical papers, and survey papers.\n\n"
            "Experimental papers propose a new method and evaluate it on one or more benchmark "
            "datasets. The methodology section describes the model architecture, training procedure, "
            "hyperparameters, and evaluation protocol. When reading this section, check whether the "
            "baseline methods are recent and fairly implemented, whether the same train/test split "
            "is used for all methods, whether the evaluation metric is appropriate for the task, "
            "and whether ablation studies are included to isolate each component's contribution.\n\n"
            "Theoretical papers prove mathematical properties of algorithms or models. The "
            "methodology is a formal proof or analysis. Check whether all assumptions are stated "
            "explicitly, whether the proof covers all edge cases, and whether the theoretical "
            "result is tight — meaning it closely matches empirical behavior.\n\n"
            "Survey papers review and categorize existing work on a topic. The methodology "
            "describes how papers were selected and how they are organized into a taxonomy. Check "
            "whether the search process is described with specific databases and keywords, whether "
            "coverage is comprehensive, and whether the taxonomy makes logical sense.\n\n"
            "Ablation studies are particularly important in deep learning papers. An ablation "
            "removes one component at a time to measure its contribution. If a model has components "
            "A, B, and C, the ablation tests A only, then A plus B, then A plus B plus C. This "
            "confirms that each component actually contributes. Papers without ablation studies "
            "are harder to trust because you cannot identify which part drives the improvement.\n\n"
            "Reproducibility is a key concern. The methodology should provide enough detail to "
            "re-implement the work. Missing hyperparameters, undisclosed preprocessing steps, "
            "or unavailable code are significant red flags."
        )
    },
    {
        "id": "doc_005",
        "topic": "Understanding Results, Metrics, and Evaluation in CS Papers",
        "text": (
            "The results section presents experimental findings, and understanding it requires "
            "knowing what the evaluation metrics mean. Different tasks use different metrics, and "
            "choosing the wrong metric can make a mediocre method appear strong.\n\n"
            "In classification tasks, accuracy measures overall correctness but is misleading on "
            "imbalanced datasets. Precision measures what fraction of positive predictions are "
            "truly positive. Recall, also called sensitivity, measures what fraction of actual "
            "positives were correctly detected. F1-score is the harmonic mean of precision and "
            "recall, used when both matter equally. AUC-ROC measures the model's discrimination "
            "ability across all decision thresholds.\n\n"
            "In segmentation tasks common in medical imaging, the Dice Similarity Coefficient "
            "(DSC) measures overlap between a predicted mask and the ground truth. A Dice score "
            "of 1.0 represents perfect overlap and 0.0 represents no overlap. Intersection over "
            "Union (IoU), also called the Jaccard index, is similar and widely used in computer "
            "vision. Hausdorff Distance measures the maximum surface distance between predicted "
            "and ground truth contours — lower values are better.\n\n"
            "In natural language processing, BLEU score measures n-gram overlap between generated "
            "and reference text, used for translation and summarization. ROUGE measures "
            "recall-oriented overlap and is common in summarization. Perplexity measures how well "
            "a language model predicts text — lower perplexity indicates a better model.\n\n"
            "When reading results tables, verify: what is the baseline? Is the improvement "
            "statistically significant with a reported p-value or confidence interval? Are results "
            "averaged over multiple independent runs? Is the test set truly independent from the "
            "validation set? A single-run result without any variance measure is a weak claim.\n\n"
            "State-of-the-art comparisons should include the most recent competitive methods. "
            "If a paper only compares against methods from two or more years ago, the claimed "
            "improvement may not hold against current work."
        )
    },
    {
        "id": "doc_006",
        "topic": "Literature Review and Related Work Section",
        "text": (
            "The related work section situates a paper within the existing body of research. It "
            "explains what has been done before, what the limitations of prior work are, and how "
            "the current paper differs. Reading this section carefully helps you build a map of "
            "the field and identify the most important prior papers you should read.\n\n"
            "A well-written related work section is organized thematically, not chronologically. "
            "Instead of listing papers one by one, it groups them by approach or category. For "
            "example: CNN-based methods focus on local feature extraction, while transformer-based "
            "methods capture global context. This grouping shows the authors understand the "
            "landscape of the field.\n\n"
            "When reading related work, look for: papers the authors cite most frequently (these "
            "are likely foundational), methods described as the closest prior work (these are "
            "likely the strongest baselines), and limitations attributed to prior work (these "
            "should be directly addressed in the proposed method).\n\n"
            "A common structure for related work in deep learning papers covers: classical "
            "approaches from before deep learning, CNN-based approaches, attention and "
            "transformer-based approaches, and task-specific methods. If a paper skips over an "
            "entire important category without explanation, that may indicate the authors are "
            "avoiding comparison with strong competitors.\n\n"
            "Literature gaps drive contributions. A gap can be: a problem that has not been "
            "studied at all, a problem studied only on small or single-site datasets, a method "
            "that works in one domain but fails in another, or a computationally expensive method "
            "that has not been made efficient for deployment. The paper's stated contribution "
            "should directly address one or more of these gap types.\n\n"
            "For your own research, use the related work section as a curated reading list. "
            "The papers cited most frequently across multiple papers you read are the foundational "
            "works you should prioritize."
        )
    },
    {
        "id": "doc_007",
        "topic": "Citation Formats: IEEE, APA, and ACM Styles",
        "text": (
            "Research papers use different citation formats depending on the field and publication "
            "venue. In computer science and engineering, the three most common formats are IEEE, "
            "ACM, and APA.\n\n"
            "IEEE format is the standard for engineering and most CS conferences and journals. "
            "In-text citations use numbers in square brackets such as [1] or [2,3]. The reference "
            "list is ordered by order of appearance in the text. A journal article in IEEE format "
            "lists author initials before surname, article title in quotation marks, journal name "
            "in italics, volume and issue number, page range, and month and year. For example: "
            "A. Sharma and B. Patel, 'Deep learning for image segmentation: A review,' IEEE "
            "Transactions on Medical Imaging, vol. 40, no. 3, pp. 512-525, Mar. 2021.\n\n"
            "ACM format is used by most ACM-sponsored conferences and publications. Depending on "
            "the template, it uses either author-year citations in parentheses such as (Smith, "
            "2021) or numbered citations. The reference list is ordered alphabetically by author "
            "surname.\n\n"
            "APA format is common in psychology and social sciences but also used in "
            "interdisciplinary and some AI journals. In-text citation format is author and year "
            "in parentheses. In the reference list, the surname comes first followed by initials, "
            "the year is in parentheses after the author name, and the article title uses sentence "
            "case while the journal name is italicized with title case.\n\n"
            "For conference papers, the venue name replaces the journal name. The proceedings "
            "title, conference location, and year are included. Most publishers provide BibTeX "
            "entries on their websites that can be imported directly into reference managers "
            "like Zotero, Mendeley, or Papers. Always verify auto-generated citations for "
            "accuracy before submitting, as they sometimes contain errors in author names, "
            "page numbers, or venue details."
        )
    },
    {
        "id": "doc_008",
        "topic": "Academic Publication Venues: Journals vs Conferences and Impact Factor",
        "text": (
            "In computer science and AI, understanding publication venues is essential for judging "
            "the quality and significance of a paper. The field is unusual among academic "
            "disciplines in that top conferences are often considered more prestigious than "
            "many journals.\n\n"
            "Top-tier conferences in AI and machine learning include NeurIPS, ICML, ICLR, CVPR, "
            "ICCV, ECCV, ACL, EMNLP, and AAAI. Acceptance rates at these venues typically range "
            "from 15 to 30 percent, with rigorous peer review. A paper accepted at NeurIPS or "
            "CVPR is generally considered high quality by the research community. Medical imaging "
            "specifically has MICCAI — the Medical Image Computing and Computer-Assisted "
            "Intervention conference — as its premier annual venue.\n\n"
            "Journals offer several advantages over conferences: longer papers with more "
            "experimental detail, no strict page limits, revision cycles that allow authors to "
            "address reviewer concerns, and permanent archival with a fixed DOI. Top journals in "
            "AI and medical imaging include Nature Machine Intelligence, IEEE Transactions on "
            "Medical Imaging, Medical Image Analysis, Journal of Machine Learning Research, and "
            "IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI).\n\n"
            "Impact Factor is a journal-level metric published annually. It measures the average "
            "number of citations received per paper published in that journal over the previous "
            "two years. Medical Image Analysis has an impact factor of approximately 10 to 13, "
            "which is very high for the field. Nature Machine Intelligence exceeds an impact "
            "factor of 20.\n\n"
            "ArXiv is a preprint server where authors share papers before or alongside peer "
            "review. Most major AI results appear on arXiv before their official conference "
            "publication. ArXiv papers are not peer-reviewed, but they are the primary way "
            "researchers stay current with the field.\n\n"
            "Predatory venues accept papers without real review in exchange for fees. Signs "
            "include unrealistically fast review times, requests for unusual fees, and names "
            "that closely mimic legitimate venues."
        )
    },
    {
        "id": "doc_009",
        "topic": "Research Metrics: H-Index, Citation Count, and Academic Impact",
        "text": (
            "Research metrics quantify the impact and productivity of researchers and their work. "
            "Understanding these metrics helps you evaluate the credibility of authors and "
            "identify the most influential papers in a field.\n\n"
            "Citation count is the most direct measure of a paper's impact — it counts how many "
            "other papers have cited it. A paper with 500 or more citations is considered highly "
            "influential. Citation counts are tracked by Google Scholar, Semantic Scholar, "
            "Scopus, and Web of Science. A recent paper will naturally have fewer citations than "
            "an older one, so always consider the year of publication when interpreting counts.\n\n"
            "The h-index is a researcher-level metric. A researcher has an h-index of h if h of "
            "their papers have been cited at least h times each. For example, an h-index of 30 "
            "means the researcher has 30 papers each cited at least 30 times. The h-index "
            "balances productivity with impact. For PhD students applying to top programs, an "
            "h-index of 2 to 5 is already noteworthy. Senior faculty at leading universities "
            "often have h-indices between 40 and 100.\n\n"
            "The i10-index, used by Google Scholar, counts the number of publications with at "
            "least 10 citations. It is a simple measure of consistent research output.\n\n"
            "Impact Factor applies to journals, not individual papers. A high impact factor for "
            "the journal does not guarantee that any specific paper in it is highly cited, and "
            "a highly cited paper can appear in a lower-impact journal.\n\n"
            "CiteScore is Elsevier's alternative to Impact Factor, calculated over four years "
            "instead of two. SCImago Journal Rank (SJR) weights citations by the prestige of "
            "the citing journal, so a citation from Nature counts more than one from a low-tier "
            "journal.\n\n"
            "Google Scholar is freely accessible and indexes arXiv preprints, making it the "
            "primary tool for most AI researchers when checking citation counts or finding "
            "related work. Semantic Scholar provides additional metrics including citation "
            "velocity and influence scores."
        )
    },
    {
        "id": "doc_010",
        "topic": "The Peer Review Process in Academic Publishing",
        "text": (
            "The peer review process is the quality-control mechanism of academic publishing. "
            "When a paper is submitted to a conference or journal, the editor or area chair "
            "assigns it to two to four expert reviewers. Reviewers are researchers in the same "
            "field who evaluate the paper for novelty, technical correctness, clarity, and "
            "significance. Most top CS conferences use double-blind review: reviewers do not "
            "know the authors' identities, and authors do not know who reviewed their work.\n\n"
            "A typical review contains: a summary of the paper's contribution, a list of "
            "strengths, a list of weaknesses, specific questions or clarification requests for "
            "the authors, and an overall recommendation such as accept, weak accept, borderline, "
            "weak reject, or reject. After all reviews are received, there is usually an author "
            "rebuttal period during which authors can respond to reviewer concerns. An area chair "
            "then synthesizes the reviews and makes a recommendation to the program chairs.\n\n"
            "Acceptance rates vary by venue. Major AI and machine learning conferences accept "
            "between 15 and 30 percent of submissions. MICCAI accepts approximately 30 percent "
            "of submissions. Journal acceptance rates after revision are often between 20 and "
            "50 percent.\n\n"
            "Journal reviews involve revision cycles. A major revision requires significant "
            "changes and another round of review. A minor revision requires small changes and "
            "is decided by the editor. Most strong papers require at least one round of "
            "revision before acceptance.\n\n"
            "Common rejection reasons include: insufficient novelty compared to prior work, "
            "missing ablation studies, unfair comparison to baselines, overclaimed results "
            "without statistical support, poor writing and unclear presentation, or missing "
            "citations to key related work.\n\n"
            "OpenReview.net hosts the full review history for ICLR papers and increasingly for "
            "NeurIPS and other venues. Reading accepted and rejected paper reviews is one of "
            "the most effective ways to develop critical evaluation skills."
        )
    },
    {
        "id": "doc_011",
        "topic": "IMRaD Structure: The Standard Research Paper Format",
        "text": (
            "Most scientific papers follow the IMRaD structure: Introduction, Methods, Results, "
            "and Discussion. This format standardizes scientific communication and makes it easy "
            "to locate specific information quickly. Understanding this structure allows you to "
            "navigate any research paper efficiently.\n\n"
            "The Introduction answers: why was this study done? It provides background context, "
            "identifies the research gap, states the research objective or hypothesis, and "
            "outlines the paper's specific contributions.\n\n"
            "The Methods section, also called Methodology or Experimental Setup, answers: how "
            "was the study done? It provides sufficient detail for another researcher to reproduce "
            "the work. In deep learning papers this includes the model architecture with all "
            "components described, the training procedure covering the optimizer, learning rate, "
            "batch size, and number of epochs, data preprocessing and augmentation strategies, "
            "and dataset descriptions including the number of samples, class distribution, and "
            "train/validation/test split ratios.\n\n"
            "The Results section answers: what was found? It presents experimental outcomes "
            "objectively using tables and figures. In machine learning papers, results tables "
            "compare the proposed method against baselines on one or more benchmarks. Ablation "
            "study results quantifying each component's contribution are also presented here.\n\n"
            "The Discussion section answers: what do the results mean? Authors interpret "
            "findings, explain surprising or counterintuitive results, discuss limitations and "
            "failure cases, and suggest directions for future work. A strong discussion "
            "acknowledges where the method fails and why. A weak discussion merely restates "
            "numbers from the results table.\n\n"
            "The Conclusion summarizes the work and its practical implications. Supplementary "
            "materials extend the paper with additional experiments, implementation details, "
            "qualitative examples, or complete mathematical proofs that could not fit within "
            "the page limit."
        )
    },
    {
        "id": "doc_012",
        "topic": "Identifying Research Contributions, Novelty, and Limitations",
        "text": (
            "Identifying a paper's true contribution requires critical reading, not just "
            "accepting the authors' claims at face value. Contributions in research fall into "
            "several categories: a new dataset or benchmark, a new method or architecture, "
            "a new theoretical result, a new evaluation protocol, or a new application of an "
            "existing method to a previously unstudied domain.\n\n"
            "The claimed contributions are listed in the introduction. As a critical reader, "
            "your task is to verify each claim against the experimental evidence. If a paper "
            "claims computational efficiency, check whether runtime comparisons or "
            "floating-point operation counts are reported. If the paper claims state-of-the-art "
            "performance, check whether the comparison includes the most recent prior methods.\n\n"
            "Novelty is the most important criterion in peer review. A contribution is novel "
            "if it meaningfully extends what existed before. Types of novelty include: "
            "algorithmic novelty such as a new architecture component or loss function, "
            "dataset novelty such as a new curated benchmark, application novelty such as "
            "applying known methods to a new domain or clinical problem, and conceptual "
            "novelty such as a theoretical insight that changes how the field thinks about "
            "a problem.\n\n"
            "Limitations are as important as contributions for scientific honesty. All methods "
            "have limitations. Responsible papers state them clearly, usually at the end of "
            "the discussion section. Common limitations in deep learning papers include: "
            "evaluation on only one dataset, requirement for large amounts of labeled training "
            "data, high computational cost at inference time, and lack of evaluation on "
            "out-of-distribution or adversarial data.\n\n"
            "When evaluating a paper for your own research, ask: does this limitation affect "
            "my specific use case? Can this method be extended to address its limitations? "
            "Is the novelty incremental or substantial? A paper that makes one clear, "
            "well-supported contribution is more valuable than one that overclaims and "
            "underdelivers."
        )
    }
]


# ═════════════════════════════════════════════════════════════════════════════
# STATE
# ═════════════════════════════════════════════════════════════════════════════

class CapstoneState(TypedDict):
    question    : str
    messages    : List[dict]
    route       : str
    retrieved   : str
    sources     : List[str]
    tool_result : str
    answer      : str
    faithfulness: float
    eval_retries: int
    user_name   : str


# ═════════════════════════════════════════════════════════════════════════════
# AGENT INITIALISATION
# Returns a compiled LangGraph app ready for invoke() calls.
# ═════════════════════════════════════════════════════════════════════════════

def init_agent(groq_api_key: str):
    """
    Call once at startup. Returns the compiled LangGraph app.
    Pass every user question to  app.invoke({"question": q}, config)
    where config = {"configurable": {"thread_id": <your_thread_id>}}
    """

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm = ChatGroq(api_key=groq_api_key, model_name=GROQ_MODEL, temperature=0)

    # ── Embedder ─────────────────────────────────────────────────────────────
    print("Loading sentence embedder (first run downloads ~90 MB) …")
    embedder = SentenceTransformer(EMBED_MODEL)

    # ── ChromaDB in-memory ───────────────────────────────────────────────────
    client = chromadb.Client()
    try:
        client.delete_collection("research_papers")
    except Exception:
        pass
    collection = client.create_collection("research_papers")

    docs_text  = [d["text"]  for d in DOCUMENTS]
    docs_ids   = [d["id"]    for d in DOCUMENTS]
    docs_meta  = [{"topic": d["topic"]} for d in DOCUMENTS]
    embeddings = embedder.encode(docs_text).tolist()

    collection.add(
        documents=docs_text,
        embeddings=embeddings,
        ids=docs_ids,
        metadatas=docs_meta
    )
    print(f"ChromaDB loaded — {len(DOCUMENTS)} documents indexed ✓")

    # ─────────────────────────────────────────────────────────────────────────
    # NODE  1 : memory_node
    # ─────────────────────────────────────────────────────────────────────────
    def memory_node(state: CapstoneState) -> dict:
        msgs = list(state.get("messages", []))
        msgs.append({"role": "user", "content": state["question"]})
        msgs = msgs[-6:]                              # sliding window — keeps last 6

        user_name = state.get("user_name", "")
        q_lower   = state["question"].lower()
        if "my name is" in q_lower:
            try:
                raw  = q_lower.split("my name is")[-1].strip()
                user_name = raw.split()[0].capitalize()
            except Exception:
                pass

        return {"messages": msgs, "user_name": user_name, "eval_retries": 0}

    # ─────────────────────────────────────────────────────────────────────────
    # NODE  2 : router_node
    # ─────────────────────────────────────────────────────────────────────────
    def router_node(state: CapstoneState) -> dict:
        history_text = "\n".join(
            f"{m['role']}: {m['content']}" for m in state.get("messages", [])[-4:]
        )
        prompt = (
            "You are a router for a Research Paper Q&A assistant.\n\n"
            "Choose exactly ONE route:\n"
            "- retrieve   : question is about research papers, academic writing, citations, "
            "peer review, publication venues, metrics, methodology, how to read papers, "
            "or any academic/research topic\n"
            "- tool       : question asks for the current date, current time, or a "
            "simple arithmetic calculation\n"
            "- memory_only: greeting, small talk, thanks, asks about the user's own name, "
            "or does not need any knowledge lookup\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Current question: {state['question']}\n\n"
            "Reply with ONE word only: retrieve, tool, or memory_only"
        )
        response = llm.invoke([HumanMessage(content=prompt)])
        route    = response.content.strip().lower().split()[0]
        if route not in ("retrieve", "tool", "memory_only"):
            route = "retrieve"
        return {"route": route}

    # ─────────────────────────────────────────────────────────────────────────
    # NODE  3 : retrieval_node
    # ─────────────────────────────────────────────────────────────────────────
    def retrieval_node(state: CapstoneState) -> dict:
        query_emb = embedder.encode([state["question"]]).tolist()
        results   = collection.query(query_embeddings=query_emb, n_results=3)
        chunks    = results["documents"][0]
        metas     = results["metadatas"][0]
        sources   = [m["topic"] for m in metas]
        context   = "\n\n".join(
            f"[{metas[i]['topic']}]\n{chunks[i]}" for i in range(len(chunks))
        )
        return {"retrieved": context, "sources": sources}

    # ─────────────────────────────────────────────────────────────────────────
    # NODE  4 : skip_retrieval_node
    # ─────────────────────────────────────────────────────────────────────────
    def skip_retrieval_node(state: CapstoneState) -> dict:
        return {"retrieved": "", "sources": []}

    # ─────────────────────────────────────────────────────────────────────────
    # NODE  5 : tool_node   (datetime + simple arithmetic)
    # ─────────────────────────────────────────────────────────────────────────
    def tool_node(state: CapstoneState) -> dict:
        try:
            q   = state["question"].lower()
            now = datetime.now()

            if any(w in q for w in ("date", "today", "day")):
                result = f"Today is {now.strftime('%A, %B %d, %Y')}."
            elif any(w in q for w in ("time", "clock", "hour", "minute")):
                result = f"The current time is {now.strftime('%I:%M %p')}."
            else:
                nums = [float(x) for x in re.findall(r"\d+\.?\d*", q)]
                if len(nums) >= 2:
                    a, b  = nums[0], nums[1]
                    orig  = state["question"]
                    if   "plus"    in q or "+"  in orig: result = f"Result: {a + b}"
                    elif "minus"   in q or "-"  in orig: result = f"Result: {a - b}"
                    elif "times"   in q or "*"  in orig or "×" in orig:
                        result = f"Result: {a * b}"
                    elif "divided" in q or "/"  in orig or "÷" in orig:
                        result = (f"Result: {a / b}" if b != 0
                                  else "Cannot divide by zero.")
                    else:
                        result = f"Today is {now.strftime('%A, %B %d, %Y')} at {now.strftime('%I:%M %p')}."
                else:
                    result = f"Today is {now.strftime('%A, %B %d, %Y')} at {now.strftime('%I:%M %p')}."
        except Exception as exc:
            result = f"Tool error (non-fatal): {exc}"
        return {"tool_result": result}

    # ─────────────────────────────────────────────────────────────────────────
    # NODE  6 : answer_node
    # ─────────────────────────────────────────────────────────────────────────
    def answer_node(state: CapstoneState) -> dict:
        user_name    = state.get("user_name", "")
        name_clause  = f" The user's name is {user_name}." if user_name else ""
        retries      = state.get("eval_retries", 0)
        retry_note   = (
            "\n\nIMPORTANT: Your previous answer scored below the faithfulness threshold. "
            "Stay strictly within the provided context. Do not add any information that "
            "is not explicitly stated in the context."
        ) if retries > 0 else ""

        history_text = "\n".join(
            f"{m['role']}: {m['content']}"
            for m in state.get("messages", [])[-4:]
        )
        context_block = ""
        if state.get("retrieved"):
            context_block += f"\n\nKNOWLEDGE BASE CONTEXT:\n{state['retrieved']}"
        if state.get("tool_result"):
            context_block += f"\n\nTOOL RESULT:\n{state['tool_result']}"

        system = (
            f"You are a Research Paper Q&A assistant helping PhD students and researchers "
            f"understand academic papers and research concepts.{name_clause}\n\n"
            "RULES:\n"
            "1. Answer ONLY from the provided context. Never fabricate facts.\n"
            "2. If the context does not contain the answer, say: "
            "'I do not have that information in my knowledge base. "
            "Please check Google Scholar or consult your research supervisor.'\n"
            "3. Be specific, accurate, and helpful.\n"
            "4. Respond warmly to greetings and casual messages."
            f"{retry_note}"
        )

        msgs_to_llm = [
            SystemMessage(content=system),
            HumanMessage(content=(
                f"Conversation so far:\n{history_text}"
                f"{context_block}\n\n"
                f"Answer the user's latest question: {state['question']}"
            ))
        ]
        response = llm.invoke(msgs_to_llm)
        return {"answer": response.content.strip()}

    # ─────────────────────────────────────────────────────────────────────────
    # NODE  7 : eval_node
    # ─────────────────────────────────────────────────────────────────────────
    def eval_node(state: CapstoneState) -> dict:
        if not state.get("retrieved"):          # tool / memory-only — skip check
            return {"faithfulness": 1.0, "eval_retries": state.get("eval_retries", 0)}

        prompt = (
            "Rate the faithfulness of the ANSWER below on a scale 0.0 to 1.0.\n\n"
            "Faithfulness = does the answer use ONLY information present in the CONTEXT, "
            "without adding external facts?\n\n"
            f"CONTEXT:\n{state['retrieved']}\n\n"
            f"ANSWER:\n{state['answer']}\n\n"
            "Respond with a single decimal number between 0.0 and 1.0. Nothing else."
        )
        try:
            resp  = llm.invoke([HumanMessage(content=prompt)])
            score = float(resp.content.strip().split()[0])
            score = max(0.0, min(1.0, score))
        except Exception:
            score = 1.0          # default PASS on parse failure

        return {"faithfulness": score, "eval_retries": state.get("eval_retries", 0) + 1}

    # ─────────────────────────────────────────────────────────────────────────
    # NODE  8 : save_node
    # ─────────────────────────────────────────────────────────────────────────
    def save_node(state: CapstoneState) -> dict:
        msgs = list(state.get("messages", []))
        msgs.append({"role": "assistant", "content": state["answer"]})
        return {"messages": msgs}

    # ─────────────────────────────────────────────────────────────────────────
    # ROUTING FUNCTIONS
    # ─────────────────────────────────────────────────────────────────────────
    def route_decision(state: CapstoneState) -> str:
        r = state.get("route", "retrieve")
        if r == "tool":        return "tool"
        if r == "memory_only": return "skip"
        return "retrieve"

    def eval_decision(state: CapstoneState) -> str:
        if (state.get("faithfulness", 1.0) < FAITHFULNESS_THRESHOLD
                and state.get("eval_retries", 0) < MAX_EVAL_RETRIES):
            return "answer"   # retry
        return "save"

    # ─────────────────────────────────────────────────────────────────────────
    # GRAPH ASSEMBLY
    # ─────────────────────────────────────────────────────────────────────────
    graph = StateGraph(CapstoneState)

    graph.add_node("memory",   memory_node)
    graph.add_node("router",   router_node)
    graph.add_node("retrieve", retrieval_node)
    graph.add_node("skip",     skip_retrieval_node)
    graph.add_node("tool",     tool_node)
    graph.add_node("answer",   answer_node)
    graph.add_node("eval",     eval_node)
    graph.add_node("save",     save_node)

    graph.set_entry_point("memory")
    graph.add_edge("memory", "router")

    graph.add_conditional_edges(
        "router", route_decision,
        {"retrieve": "retrieve", "skip": "skip", "tool": "tool"}
    )

    graph.add_edge("retrieve", "answer")
    graph.add_edge("skip",     "answer")
    graph.add_edge("tool",     "answer")
    graph.add_edge("answer",   "eval")

    graph.add_conditional_edges(
        "eval", eval_decision,
        {"answer": "answer", "save": "save"}
    )

    graph.add_edge("save", END)

    app = graph.compile(checkpointer=MemorySaver())
    print("Graph compiled successfully ✓")
    return app
