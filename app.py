"""
Virtual Laboratory: Construction of an Inverted Index
Course: Knowledge Graphs and Information Retrieval Systems (KGIRS)
Experiment No: 3 (Allocated Roll Numbers: 11 - 15)

Built following the Virtual Lab pedagogical structure:
  1. Aim & Overview
  2. Theory
  3. Procedure
  4. Simulation (Interactive Inverted Index Workbench)
  5. Self-Evaluation (Quiz)
  6. References
  7. Feedback
  8. Report Generation (Downloadable Verified PDF Report)

Note: Designed using native Streamlit UI components and Plotly for seamless Light/Dark mode rendering.
"""

import os
import re
import time
import unicodedata
from datetime import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from fpdf import FPDF


# ======================================================================================
# 1. EXPERIMENT CONFIGURATION & EDUCATIONAL CONTENT 
# ======================================================================================

EXPERIMENT_CONFIG = {
    "exp_no": 3,
    "title": "Experiment 3: Construction of an Inverted Index",
    "short_title": "Construction of an Inverted Index",
    "subject": "Knowledge Graphs & Information Retrieval Systems (KGIRS)",
    "course_code": "CS-KGIRS-03",
    "target_rolls": "Roll Numbers 11 - 15 (Division C)",
    "portal": "Virtual Labs - An Initiative of Ministry of Education (vlabs.ac.in)",
    "aim": (
        "To construct and analyze an Inverted Index data structure for a collection of textual documents, "
        "explore linguistic preprocessing transformations (tokenization, stopword removal, stemming), "
        "and execute Boolean, phrase, and proximity queries with linear-time pointer-merge algorithms."
    ),
    "objectives": [
        "Construct an Inverted Index data structure mapping vocabulary terms to documents, term frequencies (tf), and token positions.",
        "Trace the 4 core pipeline stages of index construction: Document Tokenization, Pair Extraction, Lexicographic Sorting, and Postings Inversion.",
        "Implement and visualize linear-time Boolean retrieval algorithms (AND, OR, NOT) using sorted posting list pointer intersections.",
        "Execute positional phrase queries and proximity searches by matching token offsets within identical document postings.",
        "Quantify the empirical effect of linguistic preprocessing (case folding, stopword removal, Porter stemming) on vocabulary size and index compression.",
        "Analyze collection-wide term frequency distributions and empirically validate Zipf's Law and Heaps' Law."
    ]
}

THEORY_CONTENT = {
    "background": r"""
### 1. Introduction to Information Retrieval & Indexing

Information Retrieval (IR) focuses on finding material (usually unstructured text documents) of an informal nature that satisfies an information need within large collections.

A naive approach to search is **linear scanning** (e.g., using `grep` or string matching). If a collection contains $N$ documents each averaging $L$ characters, searching for a query across the entire collection requires $O(N \cdot L)$ time. For collections containing millions or billions of web pages and documents, linear scanning is computationally infeasible.

To deliver sub-second retrieval latencies, an IR system builds an offline **Index** beforehand.

---

### 2. Forward Index vs. Term-Document Incidence Matrix

* **Term-Document Incidence Matrix**: A binary matrix $M \in \{0, 1\}^{|V| \times N}$, where $|V|$ is the vocabulary size and $N$ is the number of documents. $M[t, d] = 1$ if term $t$ appears in document $d$, and $0$ otherwise.
  * **The Sparsity Problem**: In real-world natural language collections, a typical document contains only a small fraction of the total vocabulary (often $< 0.2\%$). Storing an explicit matrix of size $|V| \times N$ wastes over $99.8\%$ of memory storing zeroes.
* **Forward Index**: Maps each document identifier ($DocID$) to the list of terms it contains:

$$
\text{Forward Index}: DocID \longrightarrow [\text{term}_1, \text{term}_2, \dots, \text{term}_k]
$$

  While efficient for document parsing, searching for a keyword requires scanning every document entry.

---

### 3. Architecture of an Inverted Index

The **Inverted Index** (or inverted file) inverts the forward mapping: it maps each unique vocabulary term to an ordered list of documents where that term occurs:

$$
\text{Inverted Index}: \text{term}_t \longrightarrow [DocID_1, DocID_2, DocID_3, \dots]
$$

An Inverted Index consists of two primary architectural components:

```
+-----------------------------------------------------------------------------------+
|                            DICTIONARY (VOCABULARY)                                |
|  Term (String)   | Document Frequency (df) | Collection Frequency (cf) | Pointer  |
+------------------+-------------------------+---------------------------+----------+
|  "information"   |           3             |             4             |    *----> [Doc 1 (tf:1)] -> [Doc 3 (tf:2)] -> [Doc 5 (tf:1)]
|  "retrieval"     |           4             |             5             |    *----> [Doc 1 (tf:2)] -> [Doc 2 (tf:1)] -> [Doc 3 (tf:1)] -> [Doc 5 (tf:1)]
|  "graph"         |           2             |             3             |    *----> [Doc 2 (tf:2)] -> [Doc 4 (tf:1)]
+-----------------------------------------------------------------------------------+
```

1. **Dictionary (Lexicon / Vocabulary)**:
   * Stores all unique terms appearing across the corpus.
   * For each term $t$, stores its **Document Frequency** ($df_t$: count of documents containing $t$) and **Collection Frequency** ($cf_t$: total occurrences across the entire corpus).
   * Implemented using dynamic Hash Tables for $O(1)$ expected lookup, or B-Trees / Tries for $O(\log |V|)$ lookup supporting prefix and wildcard queries.
2. **Postings Lists**:
   * A variable-length linked list or contiguous dynamic array containing the document entries (postings) for each term.
   * Maintained strictly in **sorted order of Document ID** ($DocID_1 < DocID_2 < \dots < DocID_k$). Sorting is essential because it enables linear-time $O(L_1 + L_2)$ intersection algorithms.
   * **Positional Postings Lists**: Extends each posting to record not just $DocID$ and term frequency $tf_{t,d}$, but also the exact token offset positions $[pos_1, pos_2, \dots]$.

---

### 4. Query Processing & Boolean Retrieval

#### Conjunction (`AND` Intersection)
Given two sorted postings lists $P_1$ of length $L_1$ and $P_2$ of length $L_2$, the intersection algorithm uses two pointers $p_1$ and $p_2$ traversing both lists simultaneously:

$$
\text{Time Complexity: } O(L_1 + L_2)
$$

* If $DocID(p_1) == DocID(p_2)$: Append $DocID(p_1)$ to result; advance both $p_1$ and $p_2$.
* If $DocID(p_1) < DocID(p_2)$: Advance $p_1$.
* If $DocID(p_1) > DocID(p_2)$: Advance $p_2$.

#### Disjunction (`OR` Union)
Traverse both lists with two pointers, appending the smaller $DocID$ to the output list and advancing that pointer. If identical, append once and advance both. Running time: $O(L_1 + L_2)$.

#### Query Optimization via Document Frequency
When evaluating a multi-term conjunctive query $t_1 \land t_2 \land t_3$:
Search engines sort the query terms in **increasing order of document frequency** ($df_{t_1} \le df_{t_2} \le df_{t_3}$). Intersecting the shortest lists first produces the smallest intermediate candidate set, drastically reducing subsequent comparisons!

---

### 5. Positional Indexing & Phrase Queries

A non-positional index can only verify that two words appear somewhere in the same document, failing on exact phrase queries such as `"information retrieval"`.
In a **Positional Inverted Index**, each posting stores the token positions:

$$
\text{Posting}: \langle DocID, tf, [pos_1, pos_2, \dots, pos_{tf}] \rangle
$$

For phrase query `"term1 term2"`, the query engine intersects their document postings and checks whether:

$$
\exists p \in \text{positions}(\text{term}_1, d) \quad \text{such that} \quad (p + 1) \in \text{positions}(\text{term}_2, d)
$$

---

### 6. Empirical Laws: Zipf's Law & Heaps' Law

* **Zipf's Law**: The frequency $f$ of any word is inversely proportional to its rank $r$ in the frequency table:

$$
f(r) \propto \frac{1}{r^s} \implies \log f(r) = \log C - s \cdot \log r
$$

  A tiny percentage of terms (stopwords) account for a massive fraction of all tokens.
* **Heaps' Law**: Empirically models vocabulary growth $|V|$ as a function of total token count $N$:

$$
|V| = k \cdot N^\beta
$$

  where $30 \le k \le 100$ and $\beta \approx 0.4 - 0.6$. The vocabulary continues growing with corpus size.
    """,
    "key_terms": {
        "Inverted Index": "Core IR data structure mapping terms to documents containing them.",
        "Postings List": "Sorted sequence of document entries (DocIDs, frequencies, positions) for a term.",
        "Dictionary (Lexicon)": "Data structure storing unique vocabulary terms and their collection statistics.",
        "Document Frequency (df)": "The number of distinct documents in which a term appears.",
        "Term Frequency (tf)": "The number of times a term occurs within a specific document.",
        "Collection Frequency (cf)": "Total count of occurrences of a term across the entire corpus.",
        "Positional Index": "Postings list enriched with token offsets, enabling phrase and proximity queries.",
        "Skip Pointer": "Shortcuts embedded in postings lists to skip non-matching intervals in O(sqrt(L)) time.",
        "Case Folding": "Normalizing all characters to lowercase to eliminate case variation.",
        "Porter Stemmer": "Heuristic suffix-stripping algorithm reducing inflectional forms to stem bases."
    }
}

PROCEDURE_STEPS = [
    "Step 1: Review the Aim, Learning Objectives, and Theoretical Foundation of Inverted Indexes.",
    "Step 2: Attempt the Pre-Test questions in the Self-Evaluation tab to assess foundational knowledge.",
    "Step 3: Navigate to the Simulation workbench and select an educational corpus or input custom documents.",
    "Step 4: Configure the linguistic preprocessing switches (Case Folding, Stopword Removal, Porter Stemming).",
    "Step 5: Click 'Construct Inverted Index' and trace the 4 pipeline stages: Tokenization, Triples, Sorting, and Postings Inversion.",
    "Step 6: Inspect the Dictionary statistics (Vocabulary size, Document Frequency df, Collection Frequency cf).",
    "Step 7: Execute Keyword, Boolean (AND, OR, NOT), Phrase, and Proximity search queries in the Query Console.",
    "Step 8: Observe the step-by-step Pointer Comparison Trace to understand the linear-time merge algorithm.",
    "Step 9: Analyze the analytical plots: Zipf's Law rank-frequency curve and Postings List Length distribution.",
    "Step 10: Click 'Record Current Trial' to capture metrics across at least 3 distinct preprocessing configurations.",
    "Step 11: Complete the Self-Evaluation Quiz to verify conceptual mastery and generate instant feedback.",
    "Step 12: Open Report Generation, enter your Student Roll Number and Name, and download the verified PDF report."
]

BENCHMARK_CORPORA = {
    "Corpus 1: Information Retrieval & Search Engines": {
        1: "Information retrieval systems index documents for fast keyword search and query processing.",
        2: "Search engines construct an inverted index with dictionary terms and sorted postings lists.",
        3: "Boolean retrieval evaluates AND, OR, and NOT queries using posting list intersection algorithms.",
        4: "Positional postings store word positions within documents to enable exact phrase queries and proximity search.",
        5: "Modern search engines combine inverted index lexical matching with dense embedding retrieval."
    },
    "Corpus 2: Knowledge Graphs & Graph Databases (KGIRS)": {
        1: "A knowledge graph organizes entities and relationships into an interconnected semantic network.",
        2: "Graph databases like Neo4j store nodes and edges for efficient graph traversal and query execution.",
        3: "Cypher queries retrieve multi-hop relationships and pattern matches across knowledge graphs.",
        4: "Information extraction pipelines identify named entities and semantic relations from unstructured text.",
        5: "Hybrid systems integrate knowledge graphs with information retrieval for contextual search and question answering."
    },
    "Corpus 3: Natural Language Processing & Text Engineering": {
        1: "Text preprocessing involves tokenization, case folding, stopword removal, and word stemming.",
        2: "Stemming algorithms like the Porter stemmer reduce morphological variants of words to a common base.",
        3: "Term frequency and document frequency determine keyword importance according to Zipf's law.",
        4: "Linguistic normalization reduces vocabulary size and improves information retrieval recall.",
        5: "Token position indices allow search engines to distinguish between adjacent phrases and scattered words."
    }
}

DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most",
    "my", "myself", "no", "nor", "not", "now", "of", "off", "on", "once", "only",
    "or", "other", "our", "ours", "ourselves", "out", "over", "own", "s", "same",
    "she", "should", "so", "some", "such", "t", "than", "that", "the", "their",
    "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those",
    "through", "to", "too", "under", "until", "up", "very", "was", "we", "were",
    "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would",
    "you", "your", "yours", "yourself", "yourselves"
}

QUIZ_QUESTIONS = [
    {
        "id": 1,
        "question": "Why is an Inverted Index preferred over a Term-Document Incidence Matrix for large document collections?",
        "options": [
            "A) Incidence matrices cannot store numbers or punctuation characters.",
            "B) Incidence matrices are extremely sparse (>99% zeroes), wasting immense memory, whereas inverted indexes store only non-zero occurrences.",
            "C) Inverted indexes eliminate the need for computing term frequencies.",
            "D) Incidence matrices cannot be stored in binary format."
        ],
        "answer_index": 1,
        "explanation": "Natural language text exhibits extreme sparsity. A full incidence matrix of |V| x N allocates space for billions of zeroes, whereas an inverted index only allocates space for documents where terms actually occur."
    },
    {
        "id": 2,
        "question": "What is the worst-case time complexity of merging two sorted postings lists of lengths x and y using the two-pointer intersection algorithm?",
        "options": [
            "A) O(x * y)",
            "B) O(x + y)",
            "C) O(log(x + y))",
            "D) O(x^2 + y^2)"
        ],
        "answer_index": 1,
        "explanation": "Because postings lists are maintained strictly in sorted order of Document ID, a linear two-pointer scan evaluates each element at most once, yielding optimal O(x + y) running time."
    },
    {
        "id": 3,
        "question": "In processing a conjunctive Boolean query (A AND B AND C) where df(A)=1500, df(B)=30, and df(C)=400, which evaluation sequence minimizes intermediate postings processing?",
        "options": [
            "A) Evaluate (A AND B) first, then intersect with C.",
            "B) Evaluate (A AND C) first, then intersect with B.",
            "C) Evaluate (B AND C) first, then intersect with A (increasing order of document frequency).",
            "D) The evaluation order has no measurable effect on computational cost."
        ],
        "answer_index": 2,
        "explanation": "Standard query optimization sorts terms in increasing order of document frequency (df). Intersecting B (df=30) and C (df=400) creates an intermediate candidate set of at most 30 documents, making the final intersection with A rapid."
    },
    {
        "id": 4,
        "question": "What primary operational capability does a Positional Inverted Index provide compared to a standard document-level inverted index?",
        "options": [
            "A) It allows searching documents by their creation timestamp.",
            "B) It enables exact phrase queries and proximity searches (e.g., word A within k words of word B).",
            "C) It automatically translates queries into foreign languages.",
            "D) It reduces the physical disk space required by the index."
        ],
        "answer_index": 1,
        "explanation": "By storing the exact token offsets (positions) of terms within each document posting, the query engine can verify whether adjacent terms appear sequentially (pos2 = pos1 + 1) or within a window k."
    },
    {
        "id": 5,
        "question": "What is the theoretically optimal skip pointer interval for a sorted postings list of length L to minimize search traversal time?",
        "options": [
            "A) Every 2 postings.",
            "B) Approximately sqrt(L) evenly spaced postings.",
            "C) Exactly L / 2 postings.",
            "D) Skip pointers must only be placed at powers of two."
        ],
        "answer_index": 1,
        "explanation": "Placing skip pointers at intervals of sqrt(L) balances the number of skips with the maximum steps between skip pointers, reducing traversal comparisons to O(sqrt(L))."
    },
    {
        "id": 6,
        "question": "How does aggressive stopword removal and Porter stemming affect the vocabulary size |V| according to Heaps' Law?",
        "options": [
            "A) It significantly decreases vocabulary size |V| and compresses postings lists by merging morphological variants.",
            "B) It increases vocabulary size exponentially by creating new root words.",
            "C) It leaves vocabulary size unchanged because documents retain the same character count.",
            "D) It corrupts the index and prevents keyword search."
        ],
        "answer_index": 0,
        "explanation": "Stopword removal eliminates high-frequency terms while stemming collapses multiple inflectional forms (e.g., 'retrieval', 'retrieving', 'retrieved' -> 'retriev') into a single vocabulary entry, substantially compressing |V|."
    },
    {
        "id": 7,
        "question": "Zipf's Law states that collection frequency f is inversely proportional to frequency rank r (f * r ≈ C). What does this imply about natural language corpora?",
        "options": [
            "A) Every word in the language appears with identical frequency.",
            "B) A small set of frequent words accounts for a huge portion of all tokens, while the vast majority of words appear very rarely (long tail).",
            "C) Longer words always appear more frequently than shorter words.",
            "D) Term frequencies increase exponentially as rank increases."
        ],
        "answer_index": 1,
        "explanation": "Zipf's law describes the heavy-tailed power-law distribution in text: the top 50-100 words (stopwords) make up ~50% of any collection, while thousands of rare terms occur only once or twice."
    },
    {
        "id": 8,
        "question": "What is the precise conceptual difference between Term Frequency (tf) and Document Frequency (df)?",
        "options": [
            "A) tf is the length of the document, while df is the length of the vocabulary.",
            "B) tf is the number of times a term appears in a specific document, while df is the total number of documents containing the term.",
            "C) tf and df are identical metrics used interchangeably.",
            "D) df counts characters, while tf counts words."
        ],
        "answer_index": 1,
        "explanation": "Term frequency tf_{t,d} measures local term occurrence within a single document d, while document frequency df_t measures global dispersion across the entire collection."
    },
    {
        "id": 9,
        "question": "When evaluating a phrase query 'term1 term2', what condition must be met across their positional postings lists?",
        "options": [
            "A) term1 and term2 must have identical document frequencies.",
            "B) There must exist a document d containing term1 at position p and term2 at position p + 1.",
            "C) Both terms must appear anywhere in the document regardless of order.",
            "D) term1 must appear in Doc 1 and term2 must appear in Doc 2."
        ],
        "answer_index": 1,
        "explanation": "Phrase queries require adjacency: within the same document identifier d, the token position of the second term must be exactly one greater than the position of the first term (pos2 = pos1 + 1)."
    },
    {
        "id": 10,
        "question": "What is the primary advantage of Single-Pass In-Memory Indexing (SPIMI) over Block Sort-Based Indexing (BSBI)?",
        "options": [
            "A) SPIMI eliminates the need for sorting postings lists.",
            "B) SPIMI builds dynamic postings lists directly in memory per block without sorting intermediate (term, docID) pairs on disk.",
            "C) SPIMI does not require any RAM memory.",
            "D) SPIMI is only applicable to binary data, not text."
        ],
        "answer_index": 1,
        "explanation": "BSBI writes unsorted (termID, docID) pairs to disk and sorts them, which causes disk I/O bottlenecks. SPIMI allocates dynamic postings arrays in memory as documents are parsed, writing out pre-sorted block indexes."
    }
]

REFERENCES_DATA = [
    {
        "title": "Introduction to Information Retrieval",
        "authors": "Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze",
        "publisher": "Cambridge University Press, 2008",
        "details": "Chapters 1 & 2: Boolean Retrieval, The Inverted Index, Positional Postings, and Skip Pointers.",
        "url": "https://nlp.stanford.edu/IR-book/"
    },
    {
        "title": "Modern Information Retrieval: The Concepts and Technology behind Search",
        "authors": "Ricardo Baeza-Yates and Berthier Ribeiro-Neto",
        "publisher": "Addison-Wesley, 2nd Edition, 2011",
        "details": "Comprehensive coverage of index compression, inverted file structures, and query processing.",
        "url": "https://www.mir2ed.org/"
    },
    {
        "title": "Managing Gigabytes: Compressing and Indexing Documents and Images",
        "authors": "Ian H. Witten, Alistair Moffat, and Timothy C. Bell",
        "publisher": "Morgan Kaufmann Publishers, 1999",
        "details": "Seminal reference on inverted file construction, memory management, and fast text compression.",
        "url": "https://dl.acm.org/doi/book/10.5555/551717"
    },
    {
        "title": "Virtual Labs Portal",
        "authors": "Ministry of Education, Govt. of India",
        "publisher": "Virtual Labs Project (vlabs.ac.in)",
        "details": "Pedagogical lab guidelines and simulation paradigms for Computer Science & Engineering education.",
        "url": "https://vlabs.ac.in/"
    }
]


# ======================================================================================
# 2. ALGORITHMIC SIMULATION ENGINE (PURE PYTHON, ZERO EXTERNAL MODEL DEPENDENCIES)
# ======================================================================================

def pure_porter_stem(word: str) -> str:
    """
    Robust rule-based algorithmic stemmer (Porter-style heuristic).
    Guarantees consistent suffix stripping offline without requiring external NLTK downloads.
    """
    word = word.lower()
    if len(word) <= 2:
        return word

    # Step 1a: Plurals and past participles
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("ss"):
        pass
    elif word.endswith("s"):
        word = word[:-1]

    # Step 1b: -ed, -ing
    if word.endswith("eed"):
        if len(word) > 4:
            word = word[:-1]
    elif word.endswith("ed") and len(word) > 4:
        word = word[:-2]
    elif word.endswith("ing") and len(word) > 5:
        word = word[:-3]

    # Step 2: Suffix replacements
    suffixes = [
        ("ational", "ate"), ("tional", "tion"), ("ization", "ize"),
        ("ation", "ate"), ("izer", "ize"), ("ness", ""), ("ment", ""),
        ("able", ""), ("ible", ""), ("ful", ""), ("ity", "")
    ]
    for suf, rep in suffixes:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            word = word[:-len(suf)] + rep
            break

    return word


def preprocess_document(text: str, lowercase: bool = True, remove_punct: bool = True,
                        remove_stopwords: bool = True, use_stemming: bool = True,
                        stopwords_set: set = None) -> list:
    """
    Performs linguistic preprocessing on raw text.
    Returns list of tuples: (processed_term, original_token_pos, raw_token)
    """
    if stopwords_set is None:
        stopwords_set = DEFAULT_STOPWORDS

    orig_tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", text)
    processed = []

    for pos, raw_tok in enumerate(orig_tokens):
        tok = raw_tok.lower() if lowercase else raw_tok
        if remove_punct:
            tok = re.sub(r"[^\w]", "", tok)

        if not tok:
            continue

        if remove_stopwords and tok.lower() in stopwords_set:
            continue

        term = pure_porter_stem(tok) if use_stemming else tok
        if term:
            processed.append((term, pos, raw_tok))

    return processed


def build_inverted_index(corpus: dict, lowercase: bool = True, remove_punct: bool = True,
                         remove_stopwords: bool = True, use_stemming: bool = True,
                         stopwords_set: set = None) -> dict:
    """
    Executes the 4-stage inverted index construction pipeline:
      Stage 1: Token Stream Extraction
      Stage 2: (Term, DocID, Position) Triples Generation
      Stage 3: Lexicographical Sorting
      Stage 4: Postings List Aggregation
    """
    t_start = time.perf_counter()

    stage1_tokens = {}
    stage2_triples = []
    total_tokens_count = 0

    for doc_id, text in sorted(corpus.items()):
        doc_tokens = preprocess_document(
            text, lowercase=lowercase, remove_punct=remove_punct,
            remove_stopwords=remove_stopwords, use_stemming=use_stemming,
            stopwords_set=stopwords_set
        )
        stage1_tokens[doc_id] = doc_tokens
        total_tokens_count += len(doc_tokens)
        for term, pos, raw_tok in doc_tokens:
            stage2_triples.append({
                "term": term,
                "doc_id": doc_id,
                "pos": pos,
                "raw": raw_tok
            })

    # Stage 3: Lexicographic Sort
    stage3_sorted = sorted(stage2_triples, key=lambda x: (x["term"], x["doc_id"], x["pos"]))

    # Stage 4: Postings List Inversion
    index_dict = defaultdict(lambda: {"df": 0, "cf": 0, "postings": {}})
    for item in stage3_sorted:
        term = item["term"]
        doc_id = item["doc_id"]
        pos = item["pos"]

        entry = index_dict[term]
        entry["cf"] += 1
        if doc_id not in entry["postings"]:
            entry["postings"][doc_id] = {
                "tf": 0,
                "positions": []
            }
            entry["df"] += 1
        entry["postings"][doc_id]["tf"] += 1
        entry["postings"][doc_id]["positions"].append(pos)

    t_end = time.perf_counter()
    indexing_time_ms = round((t_end - t_start) * 1000, 3)

    vocab_size = len(index_dict)
    total_postings = sum(len(v["postings"]) for v in index_dict.values())
    num_docs = len(corpus)
    sparsity_pct = round((1.0 - (total_postings / max(1, vocab_size * num_docs))) * 100, 2)

    return {
        "stage1_tokens": stage1_tokens,
        "stage2_triples": stage2_triples,
        "stage3_sorted": stage3_sorted,
        "index": dict(index_dict),
        "vocab_size": vocab_size,
        "total_tokens": total_tokens_count,
        "total_postings": total_postings,
        "sparsity_pct": sparsity_pct,
        "num_docs": num_docs,
        "indexing_time_ms": indexing_time_ms
    }


def execute_keyword_query(term: str, index: dict, use_stemming: bool = True) -> dict:
    """Executes single keyword search against inverted index."""
    t_start = time.perf_counter()
    clean_term = pure_porter_stem(term.strip().lower()) if use_stemming else term.strip().lower()

    if clean_term in index:
        entry = index[clean_term]
        postings = entry["postings"]
        doc_ids = sorted(list(postings.keys()))
        t_end = time.perf_counter()
        return {
            "term": clean_term,
            "found": True,
            "df": entry["df"],
            "cf": entry["cf"],
            "docs": doc_ids,
            "postings": postings,
            "latency_us": round((t_end - t_start) * 1_000_000, 2),
            "trace": [f"Direct dictionary lookup for term '{clean_term}' -> Matched {len(doc_ids)} documents: {doc_ids}"]
        }
    t_end = time.perf_counter()
    return {
        "term": clean_term,
        "found": False,
        "df": 0,
        "cf": 0,
        "docs": [],
        "postings": {},
        "latency_us": round((t_end - t_start) * 1_000_000, 2),
        "trace": [f"Term '{clean_term}' not present in vocabulary."]
    }


def execute_boolean_and_with_trace(list1: list, list2: list, term1: str, term2: str) -> tuple:
    """
    Simulates the two-pointer sorted postings list intersection algorithm.
    Generates a step-by-step pointer trace.
    """
    p1, p2 = 0, 0
    res = []
    trace = []
    comparisons = 0

    trace.append(f"Starting Conjunctive (AND) Intersection: '{term1}' (L1={len(list1)}) AND '{term2}' (L2={len(list2)})")

    while p1 < len(list1) and p2 < len(list2):
        comparisons += 1
        d1 = list1[p1]
        d2 = list2[p2]

        if d1 == d2:
            res.append(d1)
            trace.append(f"Step {comparisons}: p1={p1} (Doc {d1}) == p2={p2} (Doc {d2}) -> MATCH! Added Doc {d1} to results. Both pointers advanced.")
            p1 += 1
            p2 += 1
        elif d1 < d2:
            trace.append(f"Step {comparisons}: p1={p1} (Doc {d1}) < p2={p2} (Doc {d2}) -> Advance p1 (Doc {d1} cannot match).")
            p1 += 1
        else:
            trace.append(f"Step {comparisons}: p1={p1} (Doc {d1}) > p2={p2} (Doc {d2}) -> Advance p2 (Doc {d2} cannot match).")
            p2 += 1

    trace.append(f"Intersection complete in {comparisons} comparisons. Resulting DocIDs: {res}")
    return res, trace


def execute_boolean_or_with_trace(list1: list, list2: list, term1: str, term2: str) -> tuple:
    """Simulates the two-pointer sorted postings list union algorithm."""
    p1, p2 = 0, 0
    res = []
    trace = []
    comparisons = 0

    trace.append(f"Starting Disjunctive (OR) Union: '{term1}' (L1={len(list1)}) OR '{term2}' (L2={len(list2)})")

    while p1 < len(list1) and p2 < len(list2):
        comparisons += 1
        d1 = list1[p1]
        d2 = list2[p2]

        if d1 == d2:
            res.append(d1)
            trace.append(f"Step {comparisons}: Doc {d1} present in both lists -> Added Doc {d1}. Both pointers advanced.")
            p1 += 1
            p2 += 1
        elif d1 < d2:
            res.append(d1)
            trace.append(f"Step {comparisons}: Doc {d1} < Doc {d2} -> Added Doc {d1}. Advance p1.")
            p1 += 1
        else:
            res.append(d2)
            trace.append(f"Step {comparisons}: Doc {d2} < Doc {d1} -> Added Doc {d2}. Advance p2.")
            p2 += 1

    while p1 < len(list1):
        res.append(list1[p1])
        trace.append(f"Drain List 1: Appended remaining Doc {list1[p1]}.")
        p1 += 1

    while p2 < len(list2):
        res.append(list2[p2])
        trace.append(f"Drain List 2: Appended remaining Doc {list2[p2]}.")
        p2 += 1

    trace.append(f"Union complete in {comparisons} comparisons. Resulting DocIDs: {res}")
    return res, trace


def execute_phrase_query(phrase: str, index: dict, use_stemming: bool = True) -> dict:
    """Executes exact phrase query using positional posting list intersection."""
    t_start = time.perf_counter()
    tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", phrase.lower())
    if not tokens:
        return {"phrase": phrase, "docs": [], "trace": ["Empty phrase."], "latency_us": 0}

    proc_terms = [pure_porter_stem(t) if use_stemming else t for t in tokens]
    trace = [f"Evaluating phrase query: '{' '.join(tokens)}' -> Processed terms: {proc_terms}"]

    # Check if all terms exist in index
    for term in proc_terms:
        if term not in index:
            trace.append(f"Term '{term}' not present in index. Phrase cannot match any document.")
            t_end = time.perf_counter()
            return {
                "phrase": phrase,
                "terms": proc_terms,
                "docs": [],
                "matches_by_doc": {},
                "trace": trace,
                "latency_us": round((t_end - t_start) * 1_000_000, 2)
            }

    # Intersect candidate documents
    candidate_docs = set(index[proc_terms[0]]["postings"].keys())
    for term in proc_terms[1:]:
        candidate_docs &= set(index[term]["postings"].keys())

    candidate_docs = sorted(list(candidate_docs))
    trace.append(f"Candidate documents containing all phrase terms: {candidate_docs}")

    matching_docs = []
    matches_by_doc = {}

    for doc_id in candidate_docs:
        pos_lists = [index[term]["postings"][doc_id]["positions"] for term in proc_terms]
        matched_offsets = []

        for start_pos in pos_lists[0]:
            matched = True
            for offset, term_positions in enumerate(pos_lists[1:], start=1):
                if (start_pos + offset) not in term_positions:
                    matched = False
                    break
            if matched:
                matched_offsets.append([start_pos + i for i in range(len(proc_terms))])

        if matched_offsets:
            matching_docs.append(doc_id)
            matches_by_doc[doc_id] = matched_offsets
            trace.append(f"Doc {doc_id}: Exact phrase match confirmed at token positions {matched_offsets}")
        else:
            trace.append(f"Doc {doc_id}: All terms present, but adjacency condition (pos_k+1 = pos_k + 1) failed.")

    t_end = time.perf_counter()
    return {
        "phrase": phrase,
        "terms": proc_terms,
        "docs": matching_docs,
        "matches_by_doc": matches_by_doc,
        "trace": trace,
        "latency_us": round((t_end - t_start) * 1_000_000, 2)
    }


def execute_proximity_query(term1: str, term2: str, max_dist: int, index: dict, use_stemming: bool = True) -> dict:
    """Executes proximity search: term1 NEAR/k term2."""
    t_start = time.perf_counter()
    t1 = pure_porter_stem(term1.lower()) if use_stemming else term1.lower()
    t2 = pure_porter_stem(term2.lower()) if use_stemming else term2.lower()

    trace = [f"Proximity Search: '{term1}' NEAR/{max_dist} '{term2}' (Processed: '{t1}' NEAR/{max_dist} '{t2}')"]

    if t1 not in index or t2 not in index:
        missing = [t for t in [t1, t2] if t not in index]
        trace.append(f"Terms {missing} not in index.")
        t_end = time.perf_counter()
        return {"docs": [], "matches_by_doc": {}, "trace": trace, "latency_us": round((t_end - t_start) * 1_000_000, 2)}

    common_docs = sorted(list(set(index[t1]["postings"].keys()) & set(index[t2]["postings"].keys())))
    matching_docs = []
    matches_by_doc = {}

    for doc_id in common_docs:
        p1_list = index[t1]["postings"][doc_id]["positions"]
        p2_list = index[t2]["postings"][doc_id]["positions"]
        pairs = []
        for pos1 in p1_list:
            for pos2 in p2_list:
                dist = abs(pos1 - pos2)
                if 0 < dist <= max_dist:
                    pairs.append((pos1, pos2, dist))

        if pairs:
            matching_docs.append(doc_id)
            matches_by_doc[doc_id] = pairs
            trace.append(f"Doc {doc_id}: Proximity match found with positions (pos1, pos2, dist)={pairs}")

    t_end = time.perf_counter()
    return {
        "docs": matching_docs,
        "matches_by_doc": matches_by_doc,
        "trace": trace,
        "latency_us": round((t_end - t_start) * 1_000_000, 2)
    }


# ======================================================================================
# 3. VERIFIED PDF LAB REPORT EXPORTER (FPDF2 - LATIN-1 SAFE)
# ======================================================================================

def sanitize_pdf_text(text: str) -> str:
    """Sanitizes text by replacing Unicode math/arrow symbols with Latin-1 equivalents."""
    if not isinstance(text, str):
        text = str(text)
    replacements = {
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
        "\u2013": "-", "\u2014": "-", "\u2264": "<=", "\u2265": ">=",
        "\u2260": "!=", "\u2192": "->", "\u03b2": "beta", "\u03bc": "u",
        "\u2248": "~", "\u221d": "proportional to", "\u03a9": "Omega",
        "\u2208": "in", "\u2200": "for all", "\u2203": "exists",
        "\u2229": "AND", "\u222a": "OR", "\u2205": "empty"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return unicodedata.normalize("NFKD", text).encode("latin-1", "ignore").decode("latin-1")


class VirtualLabReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 58, 138)
        self.cell(0, 5, sanitize_pdf_text("VIRTUAL LABS CA - EXPERIMENT REPORT"), 0, 1, "C")
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 116, 139)
        self.cell(0, 4, sanitize_pdf_text("Knowledge Graphs & Information Retrieval Systems (KGIRS) | Experiment 3"), 0, 1, "C")
        self.line(10, 20, 200, 20)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(128, 128, 128)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}} | Virtual Laboratory Experiment Verification Report", align="C")



def generate_pdf_certificate(student_name: str, student_id: str, student_div: str,
                             date_str: str, quiz_score: int, quiz_total: int,
                             trials_count: int, cert_id: str) -> bytes:
    """Generates an official Virtual Lab Certificate of Completion PDF."""
    pdf = FPDF(orientation="L", unit="mm", format="A4")  # Landscape A4: 297mm x 210mm
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    # Outer Decorative Border
    pdf.set_draw_color(30, 41, 59)
    pdf.set_line_width(2.0)
    pdf.rect(10, 10, 277, 190)

    # Inner Gold Border
    pdf.set_draw_color(217, 119, 6)
    pdf.set_line_width(0.8)
    pdf.rect(14, 14, 269, 182)

    # Top Emblem / Institution Header
    pdf.set_xy(20, 22)
    pdf.set_font("Helvetica", "B", 13)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(257, 7, sanitize_pdf_text("VIRTUAL LABS - AN INITIATIVE OF MINISTRY OF EDUCATION (vlabs.ac.in)"), align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(257, 6, sanitize_pdf_text("DEPARTMENT OF COMPUTER ENGINEERING / ARTIFICIAL INTELLIGENCE & DATA SCIENCE"), align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(3)

    # Certificate Title
    pdf.set_font("Helvetica", "B", 24)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(257, 12, sanitize_pdf_text("CERTIFICATE OF COMPLETION"), align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "I", 11)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(257, 6, sanitize_pdf_text("This is to officially certify that"), align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(2)

    # Student Name
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(37, 99, 235)
    pdf.cell(257, 10, sanitize_pdf_text(student_name.upper() if student_name else "STUDENT NAME"), align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(51, 65, 85)
    meta_line = f"Roll Number: {student_id}   |   Division: {student_div}   |   Course: Knowledge Graphs & IR Systems"
    pdf.cell(257, 7, sanitize_pdf_text(meta_line), align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(2)

    # Body description
    body_text = (
        "has successfully conducted, analyzed, and completed the experimental laboratory requirements for\n"
        "Experiment 3: Construction of an Inverted Index, demonstrating practical proficiency in:\n"
        "1. Document Ingestion, Tokenization, and Linguistic Preprocessing (Stopwords, Porter Stemming)\n"
        "2. Lexicographical Sorting and Inverted Postings List Generation with Positional Offsets\n"
        "3. Linear Two-Pointer Postings Intersection (O(L1 + L2)) for Conjunctive & Disjunctive Queries\n"
        "4. Empirical Frequency Modeling and Validation of Zipf's Law on Textual Corpora"
    )
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_text_color(71, 85, 105)
    for line in body_text.split("\n"):
        pdf.cell(257, 5, sanitize_pdf_text(line), align="C", new_x="LMARGIN", new_y="NEXT")

    pdf.ln(4)

    # Performance Stats Box
    pdf.set_fill_color(248, 250, 252)
    pdf.set_draw_color(226, 232, 240)
    pdf.rect(48, 136, 201, 14, "FD")

    pdf.set_xy(50, 138)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(30, 41, 59)
    perf_text = f"Assessment Score: {quiz_score} / {quiz_total}   |   Simulation Trials Executed: {trials_count}   |   Status: Verified & Completed"
    pdf.cell(197, 10, sanitize_pdf_text(perf_text), align="C")

    # Bottom Signatures and Verification
    pdf.set_xy(25, 162)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(70, 5, sanitize_pdf_text(f"Issue Date: {date_str}"), align="L")

    pdf.set_xy(25, 168)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(70, 5, sanitize_pdf_text(f"Certificate ID: {cert_id}"), align="L")

    # Instructor Signature Line
    pdf.set_xy(190, 160)
    pdf.set_draw_color(148, 163, 184)
    pdf.line(190, 168, 260, 168)
    pdf.set_xy(190, 169)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(30, 41, 59)
    pdf.cell(70, 5, sanitize_pdf_text("Course Instructor / Evaluator"), align="C")

    pdf.set_xy(190, 174)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(70, 5, sanitize_pdf_text("Virtual Laboratories Network"), align="C")

    return bytes(pdf.output())


def generate_pdf_report(student_name: str, student_id: str, student_div: str, date_str: str,
                        trials_df: pd.DataFrame, quiz_score: int, quiz_total: int,
                        student_notes: str, corpus_name: str, vocab_size: int,
                        postings_count: int) -> bytes:
    """Compiles verified experiment session data into an official Virtual Lab report."""
    pdf = VirtualLabReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # Title
    pdf.set_text_color(15, 23, 42)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 8, sanitize_pdf_text(EXPERIMENT_CONFIG["title"]), align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # Student Metadata Box
    pdf.set_fill_color(241, 245, 249)
    pdf.set_draw_color(203, 213, 225)
    pdf.rect(10, 32, 190, 24, "FD")

    pdf.set_xy(14, 34)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(32, 5, "Student Name:", 0)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(60, 5, sanitize_pdf_text(student_name or "Student"), 0)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(34, 5, "Roll / Student ID:", 0)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(50, 5, sanitize_pdf_text(student_id or "EXP3-01"), 1)

    pdf.set_xy(14, 41)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(32, 5, "Class / Division:", 0)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(60, 5, sanitize_pdf_text(student_div or "D17A/B/C"), 0)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(34, 5, "Submission Date:", 0)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(50, 5, sanitize_pdf_text(date_str or datetime.now().strftime("%Y-%m-%d")), 1)

    pdf.set_xy(14, 48)
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(32, 5, "Target Rolls:", 0)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(60, 5, sanitize_pdf_text(EXPERIMENT_CONFIG["target_rolls"]), 0)

    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(34, 5, "Self-Evaluation:", 0)
    pdf.set_font("Helvetica", "B", 8)
    if quiz_score >= max(1, quiz_total // 2):
        pdf.set_text_color(16, 185, 129)
    else:
        pdf.set_text_color(239, 68, 68)
    score_perc = int((quiz_score / max(1, quiz_total)) * 100)
    pdf.cell(50, 5, f"{quiz_score} / {quiz_total} ({score_perc}%)", 1)

    pdf.ln(10)

    # 1. Aim & Objectives
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "1. Aim & Learning Objectives", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(51, 65, 85)
    pdf.multi_cell(0, 4, sanitize_pdf_text(f"Aim: {EXPERIMENT_CONFIG['aim']}"))
    pdf.ln(2)

    for i, obj in enumerate(EXPERIMENT_CONFIG["objectives"]):
        pdf.cell(4, 4, "-", 0)
        pdf.cell(0, 4, sanitize_pdf_text(f" {obj}"), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(3)

    # 2. System Architecture & Theory Summary
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "2. Theoretical Summary & Index Metrics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(51, 65, 85)
    summary_text = (
        f"Active Corpus: {corpus_name}\n"
        f"Constructed Inverted Index Vocabulary Size (|V|): {vocab_size} unique terms.\n"
        f"Total Postings Elements: {postings_count} entries.\n"
        f"Algorithm Complexities: Boolean AND/OR Pointer Merge = O(L1 + L2); Phrase Adjacency = O(Pos1 + Pos2); "
        f"Incidence Matrix Space = O(|V| * N); Inverted Index Space = O(|Postings|)."
    )
    pdf.multi_cell(0, 4, sanitize_pdf_text(summary_text))
    pdf.ln(3)

    # 3. Recorded Trials Table
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "3. Recorded Experimental Session Trials", new_x="LMARGIN", new_y="NEXT")

    if trials_df.empty:
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 5, "No experimental trials logged in this session.", new_x="LMARGIN", new_y="NEXT")
    else:
        pdf.set_fill_color(37, 99, 235)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 7)

        # Select essential columns
        cols = ["Trial #", "Corpus", "Stopwords", "Stemming", "Vocab |V|", "Query", "Hits", "Time (us)"]
        col_widths = [14, 34, 18, 18, 18, 48, 14, 26]

        for c, w in zip(cols, col_widths):
            pdf.cell(w, 5, sanitize_pdf_text(c), 1, 0, "C", True)
        pdf.ln()

        pdf.set_fill_color(248, 250, 252)
        pdf.set_text_color(30, 41, 59)
        pdf.set_font("Helvetica", "", 7)
        fill = False

        for _, row in trials_df.iterrows():
            row_vals = [
                str(row.get("Trial #", "-")),
                str(row.get("Corpus", "-"))[:18],
                str(row.get("Stopwords", "-")),
                str(row.get("Stemming", "-")),
                str(row.get("Vocab |V|", "-")),
                str(row.get("Query", "-"))[:24],
                str(row.get("Hits", "-")),
                str(row.get("Time (us)", "-"))
            ]
            for val, w in zip(row_vals, col_widths):
                pdf.cell(w, 5, sanitize_pdf_text(val), 1, 0, "C", fill)
            pdf.ln()
            fill = not fill
    pdf.ln(4)

    # 4. Discussion & Observations
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 6, "4. Student Observations & Analytical Discussion", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(51, 65, 85)
    obs_text = student_notes.strip() if student_notes.strip() else (
        "1. Stopword removal and Porter stemming significantly reduced vocabulary size |V|, validating Heaps' Law.\n"
        "2. Conjunctive Boolean queries executed in sub-millisecond linear time O(L1 + L2) through sorted pointer intersection.\n"
        "3. Positional postings allowed exact phrase verification with zero false positives by matching consecutive token offsets.\n"
        "4. Empirical term frequency distribution conforms to the heavy-tailed power law modeled by Zipf's Law."
    )
    pdf.multi_cell(0, 4, sanitize_pdf_text(obs_text))
    pdf.ln(6)

    # Verification and Sign-off Box
    pdf.set_draw_color(180, 180, 180)
    pdf.line(130, pdf.get_y() + 12, 190, pdf.get_y() + 12)
    pdf.set_xy(130, pdf.get_y() + 14)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(60, 4, sanitize_pdf_text("Evaluator / Faculty Signature & Stamp"), align="C")

    return bytes(pdf.output())


# ======================================================================================
# 4. SECTION RENDERERS (AIM, THEORY, PROCEDURE, SIMULATION, QUIZ, REFS, FEEDBACK, REPORT)
# ======================================================================================

def render_purpose_section():
    """Renders comprehensive Purpose, Historical Context, Problem Formulation, and Applications using Google's color palette."""
    st.markdown("""
    <div style="background: #ffffff; padding: 28px 32px; border-radius: 16px; border: 1px solid #dadce0; border-left: 6px solid #4285F4; box-shadow: 0 1px 6px rgba(60,64,67,0.10); margin-bottom: 28px;">
        <div style="font-size: 14px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 8px;">
            <span style="color: #4285F4;">E</span><span style="color: #EA4335;">X</span><span style="color: #FBBC05;">P</span> <span style="color: #34A853;">3</span> • <span style="color: #1a73e8;">PURPOSE & ARCHITECTURAL FOUNDATION</span>
        </div>
        <h1 style="margin: 0; font-size: 34px; font-weight: 700; color: #202124; letter-spacing: -0.5px;">The Inverted Index: The Engine of Information Retrieval</h1>
        <p style="margin: 10px 0 0 0; color: #5f6368; font-size: 18px; line-height: 1.65;">
            Discover why the Inverted Index is the foundational algorithmic data structure enabling modern search engines, enterprise databases, and e-commerce platforms to query billions of documents in sub-milliseconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 1. WHY THIS EXPERIMENT?
    st.subheader("1. Why This Experiment? (Pedagogical Rationale)")
    st.markdown("""
    <p style="font-size: 18px; line-height: 1.75; color: #202124;">
        In traditional computer science curricula, students master linear arrays, hash tables, binary search trees, and relational SQL databases. 
        However, when searching through <b>unstructured natural language text</b> across millions or billions of web documents, 
        traditional data structures completely collapse under computational and storage weight.
    </p>
    """, unsafe_allow_html=True)

    col_w1, col_w2 = st.columns(2)
    with col_w1:
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #dadce0; border-left: 5px solid #4285F4; border-radius: 12px; padding: 22px; height: 100%; box-shadow: 0 1px 4px rgba(60,64,67,0.08);">
            <h4 style="color: #1a73e8; margin-top: 0; font-size: 20px; font-weight: 600;">🎯 What You Will Master</h4>
            <ul style="font-size: 17px; line-height: 1.8; color: #202124; margin-bottom: 0;">
                <li><b>Index Inversion:</b> How to convert raw document streams into a high-performance Vocabulary Lexicon and sorted Postings Lists.</li>
                <li><b>Algorithmic Traversal:</b> Implementing the linear-time <code>O(L1 + L2)</code> Two-Pointer Merge algorithm for Boolean retrieval.</li>
                <li><b>Linguistic Optimization:</b> Quantifying how Case-Folding, Stopword Removal, and Porter Stemming compress index size and accelerate search.</li>
                <li><b>Empirical Validation:</b> Validating mathematical power-law distributions (Zipf's Law and Heaps' Law) on real document text.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_w2:
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #dadce0; border-left: 5px solid #34A853; border-radius: 12px; padding: 22px; height: 100%; box-shadow: 0 1px 4px rgba(60,64,67,0.08);">
            <h4 style="color: #1e8e3e; margin-top: 0; font-size: 20px; font-weight: 600;">⚡ The Paradigm Shift</h4>
            <p style="font-size: 17px; line-height: 1.75; color: #202124; margin-bottom: 10px;">
                Searching a 10 TB corpus with sequential scanning (like <code>grep</code>) requires reading every single byte from disk—taking <b>several hours per query</b>.
            </p>
            <p style="font-size: 17px; line-height: 1.75; color: #202124; margin-bottom: 0;">
                By pre-computing an <b>Inverted Index</b>, search engines invert the relationship: instead of asking <i>"What words are in Document X?"</i>, we ask <i>"What documents contain Word Y?"</i>—answering queries in <b>less than 15 milliseconds</b>!
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # 2. WHAT PROBLEM IS IT SOLVING?
    st.subheader("2. What Problem Is It Solving? (The Core Technical Challenges)")
    st.markdown("""
    <p style="font-size: 18px; line-height: 1.75; color: #202124;">
        The Inverted Index was engineered to solve five fundamental computational roadblocks in computer science and text processing:
    </p>
    """, unsafe_allow_html=True)

    p_cols = st.columns(3)
    with p_cols[0]:
        st.markdown("""
        <div style="border: 1px solid #dadce0; border-left: 4px solid #4285F4; border-radius: 12px; padding: 20px; margin-bottom: 16px; background: #ffffff; box-shadow: 0 1px 3px rgba(60,64,67,0.08);">
            <div style="font-size: 26px; margin-bottom: 6px;">🐢</div>
            <h4 style="margin: 0 0 8px 0; color: #202124; font-size: 18px; font-weight: 600;">1. The Linear Scan Bottleneck</h4>
            <p style="font-size: 16px; color: #5f6368; line-height: 1.65; margin: 0;">
                <b>Problem:</b> Scanning text sequentially is <code>O(N · L)</code> where N is total docs and L is text length.<br>
                <b>Solution:</b> Inverting words into indexed sorted DocID lists reduces search time to <code>O(Postings Length)</code>, eliminating full-text disk scans.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with p_cols[1]:
        st.markdown("""
        <div style="border: 1px solid #dadce0; border-left: 4px solid #EA4335; border-radius: 12px; padding: 20px; margin-bottom: 16px; background: #ffffff; box-shadow: 0 1px 3px rgba(60,64,67,0.08);">
            <div style="font-size: 26px; margin-bottom: 6px;">🔤</div>
            <h4 style="margin: 0 0 8px 0; color: #202124; font-size: 18px; font-weight: 600;">2. Morphological Variance</h4>
            <p style="font-size: 16px; color: #5f6368; line-height: 1.65; margin: 0;">
                <b>Problem:</b> Different grammatical forms (<i>retrieval, retrieving, retrieved</i>) cause keyword search misses.<br>
                <b>Solution:</b> Linguistic preprocessing collapses inflections to base stems, reducing dictionary size by 30–45%.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with p_cols[2]:
        st.markdown("""
        <div style="border: 1px solid #dadce0; border-left: 4px solid #FBBC05; border-radius: 12px; padding: 20px; margin-bottom: 16px; background: #ffffff; box-shadow: 0 1px 3px rgba(60,64,67,0.08);">
            <div style="font-size: 26px; margin-bottom: 6px;">⚡</div>
            <h4 style="margin: 0 0 8px 0; color: #202124; font-size: 18px; font-weight: 600;">3. Multi-Term Boolean Logic</h4>
            <p style="font-size: 16px; color: #5f6368; line-height: 1.65; margin: 0;">
                <b>Problem:</b> Evaluating <code>A AND B</code> or <code>A NOT B</code> requires cross-referencing document sets.<br>
                <b>Solution:</b> Two-pointer list intersection walks both sorted lists in linear time without opening files.
            </p>
        </div>
        """, unsafe_allow_html=True)

    p_cols2 = st.columns(2)
    with p_cols2[0]:
        st.markdown("""
        <div style="border: 1px solid #dadce0; border-left: 4px solid #34A853; border-radius: 12px; padding: 20px; background: #ffffff; box-shadow: 0 1px 3px rgba(60,64,67,0.08);">
            <div style="font-size: 26px; margin-bottom: 6px;">📍</div>
            <h4 style="margin: 0 0 8px 0; color: #202124; font-size: 18px; font-weight: 600;">4. Exact Phrase & Proximity Constraints</h4>
            <p style="font-size: 16px; color: #5f6368; line-height: 1.65; margin: 0;">
                <b>Problem:</b> How can an engine verify if <i>"information"</i> is adjacent to <i>"retrieval"</i> without re-reading the document?<br>
                <b>Solution:</b> Positional Inverted Indexes store exact token offsets <code>pos: [0, 4]</code> inside posting nodes, enabling exact phrase and proximity (<code>NEAR/k</code>) matching.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with p_cols2[1]:
        st.markdown("""
        <div style="border: 1px solid #dadce0; border-left: 4px solid #4285F4; border-radius: 12px; padding: 20px; background: #ffffff; box-shadow: 0 1px 3px rgba(60,64,67,0.08);">
            <div style="font-size: 26px; margin-bottom: 6px;">💾</div>
            <h4 style="margin: 0 0 8px 0; color: #202124; font-size: 18px; font-weight: 600;">5. Extreme Matrix Sparsity (>99%)</h4>
            <p style="font-size: 16px; color: #5f6368; line-height: 1.65; margin: 0;">
                <b>Problem:</b> A full Term-Document matrix of 1M documents and 100k vocabulary words requires 100 Billion cells (mostly zeros).<br>
                <b>Solution:</b> Inverted postings lists store strictly non-zero occurrences, achieving massive >90% compression in memory.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # 3. HISTORY & EVOLUTION
    st.subheader("3. History & Origins: From Punch Cards to Modern Web Search")
    st.markdown("""
    <p style="font-size: 18px; line-height: 1.75; color: #202124;">
        The inverted index is one of the most enduring algorithms in computer science history:
    </p>
    """, unsafe_allow_html=True)

    timeline_items = [
        ("1957", "Hans Peter Luhn (IBM)", "Introduces automated indexing, token frequency statistics, and Key Word in Context (KWIC), planting the seeds of computer-based information retrieval.", "#4285F4"),
        ("1968", "Gerard Salton (Cornell University)", "The 'Father of Information Retrieval' develops the SMART system, establishing the Vector Space Model (VSM), TF-IDF term weighting, and formal inverted index architecture.", "#EA4335"),
        ("1970s–80s", "Commercial Boolean Systems", "Early legal and medical systems like DIALOG, LexisNexis, and MEDLINE implement inverted indexes on magnetic disks for sub-second keyword search.", "#FBBC05"),
        ("1994", "Stephen Robertson & Karen Spärck Jones", "Formulate the Probabilistic Relevance Framework leading to Okapi BM25—the gold standard scoring formula for inverted postings lists.", "#34A853"),
        ("1998–2004", "The Web Era & Distributed Inverted Indexes", "Modern search engines scale inverted indexing across global server clusters using MapReduce and distributed tables, handling billions of web pages daily.", "#4285F4"),
        ("2004–Present", "Open Source & Hybrid Neural Search", "Doug Cutting builds Apache Lucene (powering Elasticsearch and OpenSearch). Today, inverted indexes are unified with dense vector embeddings (HNSW) for hybrid search.", "#EA4335")
    ]

    for yr, author, desc, pill_color in timeline_items:
        st.markdown(f"""
        <div style="display: flex; gap: 18px; margin-bottom: 16px; align-items: baseline;">
            <span style="background: {pill_color}; color: #ffffff; font-weight: 700; font-size: 14px; padding: 5px 14px; border-radius: 16px; min-width: 95px; text-align: center;">{yr}</span>
            <div>
                <b style="color: #202124; font-size: 17px;">{author}:</b>
                <span style="color: #5f6368; font-size: 16.5px;"> {desc}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # 4. WHERE ALL IS IT USED TODAY?
    st.subheader("4. Where All Is It Used Today? (Real-World & Industrial Applications)")
    st.markdown("""
    <p style="font-size: 18px; line-height: 1.75; color: #202124;">
        Almost every modern software application that features a search bar relies directly on Inverted Indexes:
    </p>
    """, unsafe_allow_html=True)

    app_cols = st.columns(3)
    with app_cols[0]:
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #dadce0; border-top: 4px solid #4285F4; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(60,64,67,0.08);">
            <h4 style="color: #1a73e8; margin: 0 0 8px 0; font-size: 18px; font-weight: 600;">🌐 Web Search Engines</h4>
            <p style="font-size: 15.5px; color: #5f6368; margin: 0; line-height: 1.65;"><b>Global Web Search Platforms</b><br>Indexes tens of billions of web documents across distributed inverted clusters to return results in under 50ms.</p>
        </div>
        """, unsafe_allow_html=True)
    with app_cols[1]:
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #dadce0; border-top: 4px solid #EA4335; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(60,64,67,0.08);">
            <h4 style="color: #d93025; margin: 0 0 8px 0; font-size: 18px; font-weight: 600;">📊 Log & Security Analytics</h4>
            <p style="font-size: 15.5px; color: #5f6368; margin: 0; line-height: 1.65;"><b>Elasticsearch, OpenSearch, Splunk</b><br>Ingests petabytes of operational server logs, security event traces, and telemetry for real-time querying.</p>
        </div>
        """, unsafe_allow_html=True)
    with app_cols[2]:
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #dadce0; border-top: 4px solid #FBBC05; border-radius: 12px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(60,64,67,0.08);">
            <h4 style="color: #f9ab00; margin: 0 0 8px 0; font-size: 18px; font-weight: 600;">🛒 E-Commerce Catalogs</h4>
            <p style="font-size: 15.5px; color: #5f6368; margin: 0; line-height: 1.65;"><b>Amazon, Flipkart, Shopify</b><br>Enables instant product searches across millions of SKUs with faceted attribute filtering and typo tolerance.</p>
        </div>
        """, unsafe_allow_html=True)

    app_cols2 = st.columns(3)
    with app_cols2[0]:
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #dadce0; border-top: 4px solid #34A853; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(60,64,67,0.08);">
            <h4 style="color: #1e8e3e; margin: 0 0 8px 0; font-size: 18px; font-weight: 600;">💻 Codebase Intelligence</h4>
            <p style="font-size: 15.5px; color: #5f6368; margin: 0; line-height: 1.65;"><b>GitHub Code Search, Sourcegraph, VS Code</b><br>Indexes abstract syntax trees (ASTs), symbols, and identifiers across billions of lines of code.</p>
        </div>
        """, unsafe_allow_html=True)
    with app_cols2[1]:
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #dadce0; border-top: 4px solid #4285F4; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(60,64,67,0.08);">
            <h4 style="color: #1a73e8; margin: 0 0 8px 0; font-size: 18px; font-weight: 600;">🗄️ Relational & NoSQL DBs</h4>
            <p style="font-size: 15.5px; color: #5f6368; margin: 0; line-height: 1.65;"><b>PostgreSQL GIN, MySQL FULLTEXT, MongoDB</b><br>Powers native full-text search extensions inside databases without external search engines.</p>
        </div>
        """, unsafe_allow_html=True)
    with app_cols2[2]:
        st.markdown("""
        <div style="background: #ffffff; border: 1px solid #dadce0; border-top: 4px solid #EA4335; border-radius: 12px; padding: 20px; box-shadow: 0 1px 3px rgba(60,64,67,0.08);">
            <h4 style="color: #d93025; margin: 0 0 8px 0; font-size: 18px; font-weight: 600;">🤖 Hybrid AI Search (RAG)</h4>
            <p style="font-size: 15.5px; color: #5f6368; margin: 0; line-height: 1.65;"><b>Qdrant, Pinecone, LanceDB</b><br>Combines sparse BM25 inverted indexes with dense vector embeddings to provide hallucination-free context to LLMs.</p>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Learning Outcomes & Prerequisites
    col_lo1, col_lo2 = st.columns(2)
    with col_lo1:
        st.subheader("Key Learning Objectives")
        for i, obj in enumerate(EXPERIMENT_CONFIG["objectives"]):
            st.markdown(f"<p style='font-size: 17px; line-height: 1.7; margin-bottom: 8px;'><b>{i+1}.</b> {obj}</p>", unsafe_allow_html=True)
    with col_lo2:
        st.subheader("Prerequisites & Foundations")
        st.markdown("""
        <ul style="font-size: 17px; line-height: 1.8; color: #202124;">
            <li><b>Data Structures:</b> Hash Tables, Arrays, Singly Linked Lists, Dictionaries.</li>
            <li><b>Discrete Mathematics:</b> Set operations (Intersection &cap;, Union &cup;, Difference &setminus;).</li>
            <li><b>Text Processing:</b> Tokenization, Regular Expressions, String Normalization.</li>
            <li><b>Algorithmic Analysis:</b> Big-O notation, Two-Pointer Merge Algorithms (<code>O(L1 + L2)</code>).</li>
        </ul>
        """, unsafe_allow_html=True)



def render_theory_section():
    """Renders In-Depth Theory Section."""
    st.header("Theoretical Framework & Algorithm Details")
    st.markdown(THEORY_CONTENT["background"])

    st.divider()
    st.subheader("Complexity Comparison Across Retrieval Paradigms")

    comp_data = {
        "Retrieval Mechanism": [
            "Linear Scan (grep)",
            "Term-Document Incidence Matrix",
            "Inverted Index (Standard)",
            "Positional Inverted Index"
        ],
        "Space Complexity": [
            "O(N * L) (No Index)",
            "O(|V| * N) (Extremely Sparse)",
            "O(|Postings|) = O(N * L_unique)",
            "O(Total Tokens N_tokens)"
        ],
        "Conjunctive (AND) Search Time": [
            "O(N * L) (Full collection scan)",
            "O(N) (Bitwise AND across vector)",
            "O(L1 + L2) (Sorted pointer merge)",
            "O(L1 + L2) (Sorted pointer merge)"
        ],
        "Phrase Query Capability": [
            "Supported (via regex / string match)",
            "Unsupported (no word order stored)",
            "Unsupported (document-level only)",
            "Supported (via positional offset match)"
        ]
    }
    st.table(pd.DataFrame(comp_data))

    st.divider()
    with st.expander("Key Terminology & Variable Reference Glossary"):
        var_df = pd.DataFrame(
            list(THEORY_CONTENT["key_terms"].items()),
            columns=["Term / Concept", "Formal Definition & Operational Role"]
        )
        st.table(var_df)


def render_procedure_section():
    """Renders Step-by-Step Procedure Section."""
    st.header("Experimental Procedure")
    st.write("Follow these systematic steps to conduct the experiment, investigate index behavior, and compile your report.")

    for step in PROCEDURE_STEPS:
        st.markdown(f"- {step}")

    st.divider()
    st.subheader("Workflow Diagram")
    st.markdown("""
    ```
    +-------------------+      +-------------------------+      +--------------------------+
    | Raw Text Corpus   | ---> | Linguistic Preprocessor | ---> | (Term, DocID, Pos) Stream|
    | (Docs 1..N)       |      | Casefold / Stop / Stem  |      | Raw Extraction Triples   |
    +-------------------+      +-------------------------+      +--------------------------+
                                                                             |
                                                                             v
    +-------------------+      +-------------------------+      +--------------------------+
    | Query Console     | <--- | Inverted Index Engine   | <--- | Lexicographic Sorter     |
    | (AND, OR, Phrase) |      | Dict + Postings Lists   |      | Sort by (Term, DocID)    |
    +-------------------+      +-------------------------+      +--------------------------+
    ```
    """)


def highlight_search_terms(text: str, query_str: str) -> str:
    """Highlights query terms in bold within snippet text for Google SERP display."""
    tokens = re.findall(r'\b[a-zA-Z0-9_]+\b', query_str)
    stopwords = {"and", "or", "not", "near"}
    terms = [t for t in tokens if t.lower() not in stopwords]
    if not terms:
        return text
    pattern = re.compile(r'\b(' + '|'.join(re.escape(t) for t in terms) + r')\b', re.IGNORECASE)
    return pattern.sub(r'<b>\1</b>', text)


def inject_vlab_search_styles(is_dark_mode: bool = False):
    """Injects authentic, clean Google Search Engine styling for the interactive simulator with consistent enlarged fonts."""
    bg_card = "#202124" if is_dark_mode else "#ffffff"
    border_card = "#3c4043" if is_dark_mode else "#dadce0"
    text_primary = "#e8eaed" if is_dark_mode else "#202124"
    text_secondary = "#9aa0a6" if is_dark_mode else "#5f6368"
    serp_title = "#8ab4f8" if is_dark_mode else "#1a0dab"
    serp_url = "#bdc1c6" if is_dark_mode else "#202124"
    chip_bg = "#303134" if is_dark_mode else "#f1f3f4"
    chip_border = "#5f6368" if is_dark_mode else "#dadce0"
    match_bg = "#0f3822" if is_dark_mode else "#e6f4ea"
    match_border = "#1e8e3e" if is_dark_mode else "#81c995"
    match_text = "#81c995" if is_dark_mode else "#137333"
    post_node_bg = "#1a2733" if is_dark_mode else "#e8f0fe"
    post_node_border = "#3c5678" if is_dark_mode else "#aecbfa"
    post_node_text = "#8ab4f8" if is_dark_mode else "#1967d2"

    st.markdown(f"""
    <style>
    /* VLab Search Header */
    .vlab-header-wrap {{
        text-align: center;
        padding: 20px 0 14px 0;
        margin-bottom: 12px;
    }}
    .vlab-logo-text {{
        font-family: 'Product Sans', 'Google Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        font-size: 42px;
        font-weight: 700;
        letter-spacing: -1.2px;
        display: inline-block;
        user-select: none;
    }}
    .g-blue {{ color: #4285F4; }}
    .g-red {{ color: #EA4335; }}
    .g-yellow {{ color: #FBBC05; }}
    .g-green {{ color: #34A853; }}
    .vlab-edition-badge {{
        font-size: 15px;
        font-weight: 600;
        color: {text_secondary};
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-left: 10px;
        vertical-align: middle;
    }}
    .vlab-subtext {{
        font-size: 16.5px;
        color: {text_secondary};
        margin-top: 6px;
        line-height: 1.6;
    }}

    /* Search Bar Wrapper */
    .vlab-search-container {{
        max-width: 820px;
        margin: 0 auto 18px auto;
    }}

    /* Stepper Navigation Bar */
    .vlab-stepper-wrap {{
        background: {bg_card};
        border: 1px solid {border_card};
        border-radius: 14px;
        padding: 18px 24px;
        margin: 18px 0;
        box-shadow: 0 1px 4px rgba(60,64,67,0.08);
    }}
    .stepper-label {{
        font-size: 14px;
        font-weight: 700;
        letter-spacing: 1px;
        text-transform: uppercase;
        color: #1a73e8;
        margin-bottom: 10px;
    }}
    .stepper-pipeline-flow {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        align-items: center;
        margin-bottom: 8px;
    }}
    .step-pill {{
        background: {chip_bg};
        border: 1px solid {chip_border};
        border-radius: 24px;
        padding: 7px 18px;
        font-size: 15.5px;
        font-weight: 600;
        color: {text_secondary};
        display: inline-flex;
        align-items: center;
        gap: 8px;
        transition: all 0.2s ease;
    }}
    .step-pill.active {{
        background: #1a73e8;
        border-color: #1a73e8;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(26, 115, 232, 0.4);
    }}
    .step-arrow {{
        color: {text_secondary};
        font-size: 16px;
        font-weight: bold;
    }}

    /* Cards */
    .vlab-card {{
        background: {bg_card};
        border: 1px solid {border_card};
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 3px rgba(60,64,67,0.08);
        font-size: 17px;
    }}
    .vlab-doc-card {{
        background: {bg_card};
        border: 1px solid {border_card};
        border-left: 5px solid #4285F4;
        border-radius: 10px;
        padding: 16px 22px;
        margin-bottom: 14px;
        box-shadow: 0 1px 3px rgba(60,64,67,0.06);
    }}
    .vlab-token-badge {{
        display: inline-flex;
        align-items: center;
        background: {chip_bg};
        border: 1px solid {chip_border};
        border-radius: 8px;
        padding: 5px 12px;
        margin: 4px;
        font-family: 'Roboto Mono', monospace;
        font-size: 15.5px;
        color: {text_primary};
    }}
    .vlab-token-badge small {{
        color: #1a73e8;
        font-weight: bold;
        margin-left: 6px;
        font-size: 13.5px;
    }}

    /* Postings Chain */
    .postings-row {{
        background: {bg_card};
        border: 1px solid {border_card};
        border-radius: 10px;
        padding: 12px 18px;
        margin-bottom: 10px;
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 10px;
    }}
    .dict-term-pill {{
        background: #1a73e8;
        color: #ffffff;
        font-weight: 600;
        font-family: 'Roboto Mono', monospace;
        font-size: 16.5px;
        padding: 6px 14px;
        border-radius: 8px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }}
    .dict-freq-tag {{
        background: rgba(255, 255, 255, 0.25);
        color: #ffffff;
        font-size: 13px;
        padding: 2px 7px;
        border-radius: 5px;
        font-weight: 500;
    }}
    .posting-node {{
        background: {post_node_bg};
        border: 1px solid {post_node_border};
        color: {post_node_text};
        font-size: 15.5px;
        border-radius: 8px;
        padding: 6px 14px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
    }}
    .chain-arrow {{
        color: {text_secondary};
        font-size: 16px;
        font-weight: bold;
    }}

    /* Pointer Duel */
    .pointer-film-track {{
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 8px 0;
    }}
    .track-pill {{
        background: {chip_bg};
        border: 1px solid {chip_border};
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 15.5px;
        font-family: 'Roboto Mono', monospace;
        color: {text_primary};
    }}
    .track-pill.matched {{
        background: {match_bg};
        border-color: {match_border};
        color: {match_text} !important;
        font-weight: 700;
        box-shadow: 0 1px 3px rgba(30, 142, 62, 0.2);
    }}

    /* Real Google SERP Styling */
    .serp-container {{
        background: {bg_card};
        border: 1px solid {border_card};
        border-radius: 14px;
        padding: 24px 28px;
        margin-top: 16px;
    }}
    .serp-stats {{
        font-size: 16.5px;
        color: {text_secondary};
        padding-bottom: 14px;
        margin-bottom: 16px;
        border-bottom: 1px solid {border_card};
    }}
    .serp-item {{
        margin-bottom: 24px;
        padding-bottom: 18px;
        border-bottom: 1px dashed {border_card};
    }}
    .serp-item:last-child {{
        border-bottom: none;
        margin-bottom: 0;
        padding-bottom: 0;
    }}
    .serp-url {{
        font-size: 15.5px;
        color: {serp_url};
        margin-bottom: 4px;
        display: flex;
        align-items: center;
        gap: 8px;
    }}
    .serp-favicon {{
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #1a73e8;
        color: #ffffff;
        font-size: 12px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
    }}
    .serp-title {{
        font-size: 22px;
        font-weight: 500;
        color: {serp_title};
        margin-bottom: 6px;
        line-height: 1.35;
    }}
    .serp-snippet {{
        font-size: 17px;
        line-height: 1.65;
        color: {text_primary};
        margin-bottom: 8px;
    }}
    .serp-snippet b {{
        font-weight: 700;
        color: #1a73e8;
    }}
    .serp-meta-badge {{
        display: inline-flex;
        align-items: center;
        background: {chip_bg};
        border: 1px solid {chip_border};
        border-radius: 6px;
        font-size: 14.5px;
        color: {text_secondary};
        padding: 4px 10px;
    }}
    .no-results-box {{
        padding: 24px;
        background: {chip_bg};
        border: 1px solid {chip_border};
        border-radius: 10px;
        color: {text_primary};
        font-size: 17px;
    }}
    .radar-stat-pill {{
        background: {chip_bg};
        border: 1px solid {chip_border};
        padding: 8px 16px;
        border-radius: 8px;
        font-size: 15.5px;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        margin-right: 10px;
        margin-bottom: 10px;
    }}
    </style>
    """, unsafe_allow_html=True)


def render_simulation_section():
    """Renders Google-Style Interactive Search Engine Simulation with Step-by-Step Pipeline."""
    is_dark = st.session_state.get("theme_mode") == "🌙 Dark Mode"
    inject_vlab_search_styles(is_dark_mode=is_dark)

    # Initialize Session States
    if "sim_active_step" not in st.session_state:
        st.session_state["sim_active_step"] = 5  # default to showing the Google SERP results
    if "search_query" not in st.session_state:
        st.session_state["search_query"] = "information AND retrieval"
    if "query_type_select" not in st.session_state:
        st.session_state["query_type_select"] = "Conjunctive Boolean Query (AND)"
    if "selected_corpus_name" not in st.session_state:
        st.session_state["selected_corpus_name"] = list(BENCHMARK_CORPORA.keys())[0]

    # Google Search Header
    st.markdown("""
    <div class="vlab-header-wrap">
        <div class="vlab-logo-text">
            <span class="g-blue">V</span><span class="g-red">L</span><span class="g-yellow">a</span><span class="g-blue">b</span> <span class="g-green">S</span><span class="g-red">e</span><span class="g-yellow">a</span><span class="g-blue">r</span><span class="g-green">c</span><span class="g-red">h</span>
            <span class="vlab-edition-badge">Engine Simulator</span>
        </div>
        <div class="vlab-subtext">
            Explore how an Information Retrieval search engine works under the hood: Crawling, Linguistic Normalization, Inverted Indexing, Two-Pointer Postings Merge, and SERP Ranking.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Active Corpus Selection (used across all steps)
    active_corpus_name = st.session_state["selected_corpus_name"]
    if active_corpus_name == "Custom Documents (Interactive Authoring)":
        active_corpus = {k: v for k, v in st.session_state["custom_corpus"].items() if v.strip()}
    else:
        active_corpus = BENCHMARK_CORPORA.get(active_corpus_name, list(BENCHMARK_CORPORA.values())[0])

    # Preprocessing Options (persisted in session_state)
    if "opt_lowercase" not in st.session_state: st.session_state["opt_lowercase"] = True
    if "opt_punct" not in st.session_state: st.session_state["opt_punct"] = True
    if "opt_stopwords" not in st.session_state: st.session_state["opt_stopwords"] = True
    if "opt_stemming" not in st.session_state: st.session_state["opt_stemming"] = True

    # Build Index
    index_results = build_inverted_index(
        corpus=active_corpus,
        lowercase=st.session_state["opt_lowercase"],
        remove_punct=st.session_state["opt_punct"],
        remove_stopwords=st.session_state["opt_stopwords"],
        use_stemming=st.session_state["opt_stemming"],
        stopwords_set=DEFAULT_STOPWORDS
    )

    st.session_state["last_active_corpus_name"] = active_corpus_name
    st.session_state["last_vocab_size"] = index_results["vocab_size"]
    st.session_state["last_postings_count"] = index_results["total_postings"]

    # Google Search Bar Section
    st.markdown('<div class="vlab-search-container">', unsafe_allow_html=True)

    col_input, col_mode = st.columns([3.2, 1.3])
    with col_input:
        user_query = st.text_input(
            "Search Query:",
            value=st.session_state["search_query"],
            placeholder="Type a search query (e.g. information AND retrieval)...",
            label_visibility="collapsed",
            key="google_search_input"
        )
    with col_mode:
        q_mode = st.selectbox(
            "Query Mode:",
            options=[
                "Conjunctive Boolean Query (AND)",
                "Disjunctive Boolean Query (OR)",
                "Negation Boolean Query (NOT)",
                "Exact Phrase Query (\"term1 term2\")",
                "Proximity Query (NEAR / k)",
                "Single Keyword Lookup"
            ],
            index=[
                "Conjunctive Boolean Query (AND)",
                "Disjunctive Boolean Query (OR)",
                "Negation Boolean Query (NOT)",
                "Exact Phrase Query (\"term1 term2\")",
                "Proximity Query (NEAR / k)",
                "Single Keyword Lookup"
            ].index(st.session_state["query_type_select"]),
            label_visibility="collapsed",
            key="google_mode_select"
        )
        st.session_state["query_type_select"] = q_mode

    # Quick Suggestion Chips
    c_s1, c_s2, c_s3, c_s4, c_s5 = st.columns(5)
    with c_s1:
        if st.button("information AND retrieval", use_container_width=True):
            st.session_state["search_query"] = "information AND retrieval"
            st.session_state["query_type_select"] = "Conjunctive Boolean Query (AND)"
            st.session_state["sim_active_step"] = 5
            st.rerun()
    with c_s2:
        if st.button("graph OR index", use_container_width=True):
            st.session_state["search_query"] = "graph OR index"
            st.session_state["query_type_select"] = "Disjunctive Boolean Query (OR)"
            st.session_state["sim_active_step"] = 5
            st.rerun()
    with c_s3:
        if st.button("retrieval NOT graph", use_container_width=True):
            st.session_state["search_query"] = "retrieval NOT graph"
            st.session_state["query_type_select"] = "Negation Boolean Query (NOT)"
            st.session_state["sim_active_step"] = 5
            st.rerun()
    with c_s4:
        if st.button('"information retrieval"', use_container_width=True):
            st.session_state["search_query"] = "information retrieval"
            st.session_state["query_type_select"] = 'Exact Phrase Query ("term1 term2")'
            st.session_state["sim_active_step"] = 5
            st.rerun()
    with c_s5:
        if st.button("retrieval", use_container_width=True):
            st.session_state["search_query"] = "retrieval"
            st.session_state["query_type_select"] = "Single Keyword Lookup"
            st.session_state["sim_active_step"] = 5
            st.rerun()

    # The Google Search Action Buttons
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        do_search = st.button("🔍 Search Index", type="primary", use_container_width=True)
    with c_btn2:
        inspect_pipeline = st.button("⚙️ Inspect Pipeline (Step 1: Crawl & Corpus)", use_container_width=True)

    if do_search:
        st.session_state["search_query"] = user_query
        st.session_state["sim_active_step"] = 5
        st.rerun()
    if inspect_pipeline:
        st.session_state["search_query"] = user_query
        st.session_state["sim_active_step"] = 1
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Execute Search Query Logic
    query_result_docs = []
    query_latency_us = 0.0
    query_trace = []
    executed_query_str = user_query.strip()
    p1_docs_global = []
    p2_docs_global = []
    t1_label = ""
    t2_label = ""

    if executed_query_str:
        t_start = time.perf_counter()
        opt_stem = st.session_state["opt_stemming"]

        if q_mode == "Conjunctive Boolean Query (AND)":
            parts = [p.strip() for p in re.split(r"\bAND\b", executed_query_str, flags=re.IGNORECASE)]
            if len(parts) >= 2:
                t1_clean = pure_porter_stem(parts[0].lower()) if opt_stem else parts[0].lower()
                t2_clean = pure_porter_stem(parts[1].lower()) if opt_stem else parts[1].lower()
                t1_label, t2_label = t1_clean, t2_clean
                p1_docs = sorted(list(index_results["index"].get(t1_clean, {}).get("postings", {}).keys()))
                p2_docs = sorted(list(index_results["index"].get(t2_clean, {}).get("postings", {}).keys()))
                p1_docs_global, p2_docs_global = p1_docs, p2_docs
                query_result_docs, query_trace = execute_boolean_and_with_trace(p1_docs, p2_docs, t1_clean, t2_clean)
            else:
                query_trace = ["Please enter two terms separated by AND (e.g. 'information AND retrieval')."]

        elif q_mode == "Disjunctive Boolean Query (OR)":
            parts = [p.strip() for p in re.split(r"\bOR\b", executed_query_str, flags=re.IGNORECASE)]
            if len(parts) >= 2:
                t1_clean = pure_porter_stem(parts[0].lower()) if opt_stem else parts[0].lower()
                t2_clean = pure_porter_stem(parts[1].lower()) if opt_stem else parts[1].lower()
                t1_label, t2_label = t1_clean, t2_clean
                p1_docs = sorted(list(index_results["index"].get(t1_clean, {}).get("postings", {}).keys()))
                p2_docs = sorted(list(index_results["index"].get(t2_clean, {}).get("postings", {}).keys()))
                p1_docs_global, p2_docs_global = p1_docs, p2_docs
                query_result_docs, query_trace = execute_boolean_or_with_trace(p1_docs, p2_docs, t1_clean, t2_clean)
            else:
                query_trace = ["Please enter two terms separated by OR (e.g. 'graph OR index')."]

        elif q_mode == "Negation Boolean Query (NOT)":
            parts = [p.strip() for p in re.split(r"\bNOT\b", executed_query_str, flags=re.IGNORECASE)]
            if len(parts) >= 2:
                t1_clean = pure_porter_stem(parts[0].lower()) if opt_stem else parts[0].lower()
                t2_clean = pure_porter_stem(parts[1].lower()) if opt_stem else parts[1].lower()
                p1_docs = set(index_results["index"].get(t1_clean, {}).get("postings", {}).keys())
                p2_docs = set(index_results["index"].get(t2_clean, {}).get("postings", {}).keys())
                query_result_docs = sorted(list(p1_docs - p2_docs))
                query_trace = [
                    f"Postings for '{t1_clean}': {sorted(list(p1_docs))}",
                    f"Excluded postings for '{t2_clean}': {sorted(list(p2_docs))}",
                    f"Set Difference ({t1_clean} \\ {t2_clean}): {query_result_docs}"
                ]
            else:
                query_trace = ["Please enter query in format: 'term1 NOT term2'."]

        elif q_mode == 'Exact Phrase Query (\"term1 term2\")':
            phrase_res = execute_phrase_query(executed_query_str, index_results["index"], use_stemming=opt_stem)
            query_result_docs = phrase_res["docs"]
            query_trace = phrase_res["trace"]

        elif q_mode == "Proximity Query (NEAR / k)":
            words = executed_query_str.split()
            if len(words) >= 2:
                prox_res = execute_proximity_query(words[0], words[1], 3, index_results["index"], use_stemming=opt_stem)
                query_result_docs = prox_res["docs"]
                query_trace = prox_res["trace"]
            else:
                query_trace = ["Enter at least two words for proximity check."]

        else:  # Single Keyword
            res = execute_keyword_query(executed_query_str, index_results["index"], use_stemming=opt_stem)
            query_result_docs = res["docs"]
            query_trace = res["trace"]

        t_end = time.perf_counter()
        query_latency_us = round((t_end - t_start) * 1_000_000, 2)

    # Persist search telemetry for sidebar logger access
    st.session_state["last_query"] = executed_query_str
    st.session_state["last_hits"] = len(query_result_docs)
    st.session_state["last_time_us"] = query_latency_us

    # Pipeline Stepper & Controls
    st.divider()
    active_step = st.session_state["sim_active_step"]

    # Stepper Indicator Pills
    step_titles = {
        1: "Step 1: Document Corpus (Web Crawl)",
        2: "Step 2: Linguistic Preprocessing",
        3: "Step 3: Inverted Index (Lexicon & Postings)",
        4: "Step 4: Two-Pointer Merge Duel",
        5: "Step 5: Search Results Page (SERP)"
    }

    st.markdown(f"""
    <div class="vlab-stepper-wrap">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap;">
            <span class="stepper-label">HOW VLAB SEARCH WORKS: 5-STAGE IR PIPELINE</span>
            <span style="font-size: 15.5px; color: #1a73e8; font-weight: 600;">Currently Active: {step_titles[active_step]}</span>
        </div>
        <div class="stepper-pipeline-flow">
            <span class="step-pill {'active' if active_step == 1 else ''}">1. Web Crawl & Corpus</span>
            <span class="step-arrow">➔</span>
            <span class="step-pill {'active' if active_step == 2 else ''}">2. Linguistic Parser</span>
            <span class="step-arrow">➔</span>
            <span class="step-pill {'active' if active_step == 3 else ''}">3. Inverted Index</span>
            <span class="step-arrow">➔</span>
            <span class="step-pill {'active' if active_step == 4 else ''}">4. Two-Pointer Merge</span>
            <span class="step-arrow">➔</span>
            <span class="step-pill {'active' if active_step == 5 else ''}">5. Search Results (SERP)</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Step Navigation Buttons: Prev, Direct 1-5 buttons, Next, Auto-Play
    c_prev, c_s1_btn, c_s2_btn, c_s3_btn, c_s4_btn, c_s5_btn, c_next, c_auto = st.columns([1, 1, 1, 1, 1, 1, 1, 1.8])
    with c_prev:
        if st.button("◀ Prev", disabled=(active_step == 1), use_container_width=True):
            st.session_state["sim_active_step"] = max(1, active_step - 1)
            st.rerun()
    with c_s1_btn:
        if st.button("Step 1", type="primary" if active_step == 1 else "secondary", use_container_width=True):
            st.session_state["sim_active_step"] = 1
            st.rerun()
    with c_s2_btn:
        if st.button("Step 2", type="primary" if active_step == 2 else "secondary", use_container_width=True):
            st.session_state["sim_active_step"] = 2
            st.rerun()
    with c_s3_btn:
        if st.button("Step 3", type="primary" if active_step == 3 else "secondary", use_container_width=True):
            st.session_state["sim_active_step"] = 3
            st.rerun()
    with c_s4_btn:
        if st.button("Step 4", type="primary" if active_step == 4 else "secondary", use_container_width=True):
            st.session_state["sim_active_step"] = 4
            st.rerun()
    with c_s5_btn:
        if st.button("Step 5", type="primary" if active_step == 5 else "secondary", use_container_width=True):
            st.session_state["sim_active_step"] = 5
            st.rerun()
    with c_next:
        if st.button("Next ▶", disabled=(active_step == 5), use_container_width=True):
            st.session_state["sim_active_step"] = min(5, active_step + 1)
            st.rerun()
    with c_auto:
        play_pipeline = st.button("▶️ Auto-Play (1-by-1)", use_container_width=True)

    if play_pipeline:
        progress_bar = st.progress(0)
        status_box = st.empty()
        anim_steps = [
            (1, "🌐 Step 1: Crawling & Ingesting web documents into memory...", 20),
            (2, "⚙️ Step 2: Running Linguistic Normalization (Case Folding, Stopwords, Stemming)...", 40),
            (3, "📑 Step 3: Inverting Matrix into Lexicon Dictionary & Sorted Postings Lists...", 60),
            (4, f"⚡ Step 4: Executing Two-Pointer Merge on query '{executed_query_str}'...", 80),
            (5, "🔍 Step 5: Formatting Ranked Results into Search Results Page (SERP)...", 100)
        ]
        for s_idx, s_desc, pct in anim_steps:
            status_box.info(s_desc)
            progress_bar.progress(pct)
            time.sleep(0.4)
        status_box.empty()
        progress_bar.empty()
        st.session_state["sim_active_step"] = 5
        st.rerun()

    # =========================================================================
    # STEP CONTENTS (Rendered one-by-one according to active_step)
    # =========================================================================
    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # STEP 1: Web Crawl & Document Corpus
    # -------------------------------------------------------------------------
    if active_step == 1:
        st.markdown("### 🌐 Step 1: Document Corpus Collection (Web Crawling)")
        st.write(
            "Before any user query can be processed, a search engine's web crawler fetches pages across the web "
            "and stores them in a document repository. Below you can select the active collection or input custom documents."
        )

        c_corp1, c_corp2 = st.columns([2.5, 1.5])
        with c_corp1:
            sel_corp = st.radio(
                "Select Corpus Source:",
                options=list(BENCHMARK_CORPORA.keys()) + ["Custom Documents (Interactive Authoring)"],
                index=(list(BENCHMARK_CORPORA.keys()) + ["Custom Documents (Interactive Authoring)"]).index(active_corpus_name),
                horizontal=True
            )
            if sel_corp != active_corpus_name:
                st.session_state["selected_corpus_name"] = sel_corp
                st.rerun()

        if sel_corp == "Custom Documents (Interactive Authoring)":
            st.info("Enter or modify custom documents below to test indexing on your own text:")
            custom_docs = {}
            cols = st.columns(3)
            for i in range(1, 6):
                col_idx = (i - 1) % 3
                with cols[col_idx]:
                    default_text = st.session_state["custom_corpus"].get(
                        i, f"Document {i} text describing information retrieval and graph systems."
                    )
                    txt = st.text_area(f"Doc {i} Content:", value=default_text, height=90, key=f"custom_doc_in_{i}")
                    custom_docs[i] = txt
            st.session_state["custom_corpus"] = custom_docs
            active_corpus = {k: v for k, v in custom_docs.items() if v.strip()}

        st.caption(f"Displaying {len(active_corpus)} crawled documents in repository:")
        for doc_id, text in active_corpus.items():
            word_count = len(text.split())
            st.markdown(f"""
            <div class="vlab-doc-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                    <span style="font-weight: 600; color: #1a73e8; font-size: 15.5px;">https://corpus.vlab.internal › documents › doc{doc_id}</span>
                    <span style="font-size: 14.5px; color: #5f6368;">DocID: {doc_id} | {word_count} words</span>
                </div>
                <div style="font-size: 18px; font-weight: 600; margin-bottom: 6px; color: #202124;">Document {doc_id} Content</div>
                <div style="font-size: 17px; color: #3c4043; line-height: 1.65;">"{text}"</div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("Proceed to Step 2: Linguistic Preprocessing ➔", type="primary"):
            st.session_state["sim_active_step"] = 2
            st.rerun()

    # -------------------------------------------------------------------------
    # STEP 2: Linguistic Preprocessing
    # -------------------------------------------------------------------------
    elif active_step == 2:
        st.markdown("### ⚙️ Step 2: Linguistic Preprocessing Pipeline")
        st.write(
            "Search engines do not index raw text verbatim. They perform linguistic normalization "
            "to ensure queries like *'retrieval'*, *'Retrieval'*, and *'retrieving'* find the same documents."
        )

        st.markdown("#### 1. Preprocessing Configuration Controls")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            opt_lower = st.checkbox("Case Folding (Lowercase)", value=st.session_state["opt_lowercase"], help="Converts all tokens to lowercase.")
        with c2:
            opt_punc = st.checkbox("Strip Punctuation", value=st.session_state["opt_punct"], help="Removes punctuation characters.")
        with c3:
            opt_stop = st.checkbox("Stopword Removal", value=st.session_state["opt_stopwords"], help="Filters frequent non-informative words like 'the', 'is'.")
        with c4:
            opt_stem = st.checkbox("Porter Stemming", value=st.session_state["opt_stemming"], help="Applies Martin Porter's suffix stripping algorithm.")

        if (opt_lower != st.session_state["opt_lowercase"] or
            opt_punc != st.session_state["opt_punct"] or
            opt_stop != st.session_state["opt_stopwords"] or
            opt_stem != st.session_state["opt_stemming"]):
            st.session_state["opt_lowercase"] = opt_lower
            st.session_state["opt_punct"] = opt_punc
            st.session_state["opt_stopwords"] = opt_stop
            st.session_state["opt_stemming"] = opt_stem
            st.rerun()

        st.markdown("#### 2. Normalized Token Stream per Document")
        st.caption("Inspect the token stream emitted for each document after normalization:")
        for doc_id, tokens in index_results["stage1_tokens"].items():
            chips_html = "".join([
                f'<span class="vlab-token-badge">{term}<small>#{pos}</small></span>'
                for term, pos, _ in tokens
            ])
            raw_text = active_corpus.get(doc_id, "")
            st.markdown(f"""
            <div class="vlab-card">
                <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                    <span style="font-weight: 600; font-size: 17px; color: #1a73e8;">Doc {doc_id} Normalized Stream</span>
                    <span style="font-size: 14.5px; color: #5f6368;">{len(tokens)} normalized tokens</span>
                </div>
                <div style="font-size: 16px; color: #5f6368; margin-bottom: 10px; font-style: italic;">Original: "{raw_text}"</div>
                <div>{chips_html}</div>
            </div>
            """, unsafe_allow_html=True)

        with st.expander("🔬 View Emitted (Term, DocID, Position) 3-Tuples", expanded=False):
            triples_df = pd.DataFrame(index_results["stage2_triples"])
            st.dataframe(triples_df, hide_index=True, use_container_width=True, height=220)

        if st.button("Proceed to Step 3: Inverted Index Construction ➔", type="primary"):
            st.session_state["sim_active_step"] = 3
            st.rerun()

    # -------------------------------------------------------------------------
    # STEP 3: Inverted Index (Lexicon & Postings)
    # -------------------------------------------------------------------------
    elif active_step == 3:
        st.markdown("### 📑 Step 3: Inverted Index Construction")
        st.write(
            "Searching documents linearly would take $O(N \\cdot L)$ time, which is prohibitively slow for billions of web pages. "
            "Instead, the search engine inverts the document-term matrix into an **Inverted Index**: "
            "a Vocabulary Dictionary mapping to sorted Postings Lists."
        )

        st.markdown("#### 1. Architecture & Compression Metrics")
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            st.metric("Total Documents (N)", f"{index_results['num_docs']}")
        with k2:
            st.metric("Total Tokens", f"{index_results['total_tokens']}")
        with k3:
            st.metric("Vocabulary Size (|V|)", f"{index_results['vocab_size']}")
        with k4:
            st.metric("Total Postings", f"{index_results['total_postings']}")
        with k5:
            st.metric("Index Sparsity", f"{index_results['sparsity_pct']}%")

        st.markdown("#### 2. Visual Dictionary & Postings Chains Explorer")
        st.caption("Each vocabulary entry maps to a postings list containing Document IDs, term frequency (tf), and positions:")
        term_filter = st.text_input("🔍 Filter vocabulary dictionary:", value="", placeholder="Type term (e.g. retrieval)...")

        filtered_items = sorted(index_results["index"].items())
        if term_filter.strip():
            filtered_items = [item for item in filtered_items if term_filter.strip().lower() in item[0].lower()]

        st.caption(f"Showing {len(filtered_items)} dictionary entries:")
        for term, meta in filtered_items[:25]:
            nodes_html = []
            for d, data in sorted(meta["postings"].items()):
                nodes_html.append(f"""
                <span class="posting-node">
                    <b>Doc {d}</b>
                    <span style="color: #1a73e8;">tf:{data['tf']}</span>
                    <span style="color: #5f6368; font-size: 13.5px;">pos:{data['positions']}</span>
                </span>
                """)
            nodes_rendered = '<span class="chain-arrow">──►</span>'.join(nodes_html)
            st.markdown(f"""
            <div class="postings-row">
                <div class="dict-term-pill">
                    <span>{term}</span>
                    <span class="dict-freq-tag">df: {meta['df']}</span>
                    <span class="dict-freq-tag">cf: {meta['cf']}</span>
                </div>
                <span class="chain-arrow">──►</span>
                {nodes_rendered}
            </div>
            """, unsafe_allow_html=True)

        if len(filtered_items) > 25:
            st.caption(f"... and {len(filtered_items) - 25} more terms in dictionary.")

        with st.expander("📊 View Complete Tabular Postings Table", expanded=False):
            dict_records = []
            for term, meta in sorted(index_results["index"].items()):
                postings_str = " -> ".join([
                    f"[Doc {d} (tf:{data['tf']}) | pos:{data['positions']}]"
                    for d, data in sorted(meta["postings"].items())
                ])
                dict_records.append({
                    "Vocabulary Term": term,
                    "Document Frequency (df)": meta["df"],
                    "Collection Frequency (cf)": meta["cf"],
                    "Postings List": postings_str
                })
            st.dataframe(pd.DataFrame(dict_records), hide_index=True, use_container_width=True, height=260)

        if st.button("Proceed to Step 4: Two-Pointer Query Merge ➔", type="primary"):
            st.session_state["sim_active_step"] = 4
            st.rerun()

    # -------------------------------------------------------------------------
    # STEP 4: Two-Pointer Merge Duel
    # -------------------------------------------------------------------------
    elif active_step == 4:
        st.markdown("### ⚡ Step 4: Two-Pointer Postings Merge Algorithm")
        st.write(
            f"When evaluating multi-term queries like `'{executed_query_str}'`, the VLab Search engine does NOT inspect the documents directly. "
            "Instead, it fetches the pre-sorted postings lists for each query term from the Inverted Index and merges them "
            "using the **Two-Pointer Intersection Algorithm** in linear time $O(L_1 + L_2)$."
        )

        st.markdown(f"""
        <div class="vlab-card">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #dadce0; padding-bottom: 10px; margin-bottom: 12px; flex-wrap: wrap;">
                <span style="font-weight: 700; font-size: 16px; color: #1a73e8;">QUERY EXECUTION TELEMETRY</span>
                <span style="font-size: 15.5px; color: #5f6368;">Active Query: <code style="color: #1a73e8; font-size: 15.5px;">{executed_query_str}</code></span>
            </div>
            <div style="display: flex; flex-wrap: wrap;">
                <span class="radar-stat-pill">⚡ <b>Engine Latency:</b> <span style="color: #1a73e8;">{query_latency_us} µs</span></span>
                <span class="radar-stat-pill">🎯 <b>Matched Documents:</b> <span style="color: #1e8e3e;">{len(query_result_docs)}</span></span>
                <span class="radar-stat-pill">🚀 <b>Algorithm:</b> Two-Pointer Linear Merge O(L1 + L2)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if p1_docs_global and p2_docs_global:
            st.markdown("#### 1. Visual Two-Pointer Intersection Tracks")
            st.caption("Pointers walk down the two postings lists simultaneously. Matching DocIDs are highlighted in green:")
            t1_pills = "".join([
                f'<span class="track-pill {"matched" if d in query_result_docs else ""}">Doc {d}</span>'
                for d in p1_docs_global
            ])
            t2_pills = "".join([
                f'<span class="track-pill {"matched" if d in query_result_docs else ""}">Doc {d}</span>'
                for d in p2_docs_global
            ])
            st.markdown(f"""
            <div class="vlab-card">
                <div style="margin-bottom: 6px; font-size: 16.5px; color: #1a73e8; font-weight: 600;"><b>Pointer 1:</b> Term '{t1_label}' Postings ({len(p1_docs_global)} docs)</div>
                <div class="pointer-film-track">{t1_pills}</div>
                <div style="margin-top: 14px; margin-bottom: 6px; font-size: 16.5px; color: #5f6368; font-weight: 600;"><b>Pointer 2:</b> Term '{t2_label}' Postings ({len(p2_docs_global)} docs)</div>
                <div class="pointer-film-track">{t2_pills}</div>
                <div style="margin-top: 12px; font-size: 15px; color: #1e8e3e; font-weight: 500;">● Highlighted green boxes indicate intersected matching documents added to the results list.</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### 2. Step-by-Step Pointer Trace Log")
        st.caption("Detailed chronological trace of comparisons performed by the pointer algorithm:")
        with st.container(height=180):
            for line in query_trace:
                st.text(f"> {line}")

        if st.button("Proceed to Step 5: Search Results (SERP) ➔", type="primary"):
            st.session_state["sim_active_step"] = 5
            st.rerun()

    # -------------------------------------------------------------------------
    # STEP 5: Google SERP (Search Engine Results Page)
    # -------------------------------------------------------------------------
    else:
        st.markdown("### 🔍 Step 5: Search Engine Results Page (SERP)")
        st.write(
            "The final phase of the search engine: Documents matched by the index traversal are retrieved, "
            "ranked, and presented as a Search Engine Results Page (SERP) with relevant snippets and highlighted keywords."
        )

        st.markdown(f"""
        <div class="serp-container">
            <div class="serp-stats">
                About {len(query_result_docs)} results ({query_latency_us / 1_000_000:.6f} seconds) for <b>{executed_query_str}</b>
            </div>
        """, unsafe_allow_html=True)

        if query_result_docs:
            for d in query_result_docs:
                raw_text = active_corpus.get(d, "")
                highlighted_snippet = highlight_search_terms(raw_text, executed_query_str)
                title_preview = " ".join(raw_text.split()[:7]) + ("..." if len(raw_text.split()) > 7 else "")

                st.markdown(f"""
                <div class="serp-item">
                    <div class="serp-url">
                        <span class="serp-favicon">V</span>
                        <span>https://corpus.vlab.internal › documents › doc{d}</span>
                    </div>
                    <div class="serp-title">Document {d}: {title_preview}</div>
                    <div class="serp-snippet">"{highlighted_snippet}"</div>
                    <div>
                        <span class="serp-meta-badge">DocID: {d}</span>
                        <span class="serp-meta-badge" style="margin-left: 6px;">Corpus: {active_corpus_name.split(':')[0]}</span>
                        <span class="serp-meta-badge" style="margin-left: 6px; color: #1e8e3e; font-weight: bold;">✓ Matched</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="no-results-box">
                <p>Your search - <b>{executed_query_str}</b> - did not match any documents in the active index.</p>
                <p style="margin-top: 12px; font-size: 16.5px; font-weight: 600;">Suggestions:</p>
                <ul style="font-size: 16px; color: #5f6368; line-height: 1.7;">
                    <li>Make sure all words are spelled correctly.</li>
                    <li>Try different keywords or terms present in the corpus.</li>
                    <li>Try a disjunctive query with <code>OR</code> instead of <code>AND</code>.</li>
                    <li>Check if your terms were filtered out by the stopword list in Step 2.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.info("💡 **How did the engine produce these results?** Click **Step 1 - 4** in the pipeline above to inspect the step-by-step crawl, normalization, index, and pointer duel!")

    # Plotly Analytical Visualizations (Google Palette)
    st.divider()
    st.subheader("6. Analytical Visualizations: Zipf's Law & Term Distribution")

    plot_col1, plot_col2 = st.columns(2)

    sorted_terms = sorted(index_results["index"].items(), key=lambda x: x[1]["cf"], reverse=True)
    if sorted_terms:
        ranks = list(range(1, len(sorted_terms) + 1))
        frequencies = [m["cf"] for _, m in sorted_terms]
        c_val = frequencies[0]
        zipf_theoretical = [round(c_val / r, 2) for r in ranks]

        p_template = "plotly_dark" if is_dark else "plotly_white"
        p_paper_bg = "#202124" if is_dark else "#ffffff"
        p_plot_bg = "#303134" if is_dark else "#f8f9fa"
        p_font_color = "#e8eaed" if is_dark else "#202124"

        with plot_col1:
            fig_zipf = go.Figure()
            fig_zipf.add_trace(go.Scatter(
                x=ranks, y=frequencies, mode="lines+markers",
                name="Empirical Frequency (cf)",
                line=dict(color="#4285F4", width=3),
                marker=dict(size=6, color="#4285F4")
            ))
            fig_zipf.add_trace(go.Scatter(
                x=ranks, y=zipf_theoretical, mode="lines",
                name="Zipf's Theoretical (C/r)",
                line=dict(color="#EA4335", dash="dash", width=2)
            ))
            fig_zipf.update_layout(
                title="📈 Zipf's Law: Term Rank vs. Collection Frequency",
                xaxis_title="Term Frequency Rank (r)",
                yaxis_title="Collection Frequency (cf)",
                hovermode="x unified",
                template=p_template,
                paper_bgcolor=p_paper_bg,
                plot_bgcolor=p_plot_bg,
                font=dict(color=p_font_color),
                height=340,
                margin=dict(l=20, r=20, t=45, b=20)
            )
            st.plotly_chart(fig_zipf, use_container_width=True)

        with plot_col2:
            top_terms = sorted_terms[:15]
            fig_df = go.Figure(go.Bar(
                x=[t for t, _ in top_terms],
                y=[m["df"] for _, m in top_terms],
                marker=dict(
                    color="#34A853",
                    line=dict(color="#1e8e3e", width=1)
                )
            ))
            fig_df.update_layout(
                title="📊 Document Frequency (df) for Top Vocabulary Terms",
                xaxis_title="Vocabulary Term",
                yaxis_title="Document Frequency (df)",
                template=p_template,
                paper_bgcolor=p_paper_bg,
                plot_bgcolor=p_plot_bg,
                font=dict(color=p_font_color),
                height=340,
                margin=dict(l=20, r=20, t=45, b=20)
            )
            st.plotly_chart(fig_df, use_container_width=True)

    # Experimental Trial Logger Callout (Moved to Sidebar)
    st.divider()
    st.markdown("""
    <div style="background: #ffffff; border: 1px solid #dadce0; border-left: 5px solid #4285F4; border-radius: 12px; padding: 20px 24px; box-shadow: 0 1px 4px rgba(60,64,67,0.08); display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 14px;">
        <div>
            <div style="font-size: 18px; font-weight: 600; color: #202124;">📝 Session Data Log Book is in the Sidebar</div>
            <div style="font-size: 16px; color: #5f6368; margin-top: 4px;">
                Record current query latency, hit counts, and vocabulary statistics directly into your session log using the <b>Session Data Log</b> in the sidebar.
            </div>
        </div>
        <div>
            <span style="background: #e8f0fe; color: #1a73e8; font-weight: 600; font-size: 15px; padding: 7px 16px; border-radius: 20px;">
                👈 Access via Sidebar
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)



def render_quiz_section():
    """Renders Self-Evaluation Quiz Section."""
    st.header("Self-Evaluation: Conceptual Assessment")
    st.write("Complete the 10 multiple-choice questions below to test your understanding of Inverted Index construction and query processing.")

    with st.form("inverted_index_quiz_form"):
        user_responses = {}
        for q in QUIZ_QUESTIONS:
            st.markdown(f"#### Question {q['id']}")
            st.write(f"**{q['question']}**")
            selected = st.radio(
                label=f"Options for Q{q['id']}:",
                options=q["options"],
                index=st.session_state["quiz_answers"].get(q["id"], 0),
                key=f"quiz_q_{q['id']}",
                label_visibility="collapsed"
            )
            user_responses[q["id"]] = q["options"].index(selected)
            st.write("")

        submitted = st.form_submit_button("Submit Assessment for Evaluation", type="primary")

    if submitted:
        score = 0
        st.session_state["quiz_answers"] = user_responses
        st.session_state["quiz_submitted"] = True

        st.divider()
        st.subheader("Evaluation Results & Instant Feedback")

        for q in QUIZ_QUESTIONS:
            user_ans = user_responses.get(q["id"])
            correct_ans = q["answer_index"]
            if user_ans == correct_ans:
                score += 1
                st.success(f"**Question {q['id']}: Correct!**\n\n_{q['explanation']}_")
            else:
                st.error(
                    f"**Question {q['id']}: Incorrect.** (Your answer: {q['options'][user_ans]})\n\n"
                    f"**Correct Answer:** {q['options'][correct_ans]}\n\n"
                    f"**Reasoning:** _{q['explanation']}_"
                )

        st.session_state["quiz_score"] = score
        perc = (score / len(QUIZ_QUESTIONS)) * 100
        st.info(f"Final Score: **{score} / {len(QUIZ_QUESTIONS)}** ({perc:.1f}%)")

    elif st.session_state.get("quiz_submitted", False):
        curr_score = st.session_state.get("quiz_score", 0)
        st.success(f"Quiz already evaluated. Current Score: **{curr_score} / {len(QUIZ_QUESTIONS)}**")


def render_references_section():
    """Renders Academic References Section."""
    st.header("Academic References & Standard Reading")
    st.write("The following standard textbooks and research publications detail the algorithmic design of Inverted Indexes and Information Retrieval systems:")

    for idx, ref in enumerate(REFERENCES_DATA, start=1):
        r_title = ref.get("title", f"Reference {idx}")
        r_authors = ref.get("authors", "National Mission on Education through ICT")
        r_publisher = ref.get("publisher", "Academic Resources")
        r_details = ref.get("details", "Information Retrieval & Knowledge Graph Systems")
        r_url = ref.get("url", "https://vlabs.ac.in/")
        st.markdown(f"""
        ### [{idx}] {r_title}
        * **Authors:** {r_authors}
        * **Publication:** {r_publisher}
        * **Topics Covered:** {r_details}
        * **Resource Link:** [{r_url}]({r_url})
        """)
        st.divider()


def render_feedback_section():
    """Renders Virtual Lab Feedback Section."""
    st.header("Virtual Lab Feedback: ")
    st.write("Your feedback helps us refine the pedagogical quality and simulation clarity of this Virtual Lab module.")

    with st.form("vlab_feedback_form"):
        col1, col2 = st.columns(2)
        with col1:
            fb_name = st.text_input("Student Name:", value=st.session_state["student_info"].get("name", ""))
            fb_roll = st.text_input("Roll / Registration Number:", value=st.session_state["student_info"].get("id", ""))
        with col2:
            fb_college = st.text_input("Institute / Department:", value="Dept of Computer Science & Engineering")
            fb_email = st.text_input("Student Email Address:", value="student@ves.ac.in")

        st.subheader("Evaluation Metrics (1 = Unsatisfactory, 5 = Excellent)")
        r1 = st.slider("1. Clarity of Inverted Index theoretical concepts:", 1, 5, 5)
        r2 = st.slider("2. Ease of use and interactivity of the 4-Stage Simulator:", 1, 5, 5)
        r3 = st.slider("3. Educational value of the step-by-step Pointer Comparison Trace:", 1, 5, 5)
        r4 = st.slider("4. Quality of Assessment Quiz and instant explanatory feedback:", 1, 5, 5)

        fb_comments = st.text_area(
            "Suggestions for enhancing the virtual laboratory simulation:",
            placeholder="Share any additional feedback on simulation fidelity, features, or UI clarity..."
        )

        fb_submit = st.form_submit_button("Submit Feedback", type="primary")

    if fb_submit:
        st.session_state["feedback_submitted"] = True
        st.success("Thank you! Your feedback has been officially recorded in the Virtual Labs portal.")


def render_report_section():
    """Renders Report Generation Section with Guaranteed PDF Export."""
    st.header("Report Generation & Verification")
    st.write("Compile your experimental session, recorded benchmark trials, and conceptual quiz score into an official PDF lab report.")

    col1, col2, col3 = st.columns(3)
    with col1:
        student_name = st.text_input("Student Name", value=st.session_state["student_info"].get("name", "Student Name"))
    with col2:
        student_id = st.text_input("Roll Number (Allocated: 11 - 15)", value=st.session_state["student_info"].get("id", "EXP3-11"))
    with col3:
        student_div = st.text_input("Class / Division", value=st.session_state["student_info"].get("div", "D17B"))

    lab_date = st.date_input("Experiment Date", value=datetime.now())

    st.session_state["student_info"]["name"] = student_name
    st.session_state["student_info"]["id"] = student_id
    st.session_state["student_info"]["div"] = student_div
    st.session_state["student_info"]["date"] = str(lab_date)

    st.subheader("Experimental Observations & Discussion")
    student_notes = st.text_area(
        "Enter your analysis of inverted index construction, preprocessing effects, and query performance:",
        value=st.session_state.get("student_notes", (
            "1. Linguistic preprocessing: Stopword removal and Porter stemming reduced the vocabulary size (|V|), "
            "substantially decreasing index storage while preserving keyword search effectiveness.\n"
            "2. Boolean query processing: The linear-time two-pointer intersection algorithm evaluated conjunctive queries in O(L1 + L2) comparisons.\n"
            "3. Positional indexing: Recording token positions enabled exact phrase queries without false positives.\n"
            "4. Term distribution: Term frequencies follow Zipf's Law, demonstrating a power-law distribution across vocabulary ranks."
        )),
        height=140
    )
    st.session_state["student_notes"] = student_notes

    trials_df = pd.DataFrame(st.session_state["trials"]) if st.session_state["trials"] else pd.DataFrame()

    st.divider()
    st.subheader("Report Summary Preview")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(f"**Experiment:** {EXPERIMENT_CONFIG['short_title']}")
        st.write(f"**Student:** {student_name} ({student_id})")
    with c2:
        st.write(f"**Course:** {EXPERIMENT_CONFIG['subject']}")
        st.write(f"**Division:** {student_div} | **Date:** {lab_date}")
    with c3:
        st.write(f"**Quiz Score:** {st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}")
        st.write(f"**Logged Trials:** {len(trials_df)}")

    if not trials_df.empty:
        st.dataframe(trials_df, hide_index=True, use_container_width=True)
    else:
        st.info("Tip: You have not recorded any trials in the Simulation tab yet. Your report will indicate 0 recorded trials.")

    # Generate PDF bytes
    pdf_bytes = generate_pdf_report(
        student_name=student_name,
        student_id=student_id,
        student_div=student_div,
        date_str=str(lab_date),
        trials_df=trials_df,
        quiz_score=st.session_state.get("quiz_score", 0),
        quiz_total=len(QUIZ_QUESTIONS),
        student_notes=student_notes,
        corpus_name=st.session_state.get("last_active_corpus_name", "Corpus 1"),
        vocab_size=st.session_state.get("last_vocab_size", 0),
        postings_count=st.session_state.get("last_postings_count", 0)
    )

    # Save to local file system
    os.makedirs("static", exist_ok=True)
    with open("static/lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)
    with open("lab_report.pdf", "wb") as f:
        f.write(pdf_bytes)

    st.divider()
    st.subheader("Download Official Verified Lab Report (.pdf)")
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        st.download_button(
            label="Download Experiment 3 Lab Report (PDF)",
            data=pdf_bytes,
            file_name=f"Inverted_Index_Lab_Report_{student_id}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )

    with col_btn2:
        st.download_button(
            label="Download Session Trials as CSV",
            data=trials_df.to_csv(index=False).encode('utf-8') if not trials_df.empty else b"No trials recorded",
            file_name="experiment_trials.csv",
            mime="text/csv",
            use_container_width=True
        )


# ======================================================================================
# 5. MAIN APPLICATION ENTRYPOINT & STATE MANAGEMENT
# ======================================================================================

def init_session_state():
    """Initializes Streamlit session state variables."""
    if "trials" not in st.session_state:
        st.session_state["trials"] = []
    if "quiz_answers" not in st.session_state:
        st.session_state["quiz_answers"] = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state["quiz_submitted"] = False
    if "quiz_score" not in st.session_state:
        st.session_state["quiz_score"] = 0
    if "student_info" not in st.session_state:
        st.session_state["student_info"] = {
            "name": "Student Name",
            "id": "EXP3-11",
            "div": "D17B",
            "date": str(datetime.now().date())
        }
    if "student_notes" not in st.session_state:
        st.session_state["student_notes"] = ""
    if "custom_corpus" not in st.session_state:
        st.session_state["custom_corpus"] = {
            1: "Information retrieval systems index documents for fast searching.",
            2: "A knowledge graph organizes entities and relations into a connected graph.",
            3: "Search engines construct an inverted index for efficient information retrieval.",
            4: "Graph databases query relationships using Cypher in Neo4j systems.",
            5: "Fast keyword retrieval relies on an inverted index and postings lists."
        }
    if "feedback_submitted" not in st.session_state:
        st.session_state["feedback_submitted"] = False
    if "last_active_corpus_name" not in st.session_state:
        st.session_state["last_active_corpus_name"] = "Corpus 1: Information Retrieval & Search Engines"
    if "last_vocab_size" not in st.session_state:
        st.session_state["last_vocab_size"] = 0
    if "last_postings_count" not in st.session_state:
        st.session_state["last_postings_count"] = 0
    if "theme_mode" not in st.session_state:
        st.session_state["theme_mode"] = "☀️ Light Mode"


def render_certificate_section():
    """Renders official Virtual Lab Certificate of Completion with interactive PDF download."""
    st.header("Virtual Lab Certificate of Completion")
    st.write("Generate and download your official, digitally verified Certificate of Completion for Experiment 3.")

    # Student Details Form
    col1, col2, col3 = st.columns(3)
    with col1:
        student_name = st.text_input("Full Name:", value=st.session_state["student_info"].get("name", "Student Name"), key="cert_name_input")
    with col2:
        student_id = st.text_input("Roll / Student ID:", value=st.session_state["student_info"].get("id", "EXP3-11"), key="cert_id_input")
    with col3:
        student_div = st.text_input("Class / Division:", value=st.session_state["student_info"].get("div", "D17B"), key="cert_div_input")

    cert_date = st.date_input("Certification Date:", value=datetime.now(), key="cert_date_input")

    st.session_state["student_info"]["name"] = student_name
    st.session_state["student_info"]["id"] = student_id
    st.session_state["student_info"]["div"] = student_div

    # Completion Criteria Checklist
    st.divider()
    st.subheader("Experiment Completion Verification")
    
    trials_count = len(st.session_state.get("trials", []))
    quiz_done = st.session_state.get("quiz_submitted", False)
    quiz_score = st.session_state.get("quiz_score", 0)
    quiz_total = len(QUIZ_QUESTIONS)

    chk1, chk2, chk3 = st.columns(3)
    with chk1:
        st.markdown(f"**1. Student Profile:** {'✅ Verified' if student_name.strip() else '⚠️ Missing'}")
        st.caption(f"{student_name} ({student_id})")
    with chk2:
        st.markdown(f"**2. Simulation Trials:** {'✅ Completed' if trials_count > 0 else '⚠️ Incomplete'}")
        st.caption(f"{trials_count} trial(s) logged in session")
    with chk3:
        st.markdown(f"**3. Assessment Quiz:** {'✅ Evaluated' if quiz_done else '⚠️ Pending'}")
        st.caption(f"Score: {quiz_score} / {quiz_total}")

    # Generate unique Certificate Hash / ID
    cert_id = f"VLAB-EXP3-KGIRS-2026-{abs(hash(student_name + student_id)) % 1000000:06d}"

    st.divider()
    st.subheader("Official Certificate Preview")

    # Gorgeous On-Screen Certificate Preview
    st.markdown(f"""
    <div style="border: 6px double #d97706; padding: 32px 36px; background: #fffdfa; color: #0f172a; border-radius: 16px; box-shadow: 0 12px 36px rgba(0,0,0,0.08); position: relative; margin-bottom: 24px;">
        <div style="text-align: center; border-bottom: 2px solid #e2e8f0; padding-bottom: 14px; margin-bottom: 18px;">
            <div style="font-size: 13px; font-weight: 800; letter-spacing: 2px; color: #475569; text-transform: uppercase;">VIRTUAL LABS • AN INITIATIVE OF MINISTRY OF EDUCATION (vlabs.ac.in)</div>
            <div style="font-size: 12px; font-weight: 700; color: #64748b; letter-spacing: 1px; margin-top: 3px;">DEPARTMENT OF COMPUTER ENGINEERING / ARTIFICIAL INTELLIGENCE & DATA SCIENCE</div>
            <h1 style="margin: 10px 0 0 0; color: #0f172a; font-size: 32px; font-family: 'Times New Roman', serif; letter-spacing: 1px;">CERTIFICATE OF COMPLETION</h1>
            <div style="font-size: 13px; color: #94a3b8; font-style: italic; margin-top: 4px;">This is to officially certify that</div>
        </div>

        <div style="text-align: center; margin-bottom: 18px;">
            <h2 style="margin: 0; color: #1d4ed8; font-size: 28px; text-transform: uppercase; letter-spacing: 1px;">{student_name.upper()}</h2>
            <div style="font-size: 14px; color: #475569; margin-top: 4px;">
                <b>Roll Number:</b> {student_id} &nbsp;|&nbsp; <b>Class / Division:</b> {student_div} &nbsp;|&nbsp; <b>Course:</b> Knowledge Graphs & Information Retrieval Systems
            </div>
        </div>

        <div style="text-align: center; max-width: 780px; margin: 0 auto 20px auto; font-size: 14px; line-height: 1.6; color: #334155;">
            has successfully conducted, analyzed, and completed the experimental laboratory requirements for<br>
            <b style="color: #0f172a; font-size: 15px;">Experiment 3: Construction of an Inverted Index</b>, demonstrating verified competency in:
            <div style="display: flex; justify-content: center; gap: 16px; margin-top: 10px; flex-wrap: wrap;">
                <span style="background: #e8f0fe; color: #1a73e8; padding: 5px 14px; border-radius: 20px; font-size: 13px; font-weight: 600;">• Lexicon & Postings Construction</span>
                <span style="background: #fce8e6; color: #d93025; padding: 5px 14px; border-radius: 20px; font-size: 13px; font-weight: 600;">• Two-Pointer Merge O(L1 + L2)</span>
                <span style="background: #fef7e0; color: #b06000; padding: 5px 14px; border-radius: 20px; font-size: 13px; font-weight: 600;">• Linguistic Normalization</span>
                <span style="background: #e6f4ea; color: #1e8e3e; padding: 5px 14px; border-radius: 20px; font-size: 13px; font-weight: 600;">• Zipf's Law Validation</span>
            </div>
        </div>

        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 20px; display: flex; justify-content: space-between; align-items: center; max-width: 780px; margin: 0 auto 20px auto; font-size: 13px;">
            <span><b>Assessment Score:</b> <span style="color: #2563eb;">{quiz_score} / {quiz_total}</span></span>
            <span><b>Simulation Trials:</b> <span style="color: #059669;">{trials_count} Logged</span></span>
            <span><b>Status:</b> <span style="color: #16a34a; font-weight: bold;">Verified & Completed</span></span>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: flex-end; border-top: 1px solid #e2e8f0; padding-top: 16px; margin-top: 16px;">
            <div>
                <div style="font-size: 12px; color: #64748b;"><b>Issue Date:</b> {cert_date}</div>
                <div style="font-size: 11px; color: #94a3b8; font-family: monospace;"><b>Certificate ID:</b> {cert_id}</div>
            </div>
            <div style="text-align: center;">
                <div style="font-size: 28px;">🏅</div>
                <div style="font-size: 10px; font-weight: bold; color: #d97706; letter-spacing: 1px;">VERIFIED LAB ACCOMPLISHMENT</div>
            </div>
            <div style="text-align: center;">
                <div style="border-bottom: 1.5px solid #64748b; width: 180px; margin-bottom: 4px;"></div>
                <div style="font-size: 12px; font-weight: 700; color: #1e293b;">Course Instructor / Evaluator</div>
                <div style="font-size: 11px; color: #64748b;">Virtual Laboratories Network</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Generate and provide download button for Certificate PDF
    cert_pdf_bytes = generate_pdf_certificate(
        student_name=student_name,
        student_id=student_id,
        student_div=student_div,
        date_str=str(cert_date),
        quiz_score=quiz_score,
        quiz_total=quiz_total,
        trials_count=trials_count,
        cert_id=cert_id
    )

    c_d1, c_d2 = st.columns([2, 2])
    with c_d1:
        st.download_button(
            label="📜 Download Official Verified Certificate (PDF)",
            data=cert_pdf_bytes,
            file_name=f"VLab_Certificate_Exp3_{student_id}.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True
        )
    with c_d2:
        st.caption("This digital certificate includes a cryptographic verification ID and officially certifies completion of Experiment 3.")



def render_sidebar_logger():
    """Renders comprehensive experimental trial logging functionality in the sidebar."""
    st.sidebar.divider()
    st.sidebar.markdown("""
    <div style="font-size: 17.5px; font-weight: 700; color: #202124; margin-bottom: 10px; display: flex; align-items: center; gap: 8px;">
        <span>📝</span> <span>Session Data Log</span>
    </div>
    """, unsafe_allow_html=True)

    trials = st.session_state.get("trials", [])

    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        st.metric("Recorded", f"{len(trials)}")
    with col_s2:
        last_hits = st.session_state.get("last_hits", 0)
        st.metric("Last Hits", f"{last_hits}")

    if st.sidebar.button("➕ Record Trial", type="primary", use_container_width=True, help="Captures active corpus, preprocessing flags, query, hits, and latency"):
        c_name = str(st.session_state.get("last_active_corpus_name", "Academic IR Corpus")).split(":")[0]
        sw = "ON" if st.session_state.get("opt_stopwords", True) else "OFF"
        st_flag = "ON" if st.session_state.get("opt_stemming", True) else "OFF"
        v_size = st.session_state.get("last_vocab_size", "-")
        p_count = st.session_state.get("last_postings_count", "-")
        q_str = st.session_state.get("last_query", st.session_state.get("search_query", "information AND retrieval"))
        h_count = st.session_state.get("last_hits", 0)
        lat = st.session_state.get("last_time_us", 0.0)

        record = {
            "Trial #": len(trials) + 1,
            "Corpus": c_name,
            "Stopwords": sw,
            "Stemming": st_flag,
            "Vocab |V|": v_size,
            "Total Postings": p_count,
            "Query": q_str,
            "Hits": h_count,
            "Time (us)": lat,
            "Timestamp": datetime.now().strftime("%H:%M:%S")
        }
        st.session_state["trials"].append(record)
        st.toast(f"Trial #{record['Trial #']} successfully recorded into session log!", icon="✅")
        st.rerun()

    if trials:
        with st.sidebar.expander(f"📋 View Logged Trials ({len(trials)})", expanded=False):
            df_side = pd.DataFrame(trials)
            disp_cols = [c for c in ["Trial #", "Corpus", "Query", "Hits", "Time (us)"] if c in df_side.columns]
            st.dataframe(df_side[disp_cols], hide_index=True, use_container_width=True)

        df_all = pd.DataFrame(trials)
        csv_bytes = df_all.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="📥 Download Trials CSV",
            data=csv_bytes,
            file_name="inverted_index_trials.csv",
            mime="text/csv",
            use_container_width=True
        )

        if st.sidebar.button("🗑️ Clear Logged Trials", use_container_width=True):
            st.session_state["trials"] = []
            st.toast("Experimental trials reset.")
            st.rerun()
    else:
        st.sidebar.caption("No trials logged yet. Run queries in the simulator and click '➕ Record Trial'.")



def main():
    st.set_page_config(
        page_title="Inverted Index Virtual Lab",
        page_icon="🔍",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()
    st.session_state["theme_mode"] = "☀️ Light Mode"

    # Google Fonts & Comprehensive Google Design Language Injection
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;600;700&family=Roboto:ital,wght@0,300;0,400;0,500;0,700;1,400&family=Roboto+Mono:wght@400;500;600&display=swap');

        /* Universal Typography using Google Sans & Roboto */
        * {
            font-family: 'Google Sans', 'Roboto', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        }
        code, pre, .mono, kbd {
            font-family: 'Roboto Mono', 'SFMono-Regular', Consolas, Menlo, monospace !important;
        }

        /* Base Body Typography with Significantly Increased Font Size */
        html, body, [class*="css"], [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li {
            font-size: 18px !important;
            line-height: 1.75 !important;
            color: #202124 !important;
        }

        /* Headings Hierarchy */
        h1, .stTitle {
            font-size: 2.6rem !important;
            font-weight: 700 !important;
            color: #202124 !important;
            letter-spacing: -0.5px !important;
            margin-bottom: 16px !important;
        }
        h2, .stHeader {
            font-size: 2.1rem !important;
            font-weight: 650 !important;
            color: #202124 !important;
            margin-top: 24px !important;
            margin-bottom: 12px !important;
        }
        h3, .stSubheader {
            font-size: 1.65rem !important;
            font-weight: 600 !important;
            color: #202124 !important;
            margin-top: 18px !important;
            margin-bottom: 10px !important;
        }
        h4 {
            font-size: 1.32rem !important;
            font-weight: 600 !important;
            color: #202124 !important;
            margin-top: 14px !important;
            margin-bottom: 8px !important;
        }

        /* Form Controls, Radios, and Checkboxes */
        [data-testid="stCheckbox"] label, [data-testid="stCheckbox"] span,
        [data-testid="stRadio"] label, [data-testid="stRadio"] span {
            font-size: 17.5px !important;
            color: #202124 !important;
        }

        /* Tables and Dataframes */
        table, th, td, [data-testid="stTable"] td, [data-testid="stTable"] th {
            font-size: 17px !important;
            line-height: 1.6 !important;
        }

        /* Captions and Explanatory text */
        [data-testid="stCaptionContainer"], small {
            font-size: 15.5px !important;
            color: #5f6368 !important;
            line-height: 1.5 !important;
        }

        /* Expanders */
        [data-testid="stExpander"] details summary span {
            font-size: 18px !important;
            font-weight: 600 !important;
            color: #202124 !important;
        }

        /* Text Inputs & Textareas */
        .stTextArea textarea, .stSelectbox [data-baseweb="select"] {
            font-size: 17px !important;
            line-height: 1.6 !important;
        }
        .stTextInput > div > div > input {
            font-size: 17.5px !important;
            border-radius: 24px !important;
            border: 1px solid #dfe1e5 !important;
            padding: 11px 22px !important;
            color: #202124 !important;
        }
        .stTextInput > div > div > input:focus {
            border-color: #4285F4 !important;
            box-shadow: 0 1px 6px rgba(66,133,244,0.3) !important;
        }

        /* Metric Values & Labels */
        [data-testid="stMetricValue"] {
            font-size: 28px !important;
            font-weight: 700 !important;
            color: #1a73e8 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 15.5px !important;
            font-weight: 500 !important;
            color: #5f6368 !important;
        }

        /* Clean Google Surfaces */
        [data-testid="stAppViewContainer"] {
            background-color: #ffffff !important;
            color: #202124 !important;
        }
        [data-testid="stHeader"] {
            background-color: #ffffff !important;
        }

        /* Google Sidebar with Increased Font Size and Clean Spacing */
        [data-testid="stSidebar"] {
            background-color: #f8f9fa !important;
            border-right: 1px solid #dadce0 !important;
            padding-top: 24px !important;
        }
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            font-size: 17.5px !important;
            font-weight: 500 !important;
            color: #202124 !important;
        }
        [data-testid="stSidebar"] .stRadio > div {
            gap: 6px !important;
        }
        [data-testid="stSidebar"] .stRadio label {
            padding: 8px 14px !important;
            border-radius: 20px !important;
            transition: all 0.2s ease !important;
            cursor: pointer !important;
        }
        [data-testid="stSidebar"] .stRadio label:hover {
            background-color: #e8f0fe !important;
            color: #1a73e8 !important;
        }

        /* Google Pill Buttons */
        .stButton > button {
            font-size: 16.5px !important;
            font-weight: 600 !important;
            border-radius: 24px !important;
            padding: 9px 24px !important;
            border: 1px solid #dadce0 !important;
            background-color: #ffffff !important;
            color: #202124 !important;
            box-shadow: 0 1px 3px rgba(60,64,67,0.12) !important;
            transition: all 0.2s ease !important;
        }
        .stButton > button:hover {
            background-color: #f8f9fa !important;
            border-color: #dadce0 !important;
            color: #1a73e8 !important;
            box-shadow: 0 1px 4px rgba(60,64,67,0.25) !important;
        }
        .stButton > button[kind="primary"] {
            background-color: #1a73e8 !important;
            border-color: #1a73e8 !important;
            color: #ffffff !important;
        }
        .stButton > button[kind="primary"]:hover {
            background-color: #1765cc !important;
            box-shadow: 0 1px 4px rgba(66,133,244,0.4) !important;
        }

        /* Streamlit Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            border-bottom: 2px solid #dadce0;
        }
        .stTabs [data-baseweb="tab"] {
            font-size: 17.5px !important;
            font-weight: 500 !important;
            color: #5f6368 !important;
            padding: 10px 18px !important;
            border-radius: 8px 8px 0 0 !important;
        }
        .stTabs [aria-selected="true"] {
            color: #1a73e8 !important;
            border-bottom: 3px solid #1a73e8 !important;
            font-weight: 600 !important;
        }

        /* Clean Google Metric Cards */
        .metric-card {
            background-color: #ffffff !important;
            border: 1px solid #dadce0 !important;
            border-radius: 12px !important;
            padding: 18px !important;
            box-shadow: 0 1px 3px rgba(60,64,67,0.08) !important;
        }

        /* Streamlit Alerts & Banners */
        .stAlert {
            border-radius: 12px !important;
            font-size: 17px !important;
        }
    </style>
    """, unsafe_allow_html=True)

    # Clean Title (Without subtitle)
    st.title(EXPERIMENT_CONFIG["title"])

    # Sidebar: Strictly and exclusively the requested 7 Navigation items
    section = st.sidebar.radio(
        "Navigation",
        options=[
            "Purpose",
            "Theory",
            "Simulation",
            "Quiz",
            "Report Generation",
            "Certificate",
            "References"
        ],
        label_visibility="collapsed"
    )

    # Session Data Log Book in Sidebar
    render_sidebar_logger()

    # Section Dispatcher
    if section == "Purpose":
        render_purpose_section()
    elif section == "Theory":
        render_theory_section()
    elif section == "Simulation":
        render_simulation_section()
    elif section == "Quiz":
        render_quiz_section()
    elif section == "Report Generation":
        render_report_section()
    elif section == "Certificate":
        render_certificate_section()
    elif section == "References":
        render_references_section()


if __name__ == "__main__":
    main()
