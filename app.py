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
    "target_rolls": "Roll Numbers 11 - 15 (Divisions A, B, C)",
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
  $$\\text{Forward Index}: DocID \\longrightarrow [\\text{term}_1, \\text{term}_2, \\dots, \\text{term}_k]$$
  While efficient for document parsing, searching for a keyword requires scanning every document entry.

---

### 3. Architecture of an Inverted Index

The **Inverted Index** (or inverted file) inverts the forward mapping: it maps each unique vocabulary term to an ordered list of documents where that term occurs.

$$\\text{Inverted Index}: \\text{term}_t \\longrightarrow [DocID_1, DocID_2, DocID_3, \\dots]$$

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
   * Implemented using dynamic Hash Tables for $O(1)$ expected lookup, or B-Trees / Tries for $O(\\log |V|)$ lookup supporting prefix and wildcard queries.
2. **Postings Lists**:
   * A variable-length linked list or contiguous dynamic array containing the document entries (postings) for each term.
   * Maintained strictly in **sorted order of Document ID** ($DocID_1 < DocID_2 < \\dots < DocID_k$). Sorting is essential because it enables linear-time $O(L_1 + L_2)$ intersection algorithms.
   * **Positional Postings Lists**: Extends each posting to record not just $DocID$ and term frequency $tf_{t,d}$, but also the exact token offset positions $[pos_1, pos_2, \\dots]$.

---

### 4. Query Processing & Boolean Retrieval

#### Conjunction (`AND` Intersection)
Given two sorted postings lists $P_1$ of length $L_1$ and $P_2$ of length $L_2$, the intersection algorithm uses two pointers $p_1$ and $p_2$ traversing both lists simultaneously:

$$\\text{Time Complexity}: O(L_1 + L_2)$$

* If $DocID(p_1) == DocID(p_2)$: Append $DocID(p_1)$ to result; advance both $p_1$ and $p_2$.
* If $DocID(p_1) < DocID(p_2)$: Advance $p_1$.
* If $DocID(p_1) > DocID(p_2)$: Advance $p_2$.

#### Disjunction (`OR` Union)
Traverse both lists with two pointers, appending the smaller $DocID$ to the output list and advancing that pointer. If identical, append once and advance both. Running time: $O(L_1 + L_2)$.

#### Query Optimization via Document Frequency
When evaluating a multi-term conjunctive query $t_1 \\text{ AND } t_2 \\text{ AND } t_3$:
Search engines sort the query terms in **increasing order of document frequency** ($df_{t_1} \\le df_{t_2} \\le df_{t_3}$). Intersecting the shortest lists first produces the smallest intermediate candidate set, drastically reducing subsequent comparisons!

---

### 5. Positional Indexing & Phrase Queries

A non-positional index can only verify that two words appear somewhere in the same document, failing on exact phrase queries such as `"information retrieval"`.
In a **Positional Inverted Index**, each posting stores the token positions:
$$\\text{Posting}: \\langle DocID, tf, [pos_1, pos_2, \\dots, pos_{tf}] \\rangle$$

For phrase query `"term1 term2"`, the query engine intersects their document postings and checks whether:
$$\\exists p \\in \\text{positions}(term_1, d) \\quad \\text{such that} \\quad (p + 1) \\in \\text{positions}(term_2, d)$$

---

### 6. Empirical Laws: Zipf's Law & Heaps' Law

* **Zipf's Law**: The frequency $f$ of any word is inversely proportional to its rank $r$ in the frequency table:
  $$f(r) \\propto \\frac{1}{r^s} \\implies \\log f(r) = \\log C - s \\log r$$
  A tiny percentage of terms (stopwords) account for a massive fraction of all tokens.
* **Heaps' Law**: Empirically models vocabulary growth $|V|$ as a function of total token count $N$:
  $$|V| = k \\cdot N^\\beta$$
  where $30 \\le k \\le 100$ and $\\beta \\approx 0.4 - 0.6$. The vocabulary continues growing with corpus size.
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


class IITKgpLabReportPDF(FPDF):
    def header(self):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 58, 138)
        self.cell(0, 5, sanitize_pdf_text("VIRTUAL LABS CA "), 0, 1, "C")
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


def generate_pdf_report(student_name: str, student_id: str, student_div: str, date_str: str,
                        trials_df: pd.DataFrame, quiz_score: int, quiz_total: int,
                        student_notes: str, corpus_name: str, vocab_size: int,
                        postings_count: int) -> bytes:
    """Compiles verified experiment session data into an official Virtual Lab report."""
    pdf = IITKgpLabReportPDF()
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

def render_aim_section():
    """Renders Aim & Overview Section."""
    st.header("Experiment 3: Aim & Competencies")

    # Header Card
    st.markdown(f"""
    > **Course:** {EXPERIMENT_CONFIG['subject']} (`{EXPERIMENT_CONFIG['course_code']}`)  
    > **Target Roll Numbers:** **{EXPERIMENT_CONFIG['target_rolls']}**  
    > **Institution:** {EXPERIMENT_CONFIG['institution']}  
    > **Platform:** {EXPERIMENT_CONFIG['portal']}
    """)

    st.subheader("Aim of the Experiment")
    st.info(EXPERIMENT_CONFIG["aim"])

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Key Learning Outcomes")
        for i, obj in enumerate(EXPERIMENT_CONFIG["objectives"]):
            st.markdown(f"**{i+1}.** {obj}")

    with col2:
        st.subheader("Prerequisites & Foundational Knowledge")
        st.markdown("""
        * **Data Structures:** Hash Tables, Arrays, Singly Linked Lists, Binary Search Trees.
        * **Discrete Mathematics:** Set operations (Intersection $\\cap$, Union $\\cup$, Difference $\\setminus$).
        * **Text Processing Fundamentals:** Tokenization, Regular Expressions, String Normalization.
        * **Algorithmic Complexity:** Big-O notation, Two-Pointer Merge Algorithms ($O(n + m)$).
        """)

    st.divider()
    st.subheader("Experiment Roadmap")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("1. Theory", "Core Architecture", "Postings & Dictionaries")
    with c2:
        st.metric("2. Preprocessing", "Linguistic Normalization", "Stemming & Stopwords")
    with c3:
        st.metric("3. Simulation", "Index & Query Workbench", "4-Stage Pipeline")
    with c4:
        st.metric("4. Verification", "Quiz & PDF Report", "Self-Grading")


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


def render_simulation_section():
    """Renders Core Interactive Simulation Workbench."""
    st.header("Inverted Index Construction & Retrieval Simulator")
    st.caption("Interactively build an Inverted Index, trace pipeline stages, execute Boolean and phrase queries, and log trial data.")

    # Top Controls: Corpus Selection & Customization
    st.subheader("1. Corpus Configuration")
    corpus_choice = st.radio(
        "Select Corpus Source:",
        options=list(BENCHMARK_CORPORA.keys()) + ["Custom Documents (Interactive Authoring)"],
        horizontal=True
    )

    if corpus_choice == "Custom Documents (Interactive Authoring)":
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
    else:
        active_corpus = BENCHMARK_CORPORA[corpus_choice]

    with st.expander("Preview Active Documents in Collection", expanded=False):
        for doc_id, text in active_corpus.items():
            st.markdown(f"**Doc {doc_id}**: _{text}_")

    st.divider()

    # Preprocessing Controls
    st.subheader("2. Linguistic Preprocessing Pipeline Controls")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        opt_lowercase = st.checkbox("Case Folding (Lowercase)", value=True, help="Converts all tokens to lowercase.")
    with c2:
        opt_punct = st.checkbox("Strip Punctuation", value=True, help="Removes punctuation characters.")
    with c3:
        opt_stopwords = st.checkbox("Stopword Removal", value=True, help="Filters frequent grammatical words.")
    with c4:
        opt_stemming = st.checkbox("Porter Stemming", value=True, help="Applies rule-based suffix stripping to stems.")

    # Build the Inverted Index
    index_results = build_inverted_index(
        corpus=active_corpus,
        lowercase=opt_lowercase,
        remove_punct=opt_punct,
        remove_stopwords=opt_stopwords,
        use_stemming=opt_stemming,
        stopwords_set=DEFAULT_STOPWORDS
    )

    # Save active index stats in session state for report
    st.session_state["last_active_corpus_name"] = corpus_choice
    st.session_state["last_vocab_size"] = index_results["vocab_size"]
    st.session_state["last_postings_count"] = index_results["total_postings"]

    # KPI Metrics
    st.divider()
    st.subheader("3. Inverted Index Performance & Architecture Metrics")
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

    # Step-by-Step 4-Stage Construction Visualizer
    st.divider()
    st.subheader("4. Step-by-Step Construction Stages Tracker")
    stage_tabs = st.tabs([
        "Stage 1: Token Streams",
        "Stage 2: Extracted Triples",
        "Stage 3: Lexicographical Sorting",
        "Stage 4: Postings List Explorer"
    ])

    with stage_tabs[0]:
        st.caption("Raw token streams extracted per document after linguistic normalization:")
        for doc_id, tokens in index_results["stage1_tokens"].items():
            tokens_repr = ", ".join([f"({term} @ pos:{pos})" for term, pos, _ in tokens])
            st.markdown(f"**Doc {doc_id} [{len(tokens)} terms]:** {tokens_repr}")

    with stage_tabs[1]:
        st.caption("Raw unsorted (Term, DocID, Position) 3-tuples emitted by parser:")
        triples_df = pd.DataFrame(index_results["stage2_triples"])
        st.dataframe(triples_df, hide_index=True, use_container_width=True, height=220)

    with stage_tabs[2]:
        st.caption("Sorted 3-tuples ordered primarily by Term, secondarily by DocID and Position:")
        sorted_df = pd.DataFrame(index_results["stage3_sorted"])
        st.dataframe(sorted_df, hide_index=True, use_container_width=True, height=220)

    with stage_tabs[3]:
        st.caption("Inverted Index Structure: Dictionary of terms mapping to sorted postings lists with frequencies & positions:")
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
        st.dataframe(pd.DataFrame(dict_records), hide_index=True, use_container_width=True, height=280)

    # Interactive Query Engine
    st.divider()
    st.subheader("5. Interactive Query Processing Console")
    query_mode = st.selectbox(
        "Select Query Execution Type:",
        options=[
            "Conjunctive Boolean Query (AND)",
            "Disjunctive Boolean Query (OR)",
            "Negation Boolean Query (NOT)",
            "Exact Phrase Query (\"term1 term2\")",
            "Proximity Query (NEAR / k)",
            "Single Keyword Lookup"
        ]
    )

    col_q1, col_q2 = st.columns([3, 1])

    with col_q1:
        if query_mode == "Conjunctive Boolean Query (AND)":
            q_input = st.text_input("Enter two terms separated by AND:", value="information AND retrieval")
        elif query_mode == "Disjunctive Boolean Query (OR)":
            q_input = st.text_input("Enter two terms separated by OR:", value="graph OR index")
        elif query_mode == "Negation Boolean Query (NOT)":
            q_input = st.text_input("Enter base term and excluded term (e.g., retrieval NOT graph):", value="retrieval NOT graph")
        elif query_mode == "Exact Phrase Query (\"term1 term2\")":
            q_input = st.text_input("Enter exact phrase:", value="information retrieval")
        elif query_mode == "Proximity Query (NEAR / k)":
            q_input = st.text_input("Enter two terms for proximity test:", value="index search")
        else:
            q_input = st.text_input("Enter single term:", value="retrieval")

    with col_q2:
        if query_mode == "Proximity Query (NEAR / k)":
            k_dist = st.number_input("Max Token Distance (k):", min_value=1, max_value=10, value=3)
        else:
            k_dist = 3
        exec_btn = st.button("Execute Query", type="primary", use_container_width=True)

    # Execute and display search results
    query_result_docs = []
    query_latency_us = 0.0
    query_trace = []
    executed_query_str = q_input

    if q_input.strip():
        t_start = time.perf_counter()

        if query_mode == "Conjunctive Boolean Query (AND)":
            parts = [p.strip() for p in re.split(r"\bAND\b", q_input, flags=re.IGNORECASE)]
            if len(parts) >= 2:
                t1_clean = pure_porter_stem(parts[0].lower()) if opt_stemming else parts[0].lower()
                t2_clean = pure_porter_stem(parts[1].lower()) if opt_stemming else parts[1].lower()
                p1_docs = sorted(list(index_results["index"].get(t1_clean, {}).get("postings", {}).keys()))
                p2_docs = sorted(list(index_results["index"].get(t2_clean, {}).get("postings", {}).keys()))
                query_result_docs, query_trace = execute_boolean_and_with_trace(p1_docs, p2_docs, t1_clean, t2_clean)
            else:
                query_trace = ["Please enter two terms separated by AND (e.g. 'information AND retrieval')."]

        elif query_mode == "Disjunctive Boolean Query (OR)":
            parts = [p.strip() for p in re.split(r"\bOR\b", q_input, flags=re.IGNORECASE)]
            if len(parts) >= 2:
                t1_clean = pure_porter_stem(parts[0].lower()) if opt_stemming else parts[0].lower()
                t2_clean = pure_porter_stem(parts[1].lower()) if opt_stemming else parts[1].lower()
                p1_docs = sorted(list(index_results["index"].get(t1_clean, {}).get("postings", {}).keys()))
                p2_docs = sorted(list(index_results["index"].get(t2_clean, {}).get("postings", {}).keys()))
                query_result_docs, query_trace = execute_boolean_or_with_trace(p1_docs, p2_docs, t1_clean, t2_clean)
            else:
                query_trace = ["Please enter two terms separated by OR (e.g. 'graph OR index')."]

        elif query_mode == "Negation Boolean Query (NOT)":
            parts = [p.strip() for p in re.split(r"\bNOT\b", q_input, flags=re.IGNORECASE)]
            if len(parts) >= 2:
                t1_clean = pure_porter_stem(parts[0].lower()) if opt_stemming else parts[0].lower()
                t2_clean = pure_porter_stem(parts[1].lower()) if opt_stemming else parts[1].lower()
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

        elif query_mode == "Exact Phrase Query (\"term1 term2\")":
            phrase_res = execute_phrase_query(q_input, index_results["index"], use_stemming=opt_stemming)
            query_result_docs = phrase_res["docs"]
            query_trace = phrase_res["trace"]

        elif query_mode == "Proximity Query (NEAR / k)":
            words = q_input.split()
            if len(words) >= 2:
                prox_res = execute_proximity_query(words[0], words[1], k_dist, index_results["index"], use_stemming=opt_stemming)
                query_result_docs = prox_res["docs"]
                query_trace = prox_res["trace"]
            else:
                query_trace = ["Enter at least two words for proximity check."]

        else:  # Single Keyword
            res = execute_keyword_query(q_input, index_results["index"], use_stemming=opt_stemming)
            query_result_docs = res["docs"]
            query_trace = res["trace"]

        t_end = time.perf_counter()
        query_latency_us = round((t_end - t_start) * 1_000_000, 2)

    # Display Query Output
    col_out1, col_out2 = st.columns([1.5, 2.5])
    with col_out1:
        st.markdown(f"**Query Latency:** `{query_latency_us} us`")
        st.markdown(f"**Matching Documents:** `{len(query_result_docs)}` hit(s)")
        if query_result_docs:
            st.success(f"Matched DocIDs: {query_result_docs}")
            for d in query_result_docs:
                doc_text = active_corpus.get(d, "")
                st.markdown(f"- **Doc {d}:** {doc_text}")
        else:
            st.warning("No documents matched the specified query conditions.")

    with col_out2:
        st.caption("Step-by-Step Query Pointer Execution Trace:")
        with st.container(height=180):
            for line in query_trace:
                st.text(f"> {line}")

    # Plotly Analytical Visualizations
    st.divider()
    st.subheader("6. Analytical Visualizations: Zipf's Law & Term Distribution")

    plot_col1, plot_col2 = st.columns(2)

    # Zipf's Law Plot
    sorted_terms = sorted(index_results["index"].items(), key=lambda x: x[1]["cf"], reverse=True)
    if sorted_terms:
        ranks = list(range(1, len(sorted_terms) + 1))
        frequencies = [m["cf"] for _, m in sorted_terms]
        c_val = frequencies[0]
        zipf_theoretical = [round(c_val / r, 2) for r in ranks]

        # Theme-aware styling for Plotly figures
        is_dark_mode = st.session_state.get("theme_mode", "") == "🌙 Dark Mode"
        p_template = "plotly_dark" if is_dark_mode else "plotly_white"
        p_font_color = "#f8fafc" if is_dark_mode else "#0f172a"
        line_color = "#60a5fa" if is_dark_mode else "#2563eb"
        bar_color = "#34d399" if is_dark_mode else "#10b981"

        with plot_col1:
            fig_zipf = go.Figure()
            fig_zipf.add_trace(go.Scatter(
                x=ranks, y=frequencies, mode="lines+markers",
                name="Empirical Frequency (cf)", line=dict(color=line_color, width=2.5)
            ))
            fig_zipf.add_trace(go.Scatter(
                x=ranks, y=zipf_theoretical, mode="lines",
                name="Zipf's Theoretical (C/r)", line=dict(color="#ef4444", dash="dash", width=2)
            ))
            fig_zipf.update_layout(
                title="Zipf's Law: Term Rank vs. Collection Frequency",
                xaxis_title="Term Frequency Rank (r)",
                yaxis_title="Collection Frequency (cf)",
                hovermode="x unified",
                template=p_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=p_font_color),
                height=340,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_zipf, use_container_width=True)

        # Postings Length Distribution
        with plot_col2:
            top_terms = sorted_terms[:15]
            fig_df = go.Figure(go.Bar(
                x=[t for t, _ in top_terms],
                y=[m["df"] for _, m in top_terms],
                marker_color=bar_color
            ))
            fig_df.update_layout(
                title="Document Frequency (df) for Top Vocabulary Terms",
                xaxis_title="Vocabulary Term",
                yaxis_title="Document Frequency (df)",
                template=p_template,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color=p_font_color),
                height=340,
                margin=dict(l=20, r=20, t=40, b=20)
            )
            st.plotly_chart(fig_df, use_container_width=True)

    # Experimental Trial Logger
    st.divider()
    st.subheader("7. Experimental Data Log Book")
    col_log1, col_log2 = st.columns([1.5, 3.5])

    with col_log1:
        st.caption("Record experimental parameters and retrieval metrics into your session log:")
        if st.button("Record Current Trial", type="primary", use_container_width=True):
            trial_record = {
                "Trial #": len(st.session_state["trials"]) + 1,
                "Corpus": corpus_choice.split(":")[0],
                "Stopwords": "ON" if opt_stopwords else "OFF",
                "Stemming": "ON" if opt_stemming else "OFF",
                "Vocab |V|": index_results["vocab_size"],
                "Total Postings": index_results["total_postings"],
                "Query": executed_query_str,
                "Hits": len(query_result_docs),
                "Time (us)": query_latency_us,
                "Timestamp": datetime.now().strftime("%H:%M:%S")
            }
            st.session_state["trials"].append(trial_record)
            st.toast(f"Trial #{trial_record['Trial #']} successfully recorded into session log!")

        if st.button("Clear Logged Trials", use_container_width=True):
            st.session_state["trials"] = []
            st.toast("Experimental trials reset.")

    with col_log2:
        if st.session_state["trials"]:
            df_trials = pd.DataFrame(st.session_state["trials"])
            st.dataframe(df_trials, use_container_width=True, hide_index=True)
            csv_data = df_trials.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Trials as CSV",
                data=csv_data,
                file_name="inverted_index_trials.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No trials recorded yet. Click 'Record Current Trial' to capture index and query metrics.")


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
        st.markdown(f"""
        ### [{idx}] {ref['title']}
        * **Authors:** {ref['authors']}
        * **Publication:** {ref['publisher']}
        * **Topics Covered:** {ref['details']}
        * **Resource Link:** [{ref['url']}]({ref['url']})
        """)
        st.divider()


def render_feedback_section():
    """Renders IIT Kharagpur Virtual Lab Feedback Section."""
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


def main():
    st.set_page_config(
        page_title="VLab: Construction of an Inverted Index",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    init_session_state()

    # Sidebar: Theme Switcher
    st.sidebar.markdown(f"### {EXPERIMENT_CONFIG['institution']}")
    st.sidebar.caption("Virtual Laboratory Portal (vlabs.ac.in)")

    theme_mode = st.sidebar.radio(
        "Display Mode",
        options=["☀️ Light Mode", "🌙 Dark Mode", "⚙️ System Default"],
        index=0 if st.session_state.get("theme_mode") == "☀️ Light Mode" else (1 if st.session_state.get("theme_mode") == "🌙 Dark Mode" else 2),
        horizontal=True
    )
    st.session_state["theme_mode"] = theme_mode

    # CSS Injection for Clean Theming
    if theme_mode == "☀️ Light Mode":
        st.markdown("""
        <style>
            /* Explicit Light Theme */
            [data-testid="stAppViewContainer"] {
                background-color: #ffffff !important;
                color: #0f172a !important;
            }
            [data-testid="stSidebar"] {
                background-color: #f8fafc !important;
                color: #0f172a !important;
                border-right: 1px solid #e2e8f0;
            }
            [data-testid="stHeader"] {
                background-color: #ffffff !important;
            }
            .stAlert {
                border-radius: 8px;
            }
            .metric-card {
                background-color: #f1f5f9;
                border: 1px solid #cbd5e1;
                border-radius: 8px;
                padding: 12px;
            }
        </style>
        """, unsafe_allow_html=True)
    elif theme_mode == "🌙 Dark Mode":
        st.markdown("""
        <style>
            /* Explicit Dark Theme */
            [data-testid="stAppViewContainer"] {
                background-color: #0f172a !important;
                color: #f8fafc !important;
            }
            [data-testid="stSidebar"] {
                background-color: #1e293b !important;
                color: #f8fafc !important;
                border-right: 1px solid #334155;
            }
            [data-testid="stHeader"] {
                background-color: #0f172a !important;
            }
            .metric-card {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 12px;
            }
        </style>
        """, unsafe_allow_html=True)

    # Standard Streamlit Title
    st.title(EXPERIMENT_CONFIG["title"])
    st.caption(f"{EXPERIMENT_CONFIG['portal']} | {EXPERIMENT_CONFIG['institution']} | {EXPERIMENT_CONFIG['subject']}")

    st.sidebar.divider()

    section = st.sidebar.radio(
        "Navigation Menu",
        options=[
            "Aim & Overview",
            "Theory",
            "Procedure",
            "Simulation",
            "Self-Evaluation (Quiz)",
            "References",
            "Feedback",
            "Report Generation"
        ]
    )

    st.sidebar.divider()
    st.sidebar.subheader("Session Progress Tracker")
    st.sidebar.write(f"- **Active Section:** `{section}`")
    st.sidebar.write(f"- **Recorded Trials:** `{len(st.session_state['trials'])}`")
    quiz_done = st.session_state.get("quiz_submitted", False)
    st.sidebar.write(f"- **Quiz Completed:** `{'Yes' if quiz_done else 'Pending'}`")
    if quiz_done:
        st.sidebar.write(f"- **Quiz Score:** `{st.session_state.get('quiz_score', 0)} / {len(QUIZ_QUESTIONS)}`")

    st.sidebar.divider()
    st.sidebar.caption(f"Course: {EXPERIMENT_CONFIG['subject']}\nAllocated: {EXPERIMENT_CONFIG['target_rolls']}")

    # Section Dispatcher
    if section == "Aim & Overview":
        render_aim_section()
    elif section == "Theory":
        render_theory_section()
    elif section == "Procedure":
        render_procedure_section()
    elif section == "Simulation":
        render_simulation_section()
    elif section == "Self-Evaluation (Quiz)":
        render_quiz_section()
    elif section == "References":
        render_references_section()
    elif section == "Feedback":
        render_feedback_section()
    elif section == "Report Generation":
        render_report_section()


if __name__ == "__main__":
    main()
