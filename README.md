# Virtual Lab: Construction of an Inverted Index
**Course:** Knowledge Graphs and Information Retrieval Systems (KGIRS)  
**Experiment Number:** 3  
**Target Student Allocation:** Roll Numbers 11 – 15 (Divisions A, B, C)  
**Platform:** Streamlit  

---

## 📌 Experiment Overview

* **Title:** Construction of an Inverted Index
* **Brief Explanation:** Create an inverted index mapping terms to documents and their occurrences.
* **Expected Outcome:** A basic searchable index supporting efficient keyword retrieval.
* **Aim:** To construct, analyze, and benchmark an Inverted Index data structure for textual document corpora, evaluate linguistic preprocessing pipelines (tokenization, stopword removal, stemming), and execute Boolean, phrase, and proximity queries with linear-time pointer-merge algorithms.

---

## 🏛️ VESIT Virtual Lab Page Structure

The application strictly mirrors the pedagogical architecture of **VESIT Virtual Labs** (`vlabs.ac.in`):

1. **Aim & Overview:**
   * Course & institution branding
   * Formal aim and detailed learning objectives
   * Prerequisites (Data structures, Discrete math, Big-O complexity)
   * Experiment roadmap
2. **Theory:**
   * Why linear scan fails ($O(N \cdot L)$)
   * Term-Document Incidence Matrix vs. Sparsity bottleneck ($>99.8\%$ zeroes)
   * Inverted Index components: Dictionary (Lexicon) and Postings Lists
   * Conjunctive (`AND`), Disjunctive (`OR`), and Negation (`NOT`) algorithms
   * Query optimization via Document Frequency ($df$) ordering
   * Skip Pointers ($O(\sqrt{L})$)
   * Comparative Complexity Analysis table
   * Key terminology glossary
3. **Procedure:**
   * Clear, 9-step guided experimental protocol
   * System workflow diagram
4. **Simulation (The Interactive Workbench):**
   * **Corpus Selection:** Choose from 3 curated domain corpora (Information Retrieval, Knowledge Graphs/KGIRS, NLP) or interactive custom document authoring
   * **Linguistic Preprocessing Controls:** Toggles for Case Folding, Punctuation Stripping, Stopword Removal, and Porter Stemming
   * **Real-time Architecture KPIs:** Total Documents ($N$), Total Tokens, Vocabulary Size ($|V|$), Total Postings, and Index Sparsity
   * **4-Stage Construction Tracker:**
     * Stage 1: Document Token Streams
     * Stage 2: Raw $(Term, DocID, Position)$ Triples
     * Stage 3: Lexicographically Sorted Triples
     * Stage 4: Interactive Dictionary & Postings Explorer
   * **Trial Logger:** Record trial runs, view session table, and export as CSV (`inverted_index_trials.csv`)
5. **Self-Evaluation (Quiz):**
   * 10 rigorous multiple-choice questions covering incidence matrix sparsity, pointer merge complexity, query optimization order, positional postings, skip pointers, Heaps' Law, Zipf's Law, $tf$ vs. $df$, phrase verification, and SPIMI vs. BSBI
   * Instant grading with detailed reasoning for each option
   * Live score synchronization with the final report
6. **References:**
   * Standard textbooks (Manning, Raghavan, Schütze; Baeza-Yates & Ribeiro-Neto; Witten, Moffat, Bell)
   * Direct links to canonical literature
7. **Feedback:**
   * VESIT standard feedback questionnaire with 5-point Likert ratings and suggestions
8. **Report Generation:**
   * Student details entry (Name, Roll Number, Class/Division, Date)
   * Discussion & analytical observations
   * Official verified PDF generation with dynamic trial tables and evaluator sign-off
   * Direct `.pdf` and `.csv` download buttons

---

## 🚀 How to Run the Virtual Lab

### Prerequisites
Make sure dependencies are installed in your virtual environment:
```bash
source .venv/bin/activate
pip install streamlit fpdf2 plotly pandas numpy
```

### Launching the Application
You can run the application using either command from within `Virtual_lab`:

```bash
# Option 1: Run app.py
streamlit run app.py

# Option 2: Run template.py
streamlit run template.py
```

Or from the workspace root:
```bash
streamlit run Virtual_lab/app.py
```

Then open your browser at `http://localhost:8501`.

---

## 🎨 Display Themes
The application includes a built-in **Theme Switcher** in the sidebar:
* **☀️ Light Mode (Default / Academic Standard):** Clean aesthetic matching standard Virtual Labs (`vlabs.ac.in`), crisp white surfaces, high-contrast typography, and `plotly_white` responsive graphs.
* **🌙 Dark Mode:** Modern dark slate background (`#0f172a`), neon accents, and `plotly_dark` charts.
* **⚙️ System Default:** Adapts dynamically to your operating system or browser theme preference.

---

## 📋 Submission Checklist
* [x] **Allocation Match:** Experiment 3: *Construction of an Inverted Index* (Target Rolls: 11 - 15).
* [x] **Pedagogical Structure:** All 8 VESIT tabs implemented (Aim, Theory, Procedure, Simulation, Self-Evaluation, References, Feedback, Report Generation).
* [x] **Execution Compatibility:** Both `app.py` and `template.py` function identically.
* [x] **Verified PDF Report:** Downloadable PDF generation with student metadata, benchmark trials table, quiz score, and signature box.
* [x] **Zero External Model Dependencies:** Offline pure-Python linguistic normalizer and Porter stemmer.
