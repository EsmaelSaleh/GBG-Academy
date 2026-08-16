# Arabic Text Correction Journey

This document outlines the step-by-step journey, experiments, and iterations made to develop an effective Arabic spelling and grammar correction pipeline.

---

## Phase 1: Basic Spelling & LLM Grammar Testing
**File Reference:** [`Arabic_Text_Correct.ipynb`](../notebooks/Arabic_Text_Correct.ipynb)

In this initial phase, the goal was to identify and fix spelling mistakes at the word level, and then use LLMs to fix the grammar contextually.

### 1. Spelling Detection with CAMeL Tools
To detect spelling errors, we used the `camel_tools.morphology.analyzer.Analyzer` initialized with `MorphologyDB.builtin_db()`.

**How it works:**
The analyzer attempts to generate all valid morphological analyses for a given word (root, prefixes, suffixes, POS, etc.). If `analyzer.analyze(word)` returns an empty list `[]`, it means the word does not exist in the Modern Standard Arabic (MSA) database and cannot be formed by standard morphological rules. We flag these words as spelling errors.

### 2. Spelling Correction with SymSpell
To correct the detected errors, we used `symspellpy`.

**How it works:**
We loaded an Arabic frequency dictionary (`ar_50k.txt`). For every misspelled word, we called `sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)`. This returns candidate words that are 1 or 2 character edits away from the misspelled word. 
To ensure we don't suggest invalid words, we passed these candidates back through the `camel_tools` analyzer. The first candidate that produced a valid morphological analysis was chosen as the correction.

### 3. Grammar Correction Attempts with LLMs
After fixing the spelling, we attempted to pass the sentence to several models for grammar correction:
*   **`gemma-3-1b-arabic-gec-v1`**: This model struggled. It either returned the exact same input with no modifications, or had trouble dealing with spelling variations.
*   **`AraT5`** *(e.g., `SuperSl6/Arabic-Text-Correction` and `UBC-NLP/AraT5-base-finetuned-arabic-gec`)*: These models exhibited severe hallucination and looping. Even when applying strict anti-hallucination generation parameters (`repetition_penalty=2.5`, `no_repeat_ngram_size=2`), the model would get stuck repeating the same phrases or sentences. 

**Conclusion for Phase 1:** Rule-based spelling correction worked well, but standard Seq2Seq LLM generation for grammar was too unstable or ineffective out of the box.

---

## Phase 2: Hardcoded Grammatical Rules
**File Reference:** [`Arabic_Text_Corrector.ipynb`](../notebooks/Arabic_Text_Corrector.ipynb)

Since standard LLMs struggled, the next approach was to manually implement grammar rules using the rich morphological features provided by CAMeL Tools.

### Deep Morphological Analysis
By using `MLEDisambiguator.pretrained()`, we extracted contextual features for every word in the sentence. The analyzer returns a rich set of tags:
*   **`pos`**: Part of Speech (e.g., 'noun', 'adj', 'verb')
*   **`gen`**: Gender ('m' for masculine, 'f' for feminine)
*   **`num`**: Number ('s' for singular, 'd' for dual, 'p' for plural)
*   **`stt`**: State (e.g., construct state)
*   **`cas`**: Case (nominative, accusative, genitive)
*   **`lex`**: The core lemma/root of the word

### Implementation of `detect_grammar_errors`
With these features, we built a rule-based engine to check for agreement:
1.  **Noun-Adjective Agreement:** Checked if a `noun` followed by an `adj` matched in gender (`gen`) and number (`num`).
2.  **Subject-Verb Agreement:** Checked if a subject (`noun` or `pronoun`) followed by a `verb` matched in gender and number.

**Correction Mechanism:**
If a mismatch was found, we used the `camel_tools.morphology.generator.Generator`. We extracted the lemma (`lex`) of the target word (e.g., the adjective) and requested the generator to re-inflect it with the corrected features (matching the gender and number of the noun).

**Conclusion for Phase 2:** This approach worked mathematically and proved that CAMeL tools' morphological engine is powerful. However, language is complex; hardcoding every grammatical edge case (plural rules, broken plurals, distant subject-verb relations) is practically impossible and brittle.

---

## Phase 3: The Hybrid Engineering Pipeline (Best Approach)
**File Reference:** [`GBG_Task1_AR_Text_Correction_Enhanced.ipynb`](../notebooks/GBG_Task1_AR_Text_Correction_Enhanced.ipynb)

To get the best of both worlds, an engineering pipeline was designed combining the reliability of rule-based spelling correction with the contextual power of a specialized Grammatical Error Correction (GEC) Transformer.

### The Two-Step Pipeline
We tested the `CAMeL-Lab/arabart-qalb14-gec-ged-13` model. It was excellent at fixing complex grammar, but it sometimes stumbled if the input words had severe spelling errors (out-of-vocabulary non-words). 

To solve this, we built a hybrid pipeline:
1.  **Step 1: Isolated Spelling Correction (Rule-Based)**
    *   The text is tokenized.
    *   The same method from Phase 1 (`SymSpell` + `camel_tools` analyzer validation) is applied to ensure every word is a valid Arabic word.
    *   This "cleans" the text of typos without altering the grammar.
2.  **Step 2: Contextual Grammar Correction (Seq2Seq Model)**
    *   The spelling-corrected text is passed to the `arabart-qalb14-gec-ged-13` model.
    *   Because the model now receives valid Arabic tokens, it doesn't get confused by typos. It can focus entirely on syntactic relationships, fixing noun-adjective mismatches, correct prepositions, and subject-verb agreements accurately.

**Conclusion for Phase 3:** This pipeline successfully mitigates the weaknesses of both approaches. It prevents the transformer from hallucinating over misspelled words, and it avoids the need to hardcode infinite grammar rules.

---

## Sample Tests & Examples

Here are some of the key test sentences run through the pipelines to validate the corrections:

**Test 1 (Severe Spelling & Grammar):**
*   **Original:** `اكلت الغنت النفاحة وهو سغيدة`
*   **Spelling Fix:** `اكلت البنت التفاحة وهو سعيدة` (Fixed typos to valid words)
*   **Grammar Fix:** `اكلت البنت التفاحة وهي سعيدة` (Pronoun gender mismatch fixed)

**Test 2 (Complex Sentence with multiple issues):**
*   **Original:** `ذهبت الطالبة الجديد إلى المدرصة ضباحاً. هي تحبان القراءة كتيراً، ولديها كتاب جميلان تقرأه كل يوم.`
*   **Issues:**
    *   `الجديد` (Masculine adj for feminine noun `الطالبة`)
    *   `المدرصة` (Spelling typo for `المدرسة`)
    *   `ضباحاً` (Spelling typo for `صباحاً`)
    *   `تحبان` (Dual verb for singular pronoun `هي`)
    *   `كتيراً` (Dialect/spelling typo for `كثيراً`)
    *   `جميلان` (Dual adj for singular noun `كتاب`)
*   **Pipeline Corrected:** `ذهبت الطالبة الجديدة إلى المدرسة صباحاً. هي تحب القراءة كثيراً، ولديها كتاب جميل تقرأه كل يوم.`

**Test 3 (Short Contextual Check):**
*   **Original:** `اشترى الطالب قلماً جديدة من مكتزة قريبة.`
*   **Issues:** `جديدة` (Feminine adj for masculine `قلم`), `مكتزة` (Typo for `مكتبة`).
*   **Pipeline Corrected:** `اشترى الطالب قلماً جديداً من مكتبة قريبة.`
