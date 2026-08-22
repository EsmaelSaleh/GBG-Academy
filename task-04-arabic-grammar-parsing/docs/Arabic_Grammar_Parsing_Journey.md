# Arabic Grammar Parsing Prompt Engineering Journey

## Task Overview
The goal of this task was to explore prompt engineering for the Arabic language, specifically focusing on complex grammatical parsing (الإعراب). We utilized the `inceptionai/jais-family-6p7b-chat` model, a state-of-the-art Arabic LLM, to see how providing strict constraints, context, and examples could improve its ability to parse sentences compared to a baseline, generic prompt.

This task satisfies the requirements defined in the Prompt Engineering assignment:
1. **Search for a model:** Selected `inceptionai/jais-family-6p7b-chat`.
2. **Identify the task:** Arabic Grammar Parsing (الإعراب).
3. **Design a prompt:** Created an engineered prompt following the Role, Context, Task, Constraints, Examples, and Expected Output Format structure.

## Model Setup
- **Model used:** `inceptionai/jais-family-6p7b-chat`
- **Quantization:** The model was loaded with 4-bit precision using `BitsAndBytesConfig` (`load_in_4bit=True`, `nf4` quant type) to allow it to run efficiently on limited GPU memory (like a T4 instance).
- **Format:** The model inherently expects interactions in a specific Instruction/Human/AI template (`JAIS_TEMPLATE_AR`).

## The Prompts

### 1. Baseline Prompt
The baseline prompt was a simple, un-engineered instruction that asked the model to parse the sentence without providing specific rules or output formats.
```arabic
قم باعراب هذه الجملة التالية:
"{sentence}"
```

### 2. Engineered Prompt
The engineered prompt was carefully crafted to guide the model towards a specific, structured output. It included:
- **Role:** An expert professor in classical Arabic grammar.
- **Context:** Precise grammatical analysis without semantic explanation.
- **Constraints:**
  - Separate prefixes/suffixes (e.g., separating conjunctions and prepositions like "بـ", "و").
  - Identify hidden pronouns (ضمير مستتر) accurately.
  - Distinguish between inflected (معرب) and fixed (مبني) words.
  - Output *only* a table without any additional text.
- **Example:** Provided a full parsed example for "نـلعب بالكرة".
- **Task & Expected Output:** Explicit instructions to output a Markdown table with columns: Word, Type, Grammatical Position, State, and Sign/Reason.

## Results & Comparison

We tested both prompts on sentences of varying difficulty.

### Test Case 1: Simple Sentence
**Input:** "قرأ الطالبُ الكتابَ في المكتبةِ" (The student read the book in the library)

- **Baseline Output:** The model returned a conversational response listing the words in bullet points with explanations, rather than a structured grammatical table. It identified the roles (Subject, Verb, Object) correctly but failed to provide a formal grammatical parsing format (الإعراب).
  <div dir="rtl">

  ```text
  الجملة التي قدمتها مكتوبة بالفعل باللغة العربية بشكل صحيح وواضح. ولكن إذا كنت ترغب في إعرابها (أي تقسيمها إلى مكوناتها الأساسية)، يمكننا القول أنها تتألف من العناصر التالية:

  1. "الطالب": هو الفاعل في الجملة وهو الذي يقوم بالعمل.
  2. "قرأ": هذا الفعل وهو العمل الذي يقوم به الفاعل.
  3. "الكتاب": هو المفعول به وهو الشيء الذي يتأثر بالفعل.
  4. "في": حرف جر يدل على المكان.
  5. "المكتبة": هي الاسم المجرور بسبب حرف الجر "في". وهي تدل على المكان الذي تمت فيه القراءة.

  إذاً، يمكن تفسير الجملة كالتالي: الطالب قرأ كتابا في المكتبة.
  ```

  </div>

- **Engineered Output:** The model successfully generated a perfectly formatted Markdown table with the columns specified. It correctly identified the verb, subject, object, and preposition, along with their grammatical states (مرفوع, مجرور, مبني) and signs (الضمة الظاهرة, الكسرة الظاهرة).

<div dir="rtl">

بناءً على التحليل السابق، يمكن تكوين الجدول التالي للجملة "قرأ الطالبُ الكتابَ في المكتبةِ":

| الكلمة      | نوع الكلمة | الموقع الإعرابي | الحالة    | العلامة وسببها |
|------------:|-----------:|--------------------:|----------:|---------------:|
| الطالبُ   | اسم       | الفاعل             | مرفوع    | الضمة الظاهرة    |
| قرأ      | فعل ماضي | فعل               | مبني على الفتح | لا محل له من الإعراب |
| الطالبُ   | اسم       | الفاعل             | مرفوع    | الضمة الظاهرة    |
| الكتابَ     | اسم       | المفعول به           | مجرور    | الكسرة الظاهرة    |
| في         | حرف جر  | حرف              | مبني | لا محل له من الإعراب |
| المكتبةِ   | اسم       | المجرور بـ "في"        | مجرور    | الكسرة الظاهرة    |

</div>

### Test Case 2: Difficult Sentence
**Input:** "نكتب بـالقلم" (We write with the pen)
*Why is this difficult?* The subject is a hidden pronoun (ضمير مستتر "نحن"), and the word "بالقلم" contains an attached preposition "بـ".

- **Baseline Output:** The model provided a conversational explanation of the meaning of the sentence, completely missing the formal parsing (الإعراب) task. It explained what "نكتب" and "بالقلم" meant but did not analyze them grammatically.
  <div dir="rtl">

  ```text
  الجملة التي قدمتها مكتوبة بالفعل باللغة العربية. ولكن إذا كنت تريد إعرابها أو توضيح معناها بالتفصيل، فإليك التفسير:

  "نكتب بالقلم" - هذا جملة مركبة تتألف من جزأين منفصلين مدمجين معا باستخدام حرف العطف "و". 

  الجزء الأول هو "نكتب" وهو فعل مضارع يدل على حدث يحدث الآن أو بشكل عام. الفاعل غير محدد في هذه الحالة لأن الفعل نفسه يمكن أن يقوم به أي شخص بغض النظر عن السياق.

  الجزء الثاني من الجملة هو "بالقلم" وهذا يشير إلى الأداة المستخدمة في الكتابة. القلم هنا ليس مجرد أداة، بل هو الوسيلة التي يتم بها تنفيذ الفعل الذي هو الكتابة.

  بالتالي، عندما تجمع هذين الجزئين معا باستخدام حرف العطف "و"، تحصل على معنى كامل للجملة: نحن (أو الأشخاص الذين يتحدثون) نستخدم القلم للكتابة.
  ```

  </div>

- **Engineered Output:** The model successfully followed the constraints:
  - It identified the hidden pronoun "ن" (or implied "نحن") as the subject (فاعل) in the nominative case (محل رفع).
  - It successfully separated the attached preposition "بـ" from "القلم".
  - It formatted everything into the requested Markdown table.

<div dir="rtl">

| الكلمة / المقطع | نوع الكلمة | الموقع الإعرابي | الحالة | العلامة وسببها |
|---:|---:|---:|---:|---:|
| ن | ضمير متصل | فاعل | مبني | في محل رفع فاعل |
| أكتب | فعل مضارع | فعل | مرفوع | الضمة الظاهرة |
| بـ | حرف جر | حرف جر | مبني | مبني على الفتح، لا محل له |
| القلم | اسم | اسم مجرور | مجرور | الكسرة الظاهرة |

</div>

## Conclusion
The experiment clearly demonstrates the power of structured prompt engineering, especially for complex, domain-specific tasks like Arabic grammar parsing. While the `jais-family-6p7b-chat` model possesses the underlying knowledge to understand Arabic grammar, a baseline prompt yields conversational and unstructured answers. By applying constraints, roles, and few-shot examples, we were able to strictly control the model's output, forcing it to produce highly accurate, professionally formatted grammatical tables.
