# Phase 2 Implementation Plan

**Goal:** Integrate the context-aware hierarchical chunking logic defined in `docs/Phase2.md` into the existing document processing pipeline (`archon/llms-txt/process_docs.py`).

**Steps:**

1.  **Create Chunker File:**
    *   Create a new file named `archon/llms-txt/chunker.py`.

2.  **Implement Chunking Logic in `chunker.py`:**
    *   Add necessary imports: `import re`.
    *   Define the `HierarchicalChunker` class based on the code provided in `docs/Phase2.md` (lines 12-119).
    *   Implement `add_hierarchical_context` (lines 135-163) as a method within the `HierarchicalChunker` class (adjusting signature to include `self`).
    *   Implement `establish_cross_references` (lines 178-233) as a method within the `HierarchicalChunker` class (adjusting signature to include `self`).
    *   Implement the `process_chunks(document_tree)` function (lines 248-261) in `chunker.py`. This function will instantiate `HierarchicalChunker` and orchestrate the calls to `create_chunks`, `add_hierarchical_context`, and `establish_cross_references`.
    *   Add basic docstrings to the class and methods for clarity.

3.  **Integrate into Processing Pipeline (`archon/llms-txt/process_docs.py`):**
    *   **Modify `process_markdown_document` function:**
        *   Add the import `from .chunker import process_chunks` near the other imports (e.g., around line 13).
        *   **After line 60** (`classified_tree = processor.apply_classification(doc_tree)`), insert the line:
            ```python
            print("Step 4: Generating hierarchical chunks...") # Add this line for feedback
            chunks = process_chunks(classified_tree)
            ```
        *   Modify the return statement (**line 63**) to return the chunks:
            ```python
            return chunks
            ```
    *   **Modify `if __name__ == "__main__":` block:**
        *   Rename the variable receiving the result on **line 101** for clarity:
            ```python
            processed_chunks = process_markdown_document(input_path, processor)
            ```
        *   Update the check on **line 103** to use the new variable name:
            ```python
            if processed_chunks:
            ```
        *   Update the JSON dumping on **line 106** to use the new variable name:
            ```python
            json_output = json.dumps(processed_chunks, indent=2)
            ```
        *   Update the print statement on **line 118** for clarity:
            ```python
            print(f"Skipping output for {filename} due to processing errors or empty chunks.")
            ```

4.  **Dependencies:**
    *   Ensure `markdown-it-py` is listed in `requirements.txt`. No new external dependencies are needed for the chunker.

**Workflow Diagram:**

```mermaid
graph LR
    A[Input Markdown File] --> B(process_docs.py);
    subgraph process_docs.py
        B -- uses --> C(MarkdownProcessor);
        C --> D[Classified Document Tree];
        B -- imports & uses --> E(chunker.py: process_chunks);
        D -- passed to --> E;
        E --> F[Final Cross-Referenced Chunks];
        B -- processes --> F;
        F --> G[Output (JSON file)];
    end

    subgraph chunker.py
        E -- instantiates --> H(HierarchicalChunker);
        H -- calls --> I(create_chunks);
        H -- calls --> J(add_hierarchical_context);
        H -- calls --> K(establish_cross_references);
        I --> J --> K --> F;
    end