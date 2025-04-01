# Phase 3 Implementation Plan: Metadata Enrichment

This plan outlines the steps to implement Phase 3 (Metadata Enrichment) based on the specifications in `docs/Phase3.md`.

**Steps:**

1.  **Create `metadata_enricher.py`:**
    *   Create a new file named `metadata_enricher.py` within the `archon/llms-txt/` directory.
    *   This file will contain the `MetadataEnricher` class.

2.  **Implement `MetadataEnricher` Class:**
    *   Copy the Python code for the `MetadataEnricher` class (lines 36-275) from `docs/Phase3.md` into the newly created `archon/llms-txt/metadata_enricher.py` file.

3.  **Update `process_docs.py`:**
    *   Modify the existing `archon/llms-txt/process_docs.py` script.
    *   **Import:** Add `from .metadata_enricher import MetadataEnricher` at the top of the file.
    *   **Instantiate:** Inside the `process_document` function, instantiate the enricher: `enricher = MetadataEnricher()`.
    *   **Integrate:** Call the enrichment process after chunking: `enriched_chunks = enricher.process_chunks(cross_ref_chunks, doc_tree)`.
    *   **Update Return:** Modify the return statement of `process_document` to return the `enriched_chunks` and `doc_tree`: `return enriched_chunks, doc_tree`.

**Visual Plan:**

```mermaid
graph TD
    A[Start Phase 3 Implementation] --> B(Create archon/llms-txt/metadata_enricher.py);
    B --> C{Implement MetadataEnricher Class};
    C -- Copy code from docs/Phase3.md --> B;
    A --> D(Modify archon/llms-txt/process_docs.py);
    D --> E[Import MetadataEnricher];
    D --> F[Instantiate MetadataEnricher in process_document];
    D --> G[Call enricher.process_chunks];
    D --> H[Update return value of process_document];
    C & H --> I[Phase 3 Logic Implemented];