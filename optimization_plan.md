# Optimization Plan: Combined Approach for Relationship Creation

This plan outlines the steps to optimize the creation of parent links and cross-references in `run_processing.py`, specifically addressing the inefficient use of `db.find_nodes_by_path` for cross-references.

**Goal:** Reduce database calls by resolving exact path matches in memory and batching fuzzy path lookups.

**Steps:**

## 1. Pre-DB Processing (After Phase 3 Enrichment)

*   **Action:** Build an in-memory dictionary `path_to_original_id_map` mapping exact chunk path strings to their original chunk IDs.
    ```python
    path_to_original_id = {}
    for chunk in enriched_chunks:
        path_str = chunk.get("metadata", {}).get("path", "") # Ensure 'path' is the correct key for the full path
        original_id = chunk.get("id")
        if path_str and original_id:
            path_to_original_id[path_str] = original_id
    ```
*   **Action:** Initialize two new data structures:
    *   `exact_resolved_references`: A list to store tuples `(source_original_id, target_original_id)` for references resolved via exact path matching.
    *   `paths_needing_fuzzy_lookup`: A `set` to store unique `related_path` strings that did *not* find an exact match.
*   **Action:** Iterate through all `enriched_chunks`. For each `source_chunk`:
    *   Get its `source_original_id`.
    *   For each `related_path` in its `metadata['related_sections']`:
        *   Attempt to find `target_original_id = path_to_original_id_map.get(related_path)`.
        *   If found (exact match): Add `(source_original_id, target_original_id)` to `exact_resolved_references`.
        *   If not found: Add the `related_path` string to the `paths_needing_fuzzy_lookup` set.

## 2. Batch Fuzzy Lookups (DB - Before Node Insertion)

*   **Action:** Initialize an empty dictionary `fuzzy_path_to_nodes_map`.
*   **Action:** Iterate through the unique paths in the `paths_needing_fuzzy_lookup` set.
*   **Action:** For each `path_pattern`, call `db.find_nodes_by_path(path_pattern=f"%{path_pattern}%", ...)` *once*. Use appropriate `max_results`.
*   **Action:** Store the list of node dictionaries returned by the database in `fuzzy_path_to_nodes_map[path_pattern]`. Handle potential errors during lookup.

## 3. Phase 4 (Database Operations in `run_processing.py`)

*   **Action:** Generate embeddings, prepare nodes for insertion.
*   **Action:** Clear existing nodes (`delete_nodes_by_document_id`).
*   **Action:** Insert nodes one by one, building the `original_id_to_db_id_map`.
*   **Action:** **Create Parent Links:** (No change) Iterate through chunks, use `original_id_to_db_id_map` to resolve parent `db_id`s, call `db.update_node_parent`.
*   **Action:** **Create References (Combined Pass):**
    *   Initialize `inserted_reference_pairs = set()` to prevent duplicate insertions.
    *   **Process Exact Matches:**
        *   Iterate through `exact_resolved_references`.
        *   Map `source_original_id` and `target_original_id` to their respective `db_id`s using `original_id_to_db_id_map`.
        *   If both `db_id`s are found:
            *   Call `db.insert_reference` with the `source_db_id` and `target_db_id`.
            *   Add the `(source_db_id, target_db_id)` pair to `inserted_reference_pairs`. Handle insertion errors.
    *   **Process Fuzzy Matches:**
        *   Iterate through the `original_id_to_chunk_map`. For each `source_chunk`:
            *   Get its `source_db_id` using `original_id_to_db_id_map`.
            *   For each `related_path` in its `metadata['related_sections']`:
                *   Check if this `related_path` is a key in `fuzzy_path_to_nodes_map`.
                *   If yes, retrieve the list of potential `target_nodes` from `fuzzy_path_to_nodes_map[related_path]`.
                *   For each `target_node` in that list:
                    *   Get its `target_db_id`.
                    *   If `target_db_id` is valid, different from `source_db_id`, and the pair `(source_db_id, target_db_id)` is *not* already in `inserted_reference_pairs`:
                        *   Call `db.insert_reference`.
                        *   Add the `(source_db_id, target_db_id)` pair to `inserted_reference_pairs`. Handle insertion errors.

**Benefits:**

*   Resolves exact path references efficiently in memory.
*   Batches database lookups for fuzzy path matching, significantly reducing DB calls compared to the original approach.
*   Prevents duplicate reference insertions.