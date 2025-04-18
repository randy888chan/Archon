import os
import json
import sys
from pathlib import Path

# Imports moved to the top level as this file is now intended to be used as a module.
# The necessary sys.path adjustments should happen in the execution script (e.g., run_processing.py).
from archon.llms_txt.chunker import process_chunks
from archon.llms_txt.markdown_processor import MarkdownProcessor
from .metadata_enricher import MetadataEnricher


# Fallback logic removed as primary imports should be robust now.


def process_markdown_document(file_path, processor):
    """
    Process a markdown document through the complete Phase 1, 2, and 3 pipeline.
    Uses an existing processor instance.
    Returns enriched chunks and the document tree.
    """
    print(f"Processing file: {file_path}", flush=True)
    try:
        # Read file
        with open(file_path, "r", encoding="utf-8") as f:
            markdown_text = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}", flush=True)
        return None, None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}", flush=True)
        return None, None

    try:
        # --- Phase 1: Document Processing ---
        print("Phase 1: Parsing document...", flush=True)
        parsed_doc = processor.parse_document(markdown_text)

        print("Phase 1: Building hierarchy tree...", flush=True)
        doc_tree = processor.build_hierarchy_tree(parsed_doc)

        print("Phase 1: Applying classification...", flush=True)
        classified_tree = processor.apply_classification(
            doc_tree
        )  # Assuming this modifies the tree in-place or returns the modified tree
        # If apply_classification modifies in-place, use doc_tree below. If it returns a new tree, use classified_tree.
        # Based on previous phases, let's assume it modifies doc_tree or returns the same object reference.

        # --- Phase 2: Chunking ---
        # Note: The original code called process_chunks which seems to encompass Phase 2.
        # We need to ensure the chunker logic (HierarchicalChunker in the plan) is represented here.
        # Assuming process_chunks from chunker.py handles Phase 2 logic (create_chunks, add_hierarchical_context, establish_cross_references)
        print("Phase 2: Generating hierarchical chunks...", flush=True)
        # TODO: Review if `process_chunks` correctly implements all of Phase 2 chunking logic from the plan.
        # For now, we assume it returns the `cross_ref_chunks` needed for Phase 3.
        cross_ref_chunks = process_chunks(
            classified_tree
        )  # Using classified_tree as input based on original code

        # --- Phase 3: Metadata Enrichment ---
        print("Phase 3: Enriching chunks with metadata...", flush=True)
        enricher = MetadataEnricher()
        # Pass the original doc_tree for context if needed by enricher methods
        enriched_chunks = enricher.process_chunks(cross_ref_chunks, doc_tree)

        print("Document processing complete (Phases 1-3).", flush=True)
        return enriched_chunks, doc_tree

    except Exception as e:
        print(f"Error processing document {file_path}: {e}", flush=True)
        import traceback

        traceback.print_exc()
        return None, None


# Removed __main__ block - execution logic moved to run_processing.py
