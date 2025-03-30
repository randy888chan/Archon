import os
import json
import sys
from pathlib import Path

# Ensure the parent directory (archon) is in the Python path
# This allows importing .markdown_processor
current_dir = Path(__file__).parent
archon_dir = current_dir.parent
sys.path.append(str(archon_dir))

try:
    from llms_txt.markdown_processor import MarkdownProcessor
except ImportError:
    print("Error importing MarkdownProcessor. Make sure archon directory is in PYTHONPATH.")
    print(f"Current sys.path: {sys.path}")
    # Fallback for direct execution if the path logic fails
    try:
        from markdown_processor import MarkdownProcessor
    except ImportError as e:
         print(f"Could not import MarkdownProcessor even with fallback: {e}")
         sys.exit(1)


def process_markdown_document(file_path, processor):
    """
    Process a markdown document through the complete Phase 1 pipeline.
    Uses an existing processor instance.
    """
    print(f"Processing file: {file_path}")
    try:
        # Read file
        with open(file_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
    except FileNotFoundError:
        print(f"Error: Input file not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

    try:
        # Step 1: Parse the markdown
        print("Step 1: Parsing document...")
        parsed_doc = processor.parse_document(markdown_text)
        # print(f"Parsed doc keys: {parsed_doc.keys()}")
        # print(f"Title: {parsed_doc.get('title')}")
        # print(f"Headers found: {len(parsed_doc.get('headers', []))}")
        # print(f"Content blocks found: {len(parsed_doc.get('content_blocks', []))}")

        # Step 2: Build hierarchical tree
        print("Step 2: Building hierarchy tree...")
        doc_tree = processor.build_hierarchy_tree(parsed_doc)
        # print(f"Tree root keys: {doc_tree.keys()}")
        # print(f"Tree children count: {len(doc_tree.get('children', []))}")


        # Step 3: Classify content throughout the tree
        print("Step 3: Applying classification...")
        classified_tree = processor.apply_classification(doc_tree)
        print("Processing complete.")

        return classified_tree
    except Exception as e:
        print(f"Error processing document {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Define base directory relative to this script
    base_dir = Path(__file__).parent
    docs_dir = base_dir.parent.parent / "docs" # Go up two levels to project root, then to docs
    output_dir = base_dir / "output"

    # List of input files relative to the 'docs' directory
    input_files = [
        "anthropic-llms.txt",
        "crewai-llms-full.txt"
    ]

    # Ensure the output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory ensured: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory {output_dir}: {e}")
        sys.exit(1)

    # Initialize the processor once
    processor = MarkdownProcessor()

    # Process each file
    for filename in input_files:
        input_path = docs_dir / filename
        output_filename = f"{Path(filename).stem}_processed.md"
        output_path = output_dir / output_filename

        print("-" * 20)
        result_tree = process_markdown_document(input_path, processor)

        if result_tree:
            try:
                # Convert the result tree to a formatted JSON string
                json_output = json.dumps(result_tree, indent=2)

                # Write the JSON output to a Markdown file inside a code block
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Processed Output for: {filename}\n\n")
                    f.write("```json\n")
                    f.write(json_output)
                    f.write("\n```\n")
                print(f"Successfully wrote processed output to: {output_path}")
            except Exception as e:
                print(f"Error writing output file {output_path}: {e}")
        else:
            print(f"Skipping output for {filename} due to processing errors.")
        print("-" * 20)

    print("\nScript finished.")