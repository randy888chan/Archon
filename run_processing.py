import os
import json
import sys
from pathlib import Path

# Import necessary components from the archon package
try:
    from archon.llms_txt.markdown_processor import MarkdownProcessor
    from archon.llms_txt.process_docs import process_markdown_document
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Ensure the script is run from the project root directory ('c:\\Users\\zcoru\\Archon')")
    print(f"Current sys.path: {sys.path}")
    # Attempt to add project root to path if running from elsewhere, though less ideal
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        print(f"Added {project_root} to sys.path. Please try running again.")
    sys.exit(1)


if __name__ == "__main__":
    # Define base directory relative to this script (project root)
    base_dir = Path(__file__).parent
    docs_dir = base_dir / "docs" # Docs directory relative to project root
    output_dir = base_dir / "archon" / "llms_txt" / "output" # Output dir relative to project root

    # List of input files relative to the 'docs' directory
    input_files = [
        "anthropic-llms.txt",
        "crewai-llms-full.txt"
        # Add other files from docs/ if needed
    ]

    # Ensure the output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory ensured: {output_dir}")
    except Exception as e:
        print(f"Error creating output directory {output_dir}: {e}")
        sys.exit(1)

    # Initialize the processor once
    print("Initializing MarkdownProcessor...")
    processor = MarkdownProcessor()
    print("Processor initialized.")

    # Process each file
    for filename in input_files:
        input_path = docs_dir / filename
        # Keep output filenames the same, but place in the correct output_dir
        output_filename = f"{Path(filename).stem}_processed.md"
        output_path = output_dir / output_filename

        print("-" * 20)
        # Call the imported function
        processed_chunks = process_markdown_document(input_path, processor)

        if processed_chunks:
            try:
                # Convert the result chunks to a formatted JSON string
                json_output = json.dumps(processed_chunks, indent=2)

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
            print(f"Skipping output for {filename} due to processing errors or empty chunks.")
        print("-" * 20)

    print("\nScript finished.")