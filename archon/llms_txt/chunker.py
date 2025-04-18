import uuid
import re
import json  # Although not directly used in the provided snippets, it might be useful for debugging/output


class HierarchicalChunker:
    """
    Implements context-aware chunking strategies for markdown documents,
    preserving hierarchical relationships.
    """

    def __init__(self, max_chunk_size=1000, overlap_size=200):
        """
        Initialize the chunker.

        Args:
            max_chunk_size (int): The target maximum size for descriptive text chunks.
            overlap_size (int): The target overlap size (in words) for descriptive text chunks.
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size

    def create_chunks(self, document_tree):
        """
        Create context-aware chunks based on content type and hierarchy
        from the document tree.

        Args:
            document_tree (dict): The hierarchical document tree structure.

        Returns:
            list: A list of chunk dictionaries.
        """
        chunks = []
        # Start processing from the root node, assuming it might have content/title
        self._process_node(document_tree, [], chunks)
        return chunks

    def _process_node(self, node, ancestor_path, chunks):
        """
        Recursively process a node in the document tree to create appropriate chunks.

        Args:
            node (dict): The current node in the document tree.
            ancestor_path (list): The list of ancestor nodes' titles and levels.
            chunks (list): The list where created chunks are appended.
        """
        # Skip empty or invalid nodes
        if not node or not isinstance(node, dict):
            return

        # Build current path including this node if it has a title
        current_path = ancestor_path.copy()
        if node.get("title"):
            current_path.append(
                {
                    "title": node["title"],
                    "level": node.get("level", 0),  # Use level if present, default 0
                }
            )

        # Process based on content type defined in metadata
        content_type = node.get("metadata", {}).get("content_type")
        node_content = node.get("content", "")  # Get content safely

        # Only create chunks if there's actual content or if it's a structural node type
        # that should be represented even if empty (e.g., maybe an empty code block placeholder?)
        # Current logic focuses on nodes with content.

        if node_content or content_type in [
            "link_list",
            "code_example",
            "table_based",
        ]:  # Process if content exists or specific types
            if content_type == "link_list":
                # Keep link collections together under their parent header
                chunks.append(self._create_chunk(node, current_path, "link_list"))

            elif content_type == "code_example":
                # Check if content is within the node itself or in a dedicated 'code_blocks' list
                if "code_blocks" in node and isinstance(node["code_blocks"], list):
                    # Handle multiple code blocks associated with a header
                    for code_block_node in node["code_blocks"]:
                        # Use the code_block_node's content
                        chunks.append(
                            self._create_chunk(
                                code_block_node, current_path, "code_example"
                            )
                        )
                elif node_content:  # Fallback if content is directly in the node
                    # Maintain atomic code examples as single chunks
                    chunks.append(
                        self._create_chunk(node, current_path, "code_example")
                    )

            elif content_type == "descriptive_text":
                # Apply semantic chunking for descriptive content
                self._chunk_descriptive_content(node, current_path, chunks)

            elif content_type == "introduction":
                # Check if it's the root document's introduction or a section intro
                if (
                    node.get("type") == "document"
                    or node.get("metadata", {}).get("section_type") == "root"
                ):
                    # Keep root introductions as a single chunk
                    chunks.append(
                        self._create_chunk(node, current_path, "introduction")
                    )
                else:
                    # Treat other short descriptions based on length
                    self._chunk_descriptive_content(node, current_path, chunks)

            elif content_type == "table_based":
                # Keep tables together
                chunks.append(self._create_chunk(node, current_path, "table_based"))

            elif (
                node_content
            ):  # Handle nodes with content but unclassified or 'default' type
                # Default chunking: treat as descriptive text for splitting logic
                self._chunk_descriptive_content(node, current_path, chunks)
                # Or simply create a single chunk if splitting isn't desired for defaults:
                # chunks.append(self._create_chunk(node, current_path, "default"))

        # Process children recursively regardless of whether the parent node created a chunk
        for child in node.get("children", []):
            self._process_node(child, current_path, chunks)

    def _create_chunk(self, node, path, chunk_type):
        """
        Create a chunk dictionary with hierarchical context.

        Args:
            node (dict): The node from which to create the chunk.
            path (list): The hierarchical path (list of ancestor dicts).
            chunk_type (str): The determined type of this chunk.

        Returns:
            dict: The created chunk dictionary.
        """
        # Ensure metadata exists
        metadata = node.get(
            "metadata", {}
        ).copy()  # Use copy to avoid modifying original node
        metadata.setdefault("section_type", "General")  # Ensure section_type exists
        metadata["content_type"] = chunk_type  # Override with the determined chunk type

        return {
            "id": uuid.uuid4().hex,  # Add unique ID
            "content": node.get("content", ""),
            "title": node.get("title"),  # Title of the specific node/header
            "type": chunk_type,
            "hierarchy_path": path,  # List of ancestor {title, level} dicts
            "metadata": {
                "section_type": metadata.get("section_type"),
                "content_type": metadata.get(
                    "content_type"
                ),  # Use the determined chunk_type
                "header_level": node.get("level", 0),  # Header level of the node
                # Add other relevant metadata from the node if needed
                "language": (
                    node.get("language") if chunk_type == "code_example" else None
                ),
            },
        }

    def _chunk_descriptive_content(self, node, path, chunks):
        """
        Create semantic chunks for descriptive content with overlap.
        Splits based on paragraphs first, then size.

        Args:
            node (dict): The node containing the descriptive content.
            path (list): The hierarchical path.
            chunks (list): The list to append created chunks to.
        """
        content = node.get("content", "")
        if not content:
            return  # Skip empty content

        # If content is small enough, keep it as a single chunk
        if len(content) <= self.max_chunk_size:
            chunks.append(self._create_chunk(node, path, "descriptive_text"))
            return

        # Split into paragraphs first (handles multiple blank lines)
        paragraphs = re.split(r"\n\s*\n", content.strip())
        if not paragraphs:
            return  # Skip if splitting results in nothing

        current_chunk_content = ""
        start_new_chunk = True

        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue  # Skip empty paragraphs resulting from split

            # Estimate length if added
            potential_len = (
                len(current_chunk_content)
                + len(paragraph)
                + (2 if current_chunk_content else 0)
            )  # +2 for '\n\n'

            if potential_len <= self.max_chunk_size:
                # Add paragraph to current chunk
                if current_chunk_content:
                    current_chunk_content += "\n\n" + paragraph
                else:
                    current_chunk_content = paragraph
                start_new_chunk = False
            else:
                # Current chunk is full or adding paragraph would exceed limit
                if current_chunk_content:
                    # Create a chunk with the current content
                    chunk_node = node.copy()  # Create a shallow copy to modify content
                    chunk_node["content"] = current_chunk_content
                    chunks.append(
                        self._create_chunk(chunk_node, path, "descriptive_text")
                    )

                    # Start new chunk with overlap if possible
                    overlap_words = current_chunk_content.split()
                    if len(overlap_words) > self.overlap_size:
                        # Take the last N words for overlap
                        overlap_text = " ".join(overlap_words[-self.overlap_size :])
                        current_chunk_content = overlap_text + "\n\n" + paragraph
                    else:
                        # Not enough words for full overlap, just start with the new paragraph
                        current_chunk_content = paragraph
                    start_new_chunk = True  # Mark that we started a new chunk
                else:
                    # Paragraph itself is larger than max_chunk_size
                    # Option 1: Create a chunk with just this large paragraph (simplest)
                    chunk_node = node.copy()
                    chunk_node["content"] = paragraph
                    chunks.append(
                        self._create_chunk(
                            chunk_node, path, "descriptive_text_oversized"
                        )
                    )  # Mark as oversized
                    current_chunk_content = ""  # Reset for next paragraph
                    start_new_chunk = True
                    # Option 2: Force split the large paragraph (more complex, requires sentence splitting)
                    # For now, we use Option 1.

        # Add the final chunk if there's content left
        if current_chunk_content:
            chunk_node = node.copy()
            chunk_node["content"] = current_chunk_content
            chunks.append(self._create_chunk(chunk_node, path, "descriptive_text"))

    def add_hierarchical_context(self, chunks):
        """
        Add formatted hierarchical path and context to each chunk.

        Args:
            chunks (list): The list of chunk dictionaries.

        Returns:
            list: The list of enhanced chunk dictionaries.
        """
        enhanced_chunks = []

        for chunk in chunks:
            # Create a formatted header path string
            path = chunk.get("hierarchy_path", [])
            # Filter out potential None titles if path wasn't built perfectly
            formatted_path = " > ".join(
                [p.get("title", "") for p in path if p.get("title")]
            )

            # Create enhanced content with header context
            enhanced_content = f"Context: {formatted_path}\n\n"

            # Add the original title as a header if it exists and isn't already in the content
            chunk_title = chunk.get("title")
            chunk_content = chunk.get("content", "")
            header_level = chunk.get("metadata", {}).get(
                "header_level", 0
            )  # Use level from metadata

            # Basic check if title seems to be the start of the content already
            title_in_content = False
            if chunk_title:
                # Check variations like "# Title", "## Title", "Title\n---" etc.
                pattern = (
                    r"^\s*(#{{1,{level}}}\s*{title}|{title}\s*\n[-=]+)\s*\n".format(
                        level=max(1, header_level), title=re.escape(chunk_title)
                    )
                )
                if re.match(pattern, chunk_content, re.IGNORECASE):
                    title_in_content = True

            if chunk_title and not title_in_content and header_level > 0:
                # Add title with appropriate markdown header level
                header_prefix = "#" * max(1, header_level)
                enhanced_content += f"{header_prefix} {chunk_title}\n\n"

            # Add the original content
            enhanced_content += chunk_content

            # Update the chunk
            enhanced_chunk = chunk.copy()  # Work on a copy
            enhanced_chunk["enhanced_content"] = enhanced_content
            enhanced_chunk["formatted_path"] = formatted_path
            # Ensure metadata exists before adding to it
            if "metadata" not in enhanced_chunk:
                enhanced_chunk["metadata"] = {}
            enhanced_chunk["metadata"]["hierarchy_levels"] = [
                p.get("level", 0) for p in path if p.get("title")
            ]

            enhanced_chunks.append(enhanced_chunk)

        return enhanced_chunks

    def establish_cross_references(self, chunks):
        """
        Create relationships (indices) between related chunks based on titles and paths.

        Args:
            chunks (list): The list of enhanced chunk dictionaries.

        Returns:
            list: The list of chunks with added 'related_chunks' and 'related_sections' metadata.
        """
        # Create mappings for efficient lookup
        title_to_indices = {}
        path_to_indices = {}

        # First pass: build mappings
        for i, chunk in enumerate(chunks):
            title = chunk.get("title")
            formatted_path = chunk.get("formatted_path")  # Path string like "A > B > C"

            # Map titles to list of chunk indices
            if title:
                if title not in title_to_indices:
                    title_to_indices[title] = []
                title_to_indices[title].append(i)

            # Map formatted paths to list of chunk indices
            if formatted_path:
                if formatted_path not in path_to_indices:
                    path_to_indices[formatted_path] = []
                path_to_indices[formatted_path].append(i)

        # Second pass: establish references for each chunk
        for i, chunk in enumerate(chunks):
            related_indices = set()
            current_path = chunk.get("formatted_path", "")
            current_title = chunk.get("title")

            # 1. Find chunks with the exact same title but different paths (potential duplicates or related sections)
            if current_title and current_title in title_to_indices:
                for idx in title_to_indices[current_title]:
                    if idx != i:  # Don't link to self
                        # Check if paths are truly different to avoid self-linking on path variations
                        if chunks[idx].get("formatted_path") != current_path:
                            related_indices.add(idx)

            # 2. Look for siblings (same parent path)
            path_parts = current_path.split(" > ")
            if len(path_parts) > 1:
                parent_path = " > ".join(path_parts[:-1])
                # Find other chunks that share this exact parent path prefix
                for path, indices in path_to_indices.items():
                    if path != current_path and path.startswith(parent_path + " > "):
                        # Check if it's a direct sibling (only one level deeper)
                        if len(path.split(" > ")) == len(path_parts):
                            for idx in indices:
                                if idx != i:
                                    related_indices.add(idx)

            # 3. Look for parent/child relationships (less direct, maybe add later if needed)
            # Example: Link to parent chunk if found
            # parent_path = " > ".join(path_parts[:-1])
            # if parent_path in path_to_indices:
            #     for idx in path_to_indices[parent_path]:
            #         related_indices.add(idx) # Could add parent index

            # Ensure metadata exists
            if "metadata" not in chunks[i]:
                chunks[i]["metadata"] = {}

            # Set related chunks (indices) in metadata
            sorted_related_indices = sorted(list(related_indices))
            chunks[i]["metadata"]["related_chunks"] = sorted_related_indices

            # Also store the related sections' paths/titles for easier reference during retrieval
            related_sections_info = []
            for idx in sorted_related_indices:
                if idx < len(chunks):  # Ensure index is valid
                    related_chunk = chunks[idx]
                    info = {
                        "path": related_chunk.get("formatted_path", ""),
                        "title": related_chunk.get("title", ""),
                        "chunk_index": idx,
                    }
                    related_sections_info.append(info)

            chunks[i]["metadata"][
                "related_sections_info"
            ] = related_sections_info  # Store richer info

        return chunks


def process_chunks(document_tree):
    """
    Process the document tree into hierarchical, context-enhanced,
    and cross-referenced chunks.

    Args:
        document_tree (dict): The hierarchical document tree from MarkdownProcessor.

    Returns:
        list: A list of final chunk dictionaries ready for storage/embedding.
    """
    # Instantiate the chunker (consider making params configurable later)
    chunker = HierarchicalChunker(
        max_chunk_size=1000, overlap_size=150
    )  # Adjusted overlap

    # Step 1: Create basic context-aware chunks
    print("   - Creating initial chunks...", flush=True)
    basic_chunks = chunker.create_chunks(document_tree)
    print(f"   - Created {len(basic_chunks)} initial chunks.", flush=True)

    # Step 2: Add hierarchical context (formatted path, enhanced content)
    print("   - Adding hierarchical context...", flush=True)
    enhanced_chunks = chunker.add_hierarchical_context(basic_chunks)
    print("   - Context added.", flush=True)

    # Step 3: Establish cross-references between chunks
    print("   - Establishing cross-references...", flush=True)
    referenced_chunks = chunker.establish_cross_references(enhanced_chunks)
    print("   - Cross-references established.", flush=True)

    return referenced_chunks
