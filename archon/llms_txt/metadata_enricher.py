# metadata_enricher.py
import re


class MetadataEnricher:
    """
    Enriches document chunks with metadata and enhances text content
    with contextual information.
    """

    def __init__(self):
        self.section_types = [
            "API documentation",
            "Concepts documentation",
            "Optional",
            "Internals",
        ]

    def enrich_chunk(self, chunk, document_tree):
        """Add comprehensive metadata to each chunk"""

        # Initialize base metadata structure
        metadata = {
            "section_type": self._determine_section_type(chunk, document_tree),
            "hierarchy_path": self._build_hierarchy_path(chunk, document_tree),
            "content_type": chunk.get("content_type", "unknown"),
            "related_sections": self._find_related_sections(chunk, document_tree),
            "document_position": self._calculate_position(chunk, document_tree),
            "header_level": chunk.get("header_level", 0),
            "contains_links": self._contains_links(chunk),
            "link_count": self._count_links(chunk),
            # Add formatted_path here
            "formatted_path": "",  # TODO: , will be calculated next
        }

        # Calculate formatted_path based on hierarchy_path
        hierarchy_path = metadata.get("hierarchy_path", [])
        if hierarchy_path:
            path_titles = [
                p.get("title", str(p)) if isinstance(p, dict) else str(p)
                for p in hierarchy_path
            ]
            metadata["formatted_path"] = " > ".join(path_titles)

        # Attach metadata to chunk
        chunk["metadata"] = metadata

        return chunk

    def _determine_section_type(self, chunk, document_tree):
        """Determine the section type based on the hierarchy path"""
        # Extract from headers or document structure
        hierarchy = chunk.get("hierarchy_path", [])
        for path_element in hierarchy:
            element_title = None
            if isinstance(path_element, dict) and "title" in path_element:
                element_title = path_element.get("title", "")  # Get title from dict
            elif isinstance(path_element, str):
                element_title = path_element  # Use string directly

            if element_title:  # Check if we successfully got a title string
                for section_type in self.section_types:
                    if section_type.lower() in element_title.lower():
                        return section_type
            # else: # Optional: Handle or log the case where path_element is neither string nor dict with title
            #     print(f"Warning: Unexpected element type in hierarchy path: {path_element}")

        # Default if not found
        return "General"

    def _build_hierarchy_path(self, chunk, document_tree):
        """
        Returns the hierarchy path for the chunk.
        Prioritizes using the 'hierarchy_path' already attached to the chunk.
        Returns an empty list if the path is not found on the chunk.
        """
        # Prioritize the path already attached during previous phases
        if "hierarchy_path" in chunk and chunk["hierarchy_path"]:
            return chunk["hierarchy_path"]

        # Fallback: If not present on the chunk, return empty list.
        # (Attempting to rebuild via parent links here proved unreliable previously)
        return []

    def _find_related_sections(self, chunk, document_tree):
        """Identify related sections based on title/content similarity"""
        related = []
        chunk_title = chunk.get("title", "")

        # Simple approach: match on similar titles
        if chunk_title:
            # Search for sections with similar titles
            for section in self._flatten_sections(document_tree):
                section_title = section.get("title", "")
                if (
                    section_title
                    and section_title != chunk_title
                    and (section_title in chunk_title or chunk_title in section_title)
                ):
                    # Found related section, add its path
                    related.append(self._build_hierarchy_path(section, document_tree))

        return related

    def _calculate_position(self, chunk, document_tree):
        """Calculate normalized position (0-1) of chunk in document"""
        # We'll need a flattened list of all chunks to determine position
        all_chunks = self._flatten_chunks(document_tree)

        # Find position of this chunk in the document
        chunk_index = next(
            (i for i, c in enumerate(all_chunks) if c.get("id") == chunk.get("id")), 0
        )

        # Normalize to 0-1 range
        total_chunks = len(all_chunks)
        return chunk_index / max(total_chunks - 1, 1)

    def _contains_links(self, chunk):
        """Check if chunk contains links"""
        content = chunk.get("content", "")
        # Simple check for markdown link pattern
        return "[" in content and "](" in content

    def _count_links(self, chunk):
        """Count number of links in chunk"""
        content = chunk.get("content", "")
        # Count markdown link patterns
        import re

        link_pattern = r"\[.*?\]\(.*?\)"
        links = re.findall(link_pattern, content)
        return len(links)

    def _flatten_sections(self, document_tree):
        """Flatten the document tree into a list of sections"""
        sections = []

        def traverse(node):
            # Add this node if it's a header/section
            if node.get("type") == "header":
                sections.append(node)

            # Recursively process children
            for child in node.get("children", []):
                traverse(child)

        traverse(document_tree)
        return sections

    def _flatten_chunks(self, document_tree):
        """Flatten the document into a list of all chunks"""
        chunks = []

        def traverse(node):
            # Add this node if it's a content chunk
            if "content" in node and node["content"]:
                chunks.append(node)

            # Recursively process children
            for child in node.get("children", []):
                traverse(child)

        traverse(document_tree)
        return chunks

    def enhance_chunk_text(self, chunk):
        """Enrich chunk text with contextual information"""
        enhanced_text = ""

        # Add hierarchy context if available
        hierarchy_path = chunk.get("metadata", {}).get("hierarchy_path", [])
        if hierarchy_path:
            # Extract titles from path elements (assuming dicts with 'title' or strings)
            path_titles = [
                p.get("title", str(p)) if isinstance(p, dict) else str(p)
                for p in hierarchy_path
            ]
            breadcrumb = " > ".join(path_titles)
            enhanced_text += f"Context: {breadcrumb}\n\n"

        # Add title if available
        if chunk.get("title"):
            header_level = chunk.get("metadata", {}).get("header_level", 2)
            header_markup = "#" * header_level
            enhanced_text += f"{header_markup} {chunk['title']}\n\n"

        # Add the original content
        enhanced_text += chunk.get("content", "")

        # Add related section references if available
        related_sections = chunk.get("metadata", {}).get("related_sections", [])
        if related_sections:
            enhanced_text += "\n\nRelated sections:\n"
            for section_path_list in related_sections:
                # Handle cases where related_sections might contain single paths or lists of paths
                # Assuming each 'section' reference is a list representing a path
                if isinstance(section_path_list, list):
                    # Extract titles from path elements (assuming dicts with 'title' or strings)
                    path_titles = [
                        p.get("title", str(p)) if isinstance(p, dict) else str(p)
                        for p in section_path_list
                    ]
                    section_path_str = " > ".join(path_titles)
                else:
                    # Fallback if it's not a list (e.g., just a string)
                    section_path_str = str(section_path_list)
                enhanced_text += f"- {section_path_str}\n"

        # Update the chunk with enhanced text
        chunk["enhanced_text"] = enhanced_text

        return chunk

    def normalize_metadata(self, chunks):
        """Ensure consistent metadata across chunks"""
        # Collect all section types for normalization
        section_types = set()
        content_types = set()

        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            if "section_type" in metadata:
                section_types.add(metadata["section_type"])
            if "content_type" in metadata:
                content_types.add(metadata["content_type"])

        # Normalize section types (ensure consistent casing, etc.)
        section_type_map = {}
        for section_type in section_types:
            normalized = section_type.strip().title()
            section_type_map[section_type] = normalized

        # Normalize content types
        content_type_map = {
            "link_list": "Link Collection",
            "descriptive_text": "Descriptive Content",
            "code_example": "Code Example",
            "introduction": "Introduction",
            "table_based": "Table Content",
        }

        # Apply normalization to all chunks
        for chunk in chunks:
            metadata = chunk.get("metadata", {})

            # Normalize section type
            if "section_type" in metadata:
                orig_type = metadata["section_type"]
                if orig_type in section_type_map:
                    metadata["section_type"] = section_type_map[orig_type]

            # Normalize content type
            if "content_type" in metadata:
                orig_type = metadata["content_type"]
                if orig_type in content_type_map:
                    metadata["content_type"] = content_type_map[orig_type]

        return chunks

    def process_chunks(self, chunks, document_tree):
        """Process all chunks with metadata enrichment and text enhancement"""
        enriched_chunks = []

        for chunk in chunks:
            # Add metadata
            enriched_chunk = self.enrich_chunk(chunk, document_tree)
            # Enhance text
            enhanced_chunk = self.enhance_chunk_text(enriched_chunk)
            enriched_chunks.append(enhanced_chunk)

        # Normalize metadata across all chunks
        normalized_chunks = self.normalize_metadata(enriched_chunks)

        return normalized_chunks
