import re
from markdown_it import MarkdownIt
from markdown_it.token import Token
# Note: markdown_it.tree is not directly used in the provided code snippets,
# but keeping it in case future extensions need it or if it's implicitly used.
from markdown_it.tree import SyntaxTreeNode

class MarkdownProcessor:
    def __init__(self):
        """Initialize the Markdown processor."""
        # Initialize with markdown-it for baseline parsing
        self.md_parser = MarkdownIt("commonmark")

    def parse_document(self, markdown_text):
        """Parse markdown into structured components"""
        # Get tokens from parser
        tokens = self.md_parser.parse(markdown_text)

        # Initialize document structure
        document = {
            "title": None,
            "introduction": [],
            "headers": [],
            "content_blocks": [],
            "links": [],
            "code_blocks": []
        }

        # Tracking variables
        current_header = None
        current_header_level = 0
        current_content = []
        in_introduction = True # Assume content before the first header is introduction

        # Iterate through tokens to extract structure
        for i, token in enumerate(tokens):
            # Header detection
            if token.type == "heading_open":
                # Finalize the previous content block before starting a new header
                if current_content:
                    content_str = "".join(current_content).strip()
                    if content_str: # Only add if there's actual content
                        if in_introduction and not document["introduction"]:
                            document["introduction"] = content_str
                        else:
                            document["content_blocks"].append({
                                "header": current_header,
                                "header_level": current_header_level,
                                "content": content_str
                            })
                    current_content = [] # Reset content for the new section

                current_header_level = int(token.tag[1])
                in_introduction = False # Any header means we are past the introduction

            elif token.type == "inline" and tokens[i-1].type == "heading_open": # Check previous token
                header_text = token.content.strip()

                # Set document title if this is the first h1
                if current_header_level == 1 and document["title"] is None:
                    document["title"] = header_text

                current_header = header_text
                document["headers"].append({
                    "text": header_text,
                    "level": current_header_level
                })

            # Code blocks
            elif token.type == "fence":
                document["code_blocks"].append({
                    "header": current_header, # Associate with the current header context
                    "language": token.info.strip() if token.info else None,
                    "content": token.content.strip()
                })
                # Add code block content to current_content as well, so it's part of the section
                current_content.append(f"\n```\n{token.content}```\n")


            # Links
            elif token.type == "link_open":
                link_href = token.attrGet("href")
                link_text = ""
                # Look ahead for the text token within the link
                if i + 1 < len(tokens) and tokens[i+1].type == "text":
                    link_text = tokens[i+1].content
                # Look ahead for the closing token
                if i + 2 < len(tokens) and tokens[i+2].type == "link_close":
                     document["links"].append({
                        "text": link_text,
                        "href": link_href,
                        "header": current_header # Associate with the current header context
                    })
                     # Add link markdown to current_content
                     current_content.append(f"[{link_text}]({link_href})")


            # Regular content - accumulate text from various relevant tokens
            elif token.type in ["text", "paragraph_open", "paragraph_close",
                               "bullet_list_open", "bullet_list_close",
                               "list_item_open", "list_item_close", "inline"]:
                 # We specifically care about 'text' and 'inline' content here
                 # Paragraph/list markers help structure but don't contain the text itself directly in token.content
                 # 'inline' type often contains the actual text content within paragraphs or list items
                if token.content:
                    # Append content, ensuring spaces are handled reasonably between elements
                    if current_content and not current_content[-1].endswith('\n'):
                         current_content.append(" ") # Add space if needed
                    current_content.append(token.content)

            # Add line breaks after block elements like paragraph_close for better spacing
            elif token.type in ["paragraph_close", "bullet_list_close"]:
                 if current_content and not current_content[-1].endswith('\n'):
                     current_content.append("\n")


        # Add any remaining content after the last header
        if current_content:
            content_str = "".join(current_content).strip()
            if content_str:
                 # If no headers were found at all, this is all introduction
                 if in_introduction and not document["introduction"]:
                     document["introduction"] = content_str
                 else:
                    document["content_blocks"].append({
                        "header": current_header,
                        "header_level": current_header_level,
                        "content": content_str
                    })

        # Refine introduction: If introduction is empty but first content block has no header, use that.
        if not document["introduction"] and document["content_blocks"]:
            first_block = document["content_blocks"][0]
            # Check if the first block is effectively introduction (associated with header level 0 or no header)
            if first_block["header"] is None or first_block["header_level"] == 0:
                 document["introduction"] = first_block["content"]
                 # Remove it from content_blocks if it was truly introductory
                 if first_block["header"] is None:
                     document["content_blocks"].pop(0)


        return document

    def build_hierarchy_tree(self, parsed_content):
        """Build document tree with parent-child relationships"""
        # Initialize tree with root node using document title and introduction
        tree = {
            "type": "document",
            "title": parsed_content.get("title", "Untitled Document"), # Default title
            "content": parsed_content.get("introduction", ""), # Use extracted introduction
            "metadata": {
                "section_type": "root",
                "content_type": self.classify_content({"content": parsed_content.get("introduction", "")}) if parsed_content.get("introduction") else None
            },
            "children": []
        }

        # Stack to track current position in hierarchy, starting with the root
        # Each item: {"node": node_dict, "level": header_level}
        node_stack = [{"node": tree, "level": 0}]

        # Process headers to build the main structure
        for header_info in parsed_content["headers"]:
            level = header_info["level"]
            text = header_info["text"]

            # Find the correct parent node in the stack based on header level
            while node_stack[-1]["level"] >= level:
                node_stack.pop()
            parent_node = node_stack[-1]["node"]

            # Create the new header node
            header_node = {
                "type": "header",
                "title": text,
                "level": level,
                "content": "",  # Initialize content, will be populated later
                "metadata": {
                    "section_type": self._determine_section_type(text),
                    "content_type": None # Will be classified later
                },
                "children": []
            }

            # Add the new node to its parent's children list
            parent_node["children"].append(header_node)
            # Push the new node onto the stack
            node_stack.append({"node": header_node, "level": level})

        # Associate content blocks with the correct header nodes in the tree
        # Create a mapping from header text to node for easier lookup
        header_node_map = {}
        def traverse_and_map(node):
            if node.get("type") == "header":
                header_node_map[node["title"]] = node
            for child in node.get("children", []):
                traverse_and_map(child)
        traverse_and_map(tree)


        # Add content blocks to their corresponding header nodes
        for block in parsed_content["content_blocks"]:
            header_text = block["header"]
            content_text = block["content"]

            target_node = header_node_map.get(header_text)

            if target_node:
                # Append content to the header node itself
                # Simple approach: concatenate content. Could be refined later.
                if not target_node["content"]:
                    target_node["content"] = content_text
                else:
                    # If content already exists, append with a newline separator
                    target_node["content"] += "\n\n" + content_text
            # else:
                # Optional: Handle content blocks that couldn't be matched to a header
                # Could add them to a general 'unassociated_content' list or similar

        # Associate code blocks and links with the most recent header context
        # This requires iterating through the original parsed structure again or enhancing it
        # For simplicity here, we might add them as metadata or children of relevant nodes later
        # The current build_hierarchy focuses on header-content association.

        # Add code blocks to the tree (simple approach: attach to header nodes)
        for code_block in parsed_content.get("code_blocks", []):
            header_text = code_block.get("header")
            target_node = header_node_map.get(header_text)
            if target_node:
                 code_node = {
                     "type": "code_block",
                     "language": code_block.get("language"),
                     "content": code_block.get("content"),
                     "metadata": {"section_type": "code", "content_type": "code_example"},
                     "children": []
                 }
                 # Add as child or append to a specific list in metadata
                 if "code_blocks" not in target_node:
                     target_node["code_blocks"] = []
                 target_node["code_blocks"].append(code_node) # Or append directly to children

        # Add links similarly (e.g., store in metadata)
        for link in parsed_content.get("links", []):
            header_text = link.get("header")
            target_node = header_node_map.get(header_text)
            if target_node:
                if "links" not in target_node.get("metadata", {}):
                     if "metadata" not in target_node: target_node["metadata"] = {}
                     target_node["metadata"]["links"] = []
                target_node["metadata"]["links"].append({"text": link.get("text"), "href": link.get("href")})


        return tree

    def _determine_section_type(self, header_text):
        """Determine the section type based on keywords in the header text"""
        if not header_text: return "General" # Handle empty or None header text
        lower_text = header_text.lower()
        if "api" in lower_text or "endpoint" in lower_text:
            return "API documentation"
        elif "concept" in lower_text or "overview" in lower_text:
            return "Concepts documentation"
        elif "example" in lower_text or "usage" in lower_text:
            return "Examples"
        elif "install" in lower_text or "setup" in lower_text:
            return "Installation/Setup"
        elif "tutorial" in lower_text or "guide" in lower_text:
            return "Tutorial/Guide"
        elif "optional" in lower_text:
            return "Optional"
        elif "internal" in lower_text or "advanced" in lower_text:
            return "Internals/Advanced"
        else:
            return "General"

    def classify_content(self, node):
        """Determine content type of a node based on its content"""
        content = node.get("content", "")
        if not content or not isinstance(content, str):
            return None # No content or not string type

        # Basic metrics
        lines = content.split('\n')
        num_lines = len(lines)
        num_words = len(content.split())
        content_length = len(content)

        if content_length == 0: return None # Empty content

        # Regex patterns
        link_pattern = r"\[.*?\]\(.*?\)"
        code_block_pattern = r"```[\s\S]*?```" # Fenced code blocks
        inline_code_pattern = r"`[^`]+`" # Inline code
        table_pattern = r"\|.*\|" # Basic table row detection
        list_pattern = r"^\s*[-*+]\s+" # Basic list item start

        # Calculate densities/counts
        link_matches = re.findall(link_pattern, content)
        link_density = len(link_matches) / max(1, num_lines)

        code_block_matches = re.findall(code_block_pattern, content)
        inline_code_matches = re.findall(inline_code_pattern, content)
        # Consider both block and inline code presence
        code_density = (len(code_block_matches) + len(inline_code_matches)) / max(1, num_words)

        table_rows = re.findall(table_pattern, content, re.MULTILINE)
        list_items = re.findall(list_pattern, content, re.MULTILINE)

        # Classification logic (refined)
        if len(code_block_matches) > 0 or code_density > 0.1: # Presence of code blocks or high inline code density
            return "code_example"
        elif len(table_rows) >= 2: # More than one table row suggests a table
             # Check if it's mostly table separators
             if content.count('|') / max(1, content_length) > 0.05:
                 return "table_based"
        elif link_density > 0.3 and num_words < 150: # High link density in shorter sections
            return "link_list"
        elif len(list_items) / max(1, num_lines) > 0.5: # Majority of lines are list items
             return "list_based"
        elif num_words < 50 and node.get("level", 0) > 0: # Very short content under a header might be introductory for that section
             # Check if it's the first piece of content under that header? (More complex logic needed)
             # Simple check for brevity:
             return "short_description" # Or potentially "section_introduction"
        elif num_words < 100 and node.get("type") == "document": # Short root content
             return "introduction"


        # Default fallback
        return "descriptive_text"


    def apply_classification(self, tree_node):
        """Apply content classification recursively to the entire document tree"""
        # Classify the current node's content
        content_type = self.classify_content(tree_node)
        if content_type:
            # Ensure metadata dictionary exists
            if "metadata" not in tree_node:
                tree_node["metadata"] = {}
            tree_node["metadata"]["content_type"] = content_type

        # Recursively process all children
        for child in tree_node.get("children", []):
            self.apply_classification(child)

        # Return the node (though modification is in-place)
        return tree_node