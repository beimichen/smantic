#!/usr/bin/env python3
"""
Block splitting strategies for hierarchical chunking.

Implements AST-based code splitting, row-group table splitting,
and environment-based formula splitting.
"""

import ast
import logging
import re

logger = logging.getLogger(__name__)


class CodeBlockSplitter:
    """Split large code blocks by AST boundaries (functions/classes)."""
    
    def split_python_code(self, code: str) -> list[tuple[str, dict]]:
        """
        Split Python code by function/class boundaries.
        
        Returns:
            List of (content, context) tuples where context contains AST metadata
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            logger.warning(f"Failed to parse Python code: {e}")
            return self._split_code_heuristic(code)
        
        # Extract imports (shared context)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_src = ast.get_source_segment(code, node)
                if import_src:
                    imports.append(import_src)
        
        imports_text = '\n'.join(imports) + '\n\n' if imports else ''
        
        # Extract top-level functions and classes
        children = []
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                node_source = ast.get_source_segment(code, node)
                if not node_source:
                    continue
                
                # Prepend imports for context
                child_content = imports_text + node_source
                
                # end_lineno available in Python 3.8+; fallback to lineno if missing
                end_lineno = getattr(node, 'end_lineno', None)
                context = {
                    'type': type(node).__name__,
                    'name': node.name,
                    'lineno': node.lineno,
                    'end_lineno': end_lineno,  # May be None on older Python
                }
                children.append((child_content, context))
        
        # If no functions/classes found, return original
        if not children:
            return [(code, {'type': 'module'})]
        
        return children
    
    def split_javascript_code(self, code: str) -> list[tuple[str, dict]]:
        """
        Split JavaScript/TypeScript code by function boundaries.
        
        Uses regex-based heuristics since we don't have a JS AST parser.
        """
        # Pattern for function declarations and arrow functions
        function_pattern = r'((?:export\s+)?(?:async\s+)?(?:function\s+\w+|const\s+\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>))'
        
        matches = list(re.finditer(function_pattern, code, re.MULTILINE))
        
        if len(matches) < 2:
            # Not enough functions to split
            return [(code, {'type': 'module'})]
        
        children = []
        for i, match in enumerate(matches):
            start = match.start()
            # Next function or end of code
            end = matches[i + 1].start() if i + 1 < len(matches) else len(code)
            
            func_code = code[start:end].strip()
            
            # Extract function name
            name_match = re.search(r'function\s+(\w+)|const\s+(\w+)', match.group(0))
            func_name = name_match.group(1) or name_match.group(2) if name_match else 'anonymous'
            
            context = {
                'type': 'FunctionDeclaration',
                'name': func_name,
                'start_offset': start,
                'end_offset': end,
            }
            children.append((func_code, context))
        
        return children if children else [(code, {'type': 'module'})]
    
    def _split_code_heuristic(self, code: str) -> list[tuple[str, dict]]:
        """
        Fallback heuristic splitting for unrecognized languages.
        
        Splits on blank lines and function-like patterns.
        """
        # Split on double blank lines or function-like patterns
        sections = re.split(r'\n\n+|(?=\n(?:def|function|class|fn)\s+\w+)', code)
        
        children = []
        for i, section in enumerate(sections):
            section = section.strip()
            if section:
                context = {
                    'type': 'section',
                    'section_index': i,
                    'split_method': 'heuristic'
                }
                children.append((section, context))
        
        return children if children else [(code, {'type': 'module'})]


class TableBlockSplitter:
    """Split large tables by row groups with headers repeated."""
    
    def split_table(
        self,
        content: str,
        target_tokens: int,
        token_counter,
        metadata: dict | None = None
    ) -> list[tuple[str, dict]]:
        """
        Split table by row groups based on TOKEN BUDGET.
        
        Row widths vary significantly, so we accumulate until target is reached.
        
        Args:
            content: Table content (markdown or plain text)
            target_tokens: Target tokens per child chunk
            token_counter: Function to count tokens
            metadata: Optional table metadata (OTSL structure, etc.)
        
        Returns:
            List of (content, context) tuples
        """
        # Check for structured OTSL data
        if metadata and 'otsl_structure' in metadata:
            return self._split_otsl_table(content, metadata['otsl_structure'], target_tokens, token_counter)
        
        # Parse markdown-style table
        lines = content.strip().split('\n')
        if not lines:
            return [(content, {'row_start': 0, 'row_end': 0})]
        
        # Identify header via markdown separator line (|---|---|).
        # Only rows ABOVE a separator are headers. If no separator exists,
        # the table has no header — don't fabricate one (fixes bibliography
        # tables where the first row is a citation, not a column header).
        separator_re = re.compile(r'^\s*\|[\s\-:|]+\|\s*$')
        separator_idx = None
        for i, line in enumerate(lines[:5]):  # Separator is always near the top
            if separator_re.match(line):
                separator_idx = i
                break

        if separator_idx is not None:
            # Header = rows above separator + the separator line itself
            header_lines = lines[:separator_idx + 1]
            data_lines = lines[separator_idx + 1:]
        else:
            # No markdown separator — no header to repeat
            header_lines = []
            data_lines = lines
        
        if not data_lines:
            # Table is all headers, no data
            return [(content, {'row_start': 0, 'row_end': 0})]
        
        header = '\n'.join(header_lines) if header_lines else ''
        header_tokens = token_counter(header) if header else 0

        # Split data rows into groups based on TOKEN BUDGET
        row_groups = []
        current_group = []
        current_tokens = header_tokens  # Header (if any) is repeated in each child

        for row in data_lines:
            row_tokens = token_counter(row)
            if current_tokens + row_tokens > target_tokens and current_group:
                # Current group is full, start new one
                row_groups.append(current_group)
                current_group = [row]
                current_tokens = header_tokens + row_tokens
            else:
                current_group.append(row)
                current_tokens += row_tokens

        # Don't forget the last group
        if current_group:
            row_groups.append(current_group)

        # Create child contents, repeating header only if one was detected
        children = []
        row_offset = 0
        for rows in row_groups:
            if header:
                child_content = header + '\n' + '\n'.join(rows)
            else:
                child_content = '\n'.join(rows)
            context = {
                'row_start': row_offset,
                'row_end': row_offset + len(rows),
                'column_headers': header_lines[0] if header_lines else '',
            }
            children.append((child_content, context))
            row_offset += len(rows)
        
        return children if children else [(content, {'row_start': 0, 'row_end': len(data_lines)})]
    
    def _split_otsl_table(
        self,
        content: str,
        otsl_structure: dict,
        target_tokens: int,
        token_counter
    ) -> list[tuple[str, dict]]:
        """
        Split table using OTSL structure information.
        
        OTSL provides structured cell data which we can use for precise splitting.
        """
        # For now, fall back to text-based splitting
        # TODO: Implement OTSL-aware splitting when OTSL structure is available
        logger.info("OTSL table splitting not yet implemented, using text-based fallback")
        return self.split_table(content, target_tokens, token_counter, metadata=None)


class FormulaBlockSplitter:
    """Split large derivations/equation sequences by environment or step."""
    
    def split_formula(self, content: str) -> list[tuple[str, dict]]:
        """
        Split large formula block by LaTeX environments or steps.
        
        Returns:
            List of (content, context) tuples
        """
        # Detect LaTeX environments
        env_pattern = r'\\begin\{(equation|align|gather|eqnarray)\*?\}.*?\\end\{\1\*?\}'
        environments = list(re.finditer(env_pattern, content, re.DOTALL))
        
        if len(environments) > 1:
            # Multiple environments: each becomes a child
            children = []
            for match in environments:
                # Store actual character offsets from regex match for provenance
                context = {
                    'env': match.group(1),
                    'span_start': match.start(),
                    'span_end': match.end(),
                    'split_method': 'environment'
                }
                children.append((match.group(0), context))
            return children
        
        elif len(environments) == 1:
            # Single large environment: split by \\ or numbered steps
            env_content = environments[0].group(0)
            steps = re.split(r'\\\\(?:\s*\n)?', env_content)
            
            if len(steps) > 3:
                # For step splits, we don't have precise character offsets
                children = [
                    (step.strip(), {'step': i, 'split_method': 'step_based'})
                    for i, step in enumerate(steps) if step.strip()
                ]
                return children if children else [(content, {'split_method': 'none'})]
            else:
                # Too few steps, keep intact
                return [(content, {'split_method': 'none'})]
        
        else:
            # No LaTeX environments detected, keep intact
            return [(content, {'split_method': 'none'})]


# Singleton instances
code_splitter = CodeBlockSplitter()
table_splitter = TableBlockSplitter()
formula_splitter = FormulaBlockSplitter()