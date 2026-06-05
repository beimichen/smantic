#!/usr/bin/env python3
"""
Tests for block splitting strategies (Phase 2).

Tests AST-based code splitting, row-group table splitting,
and environment-based formula splitting.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from smantic.block_splitters import (
    CodeBlockSplitter,
    FormulaBlockSplitter,
    TableBlockSplitter,
    code_splitter,
    formula_splitter,
    table_splitter,
)


class TestCodeBlockSplitter:
    """Test Python and JavaScript AST-based code splitting."""
    
    def test_split_simple_python_functions(self):
        """Python code with multiple functions splits by function boundaries."""
        code = '''import numpy as np
import pandas as pd

def calculate_mean(data):
    """Calculate mean of data."""
    return sum(data) / len(data)

def calculate_std(data):
    """Calculate standard deviation."""
    mean = calculate_mean(data)
    variance = sum((x - mean) ** 2 for x in data) / len(data)
    return variance ** 0.5

def process_data(data):
    """Process data and return stats."""
    return {
        'mean': calculate_mean(data),
        'std': calculate_std(data)
    }
'''
        
        splitter = CodeBlockSplitter()
        children = splitter.split_python_code(code)
        
        # Should split into 3 functions
        assert len(children) == 3
        
        # Each child should contain the function and imports
        for content, context in children:
            assert 'import numpy as np' in content
            assert 'import pandas as pd' in content
            assert context['type'] in ['FunctionDef', 'AsyncFunctionDef']
            assert 'name' in context
            assert 'lineno' in context
        
        # Check function names
        names = [ctx['name'] for _, ctx in children]
        assert 'calculate_mean' in names
        assert 'calculate_std' in names
        assert 'process_data' in names
    
    def test_split_python_classes(self):
        """Python code with classes splits by class boundaries."""
        code = '''class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    def process(self):
        return [x * 2 for x in self.data]

class ResultFormatter:
    def format(self, result):
        return str(result)
'''
        
        splitter = CodeBlockSplitter()
        children = splitter.split_python_code(code)
        
        # Should split into 2 classes
        assert len(children) == 2
        
        names = [ctx['name'] for _, ctx in children]
        assert 'DataProcessor' in names
        assert 'ResultFormatter' in names
    
    def test_split_python_mixed_functions_and_classes(self):
        """Python code with both functions and classes."""
        code = '''def helper_function():
    return 42

class MyClass:
    def method(self):
        return helper_function()

async def async_function():
    return await some_task()
'''
        
        splitter = CodeBlockSplitter()
        children = splitter.split_python_code(code)
        
        # Should split into 3 items
        assert len(children) == 3
        
        types = [ctx['type'] for _, ctx in children]
        assert 'FunctionDef' in types
        assert 'ClassDef' in types
        assert 'AsyncFunctionDef' in types
    
    def test_split_python_no_functions(self):
        """Python code with no top-level functions returns as single module."""
        code = '''# Just some variables
x = 1
y = 2
z = x + y
'''
        
        splitter = CodeBlockSplitter()
        children = splitter.split_python_code(code)
        
        # Should return as single module
        assert len(children) == 1
        content, context = children[0]
        assert context['type'] == 'module'
        assert 'x = 1' in content
    
    def test_split_python_syntax_error_fallback(self):
        """Python code with syntax errors falls back to heuristic splitting."""
        code = '''def broken_function(
    # Missing closing parenthesis
    return 42

def another_function():
    return 99
'''
        
        splitter = CodeBlockSplitter()
        children = splitter.split_python_code(code)
        
        # Should use heuristic fallback
        assert len(children) > 0
        for _, context in children:
            # Heuristic splitting produces different context
            assert context.get('split_method') == 'heuristic' or context['type'] in ['FunctionDef', 'section']
    
    def test_split_javascript_functions(self):
        """JavaScript code with multiple functions splits by function boundaries."""
        code = '''export async function fetchData(url) {
    const response = await fetch(url);
    return response.json();
}

function processData(data) {
    return data.map(item => item * 2);
}

const formatResult = (result) => {
    return JSON.stringify(result);
};
'''
        
        splitter = CodeBlockSplitter()
        children = splitter.split_javascript_code(code)
        
        # Should split into multiple functions
        assert len(children) >= 2
        
        # Check that each child has context
        for _content, context in children:
            assert context['type'] == 'FunctionDeclaration'
            assert 'name' in context
            assert 'start_offset' in context
    
    def test_split_javascript_single_function(self):
        """JavaScript code with single function returns as module."""
        code = '''function singleFunction() {
    return "Hello World";
}
'''
        
        splitter = CodeBlockSplitter()
        children = splitter.split_javascript_code(code)
        
        # Single function returns as module (not worth splitting)
        assert len(children) == 1
        _, context = children[0]
        assert context['type'] == 'module'
    
    def test_heuristic_splitting(self):
        """Heuristic splitting works for unrecognized languages."""
        code = '''func firstFunction() {
    return 1
}

func secondFunction() {
    return 2
}
'''
        
        splitter = CodeBlockSplitter()
        children = splitter._split_code_heuristic(code)
        
        # Should split by function-like patterns
        assert len(children) >= 2
        
        for _, context in children:
            assert context['split_method'] == 'heuristic'


class TestTableBlockSplitter:
    """Test row-group table splitting with headers repeated."""
    
    def simple_token_counter(self, text: str) -> int:
        """Simple token counter for testing."""
        return len(text.split())
    
    def test_split_table_by_token_budget(self):
        """Large table splits into row groups based on token budget."""
        # Create a table with header and many rows
        header = '| Name | Age | City |\n|------|-----|------|'
        rows = [f'| Person{i} | {20+i} | City{i} |' for i in range(20)]
        content = header + '\n' + '\n'.join(rows)
        
        splitter = TableBlockSplitter()
        children = splitter.split_table(
            content=content,
            target_tokens=50,  # Small target to force splitting
            token_counter=self.simple_token_counter
        )
        
        # Should split into multiple groups
        assert len(children) > 1
        
        # Each child should have headers repeated
        for child_content, context in children:
            assert '| Name | Age | City |' in child_content
            assert 'row_start' in context
            assert 'row_end' in context
            assert context['row_end'] > context['row_start']
    
    def test_split_table_headers_preserved(self):
        """Headers are repeated in each child chunk."""
        header = '| Col1 | Col2 |\n|------|------|'
        rows = [f'| A{i} | B{i} |' for i in range(10)]
        content = header + '\n' + '\n'.join(rows)
        
        splitter = TableBlockSplitter()
        children = splitter.split_table(
            content=content,
            target_tokens=30,
            token_counter=self.simple_token_counter
        )
        
        # All children should have the header
        for child_content, _ in children:
            assert '| Col1 | Col2 |' in child_content
            assert '|------|------|' in child_content
    
    def test_split_table_small_stays_intact(self):
        """Small table stays as single chunk."""
        content = '''| Name | Age |
|------|-----|
| Alice | 30 |
| Bob | 25 |'''
        
        splitter = TableBlockSplitter()
        children = splitter.split_table(
            content=content,
            target_tokens=100,
            token_counter=self.simple_token_counter
        )
        
        # Should not split
        assert len(children) == 1
        child_content, context = children[0]
        assert 'Alice' in child_content
        assert 'Bob' in child_content
    
    def test_split_table_context_includes_row_ranges(self):
        """Context includes row start/end indices."""
        header = '| A | B |\n|---|---|'
        rows = [f'| {i} | {i*2} |' for i in range(15)]
        content = header + '\n' + '\n'.join(rows)
        
        splitter = TableBlockSplitter()
        children = splitter.split_table(
            content=content,
            target_tokens=40,
            token_counter=self.simple_token_counter
        )
        
        # Check row ranges are sequential and non-overlapping
        prev_end = 0
        for _, context in children:
            assert context['row_start'] == prev_end
            assert context['row_end'] > context['row_start']
            prev_end = context['row_end']
        
        # Should cover all rows
        assert prev_end == 15
    
    def test_split_table_empty(self):
        """Empty table returns single chunk."""
        content = ""
        
        splitter = TableBlockSplitter()
        children = splitter.split_table(
            content=content,
            target_tokens=100,
            token_counter=self.simple_token_counter
        )
        
        assert len(children) == 1
    
    def test_split_table_headers_only(self):
        """Table with only headers returns single chunk."""
        content = '''| Name | Age |
|------|-----|'''
        
        splitter = TableBlockSplitter()
        children = splitter.split_table(
            content=content,
            target_tokens=100,
            token_counter=self.simple_token_counter
        )
        
        assert len(children) == 1


class TestFormulaBlockSplitter:
    """Test environment-based formula splitting."""
    
    def test_split_multiple_environments(self):
        """Multiple LaTeX environments split into separate children."""
        content = r'''
\begin{equation}
E = mc^2
\end{equation}

\begin{align}
F &= ma \\
a &= F/m
\end{align}

\begin{gather}
x + y = z \\
z = 10
\end{gather}
'''
        
        splitter = FormulaBlockSplitter()
        children = splitter.split_formula(content)
        
        # Should split into 3 environments
        assert len(children) == 3
        
        # Check each environment is captured
        for child_content, context in children:
            assert r'\begin{' in child_content
            assert r'\end{' in child_content
            assert 'env' in context
            assert context['env'] in ['equation', 'align', 'gather']
            assert context['split_method'] == 'environment'
    
    def test_split_single_large_environment_by_steps(self):
        """Single large environment with many steps splits by \\\\ boundaries."""
        content = r'''
\begin{align}
x^2 + 2x + 1 &= 0 \\
(x + 1)^2 &= 0 \\
x + 1 &= 0 \\
x &= -1
\end{align}
'''
        
        splitter = FormulaBlockSplitter()
        children = splitter.split_formula(content)
        
        # Should split into multiple steps
        assert len(children) > 1
        
        for _, context in children:
            assert 'step' in context
            assert context['split_method'] == 'step_based'
    
    def test_split_single_small_environment_stays_intact(self):
        """Single small environment with few steps stays intact."""
        content = r'''
\begin{equation}
E = mc^2
\end{equation}
'''
        
        splitter = FormulaBlockSplitter()
        children = splitter.split_formula(content)
        
        # Should not split
        assert len(children) == 1
        _, context = children[0]
        assert context['split_method'] == 'none'
    
    def test_split_no_latex_environments(self):
        """Plain text with no LaTeX stays intact."""
        content = "E = mc^2\nF = ma"
        
        splitter = FormulaBlockSplitter()
        children = splitter.split_formula(content)
        
        # Should not split
        assert len(children) == 1
        child_content, context = children[0]
        assert child_content == content
        assert context['split_method'] == 'none'
    
    def test_split_equation_star_environment(self):
        """Starred environments (equation*) are recognized."""
        content = r'''
\begin{equation*}
x = y + z
\end{equation*}

\begin{align*}
a &= b \\
c &= d
\end{align*}
'''
        
        splitter = FormulaBlockSplitter()
        children = splitter.split_formula(content)
        
        # Should split into 2 environments
        assert len(children) == 2


class TestSingletonInstances:
    """Test that singleton instances work correctly."""
    
    def test_code_splitter_singleton(self):
        """code_splitter singleton is usable."""
        code = '''def test():
    return 42
'''
        children = code_splitter.split_python_code(code)
        assert len(children) >= 1
    
    def test_table_splitter_singleton(self):
        """table_splitter singleton is usable."""
        content = "| A | B |\n|---|---|\n| 1 | 2 |"

        def counter(text):
            return len(text.split())

        children = table_splitter.split_table(content, 100, counter)
        assert len(children) >= 1
    
    def test_formula_splitter_singleton(self):
        """formula_splitter singleton is usable."""
        content = r"\begin{equation}E=mc^2\end{equation}"
        children = formula_splitter.split_formula(content)
        assert len(children) >= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])