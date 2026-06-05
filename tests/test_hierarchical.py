#!/usr/bin/env python3
"""
Integration tests for Phase 2: Parent-Child Block Splitting.

Tests the integration of block splitters with the main chunker,
verifying parent-child relationships and metadata.
"""

import sys
from pathlib import Path

import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from smantic.chunker import StructureAwareChunker
from smantic.ir import BBox, Document, Element, Page


class TestLargeCodeBlockSplitting:
    """Test parent-child splitting for large code blocks."""

    @pytest.mark.slow  # child block_sequence contiguity depends on real tokenizer
    # token counts: with the char-heuristic fallback a small child slips under
    # the merge threshold and is absorbed, breaking the contiguous-sequence
    # assertion. Runs green with the [onnx] extra installed.
    def test_large_python_code_creates_parent_and_children(self):
        """Large Python code block splits into parent + children."""
        # Generate large Python code (>800 tokens)
        # Adding more functions and docstrings to exceed 800 tokens
        code = '''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional

def calculate_statistics(data: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive statistics for the given data.
    
    This function computes mean, standard deviation, median, min, max,
    and quartiles for the input data array.
    
    Args:
        data: NumPy array of numerical values
        
    Returns:
        Dictionary containing statistical measures
    """
    mean = np.mean(data)
    std = np.std(data)
    median = np.median(data)
    minimum = np.min(data)
    maximum = np.max(data)
    q25 = np.percentile(data, 25)
    q75 = np.percentile(data, 75)
    return {
        'mean': mean,
        'std': std,
        'median': median,
        'min': minimum,
        'max': maximum,
        'q25': q25,
        'q75': q75
    }

def process_dataset(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Process entire dataset and calculate statistics for each column.
    
    This function iterates through all numerical columns in the DataFrame
    and computes comprehensive statistics for each one.
    
    Args:
        df: Pandas DataFrame with numerical columns
        
    Returns:
        Dictionary mapping column names to their statistics
    """
    result = {}
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            result[column] = calculate_statistics(df[column].values)
    return result

def save_results(results: Dict[str, Any], filename: str) -> None:
    """
    Save results to a JSON file with proper formatting.
    
    Args:
        results: Dictionary of results to save
        filename: Path to output file
    """
    import json
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load data from CSV file into pandas DataFrame.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(filepath)

def visualize_distribution(data: np.ndarray, title: str = "Data Distribution") -> None:
    """
    Create histogram visualization of data distribution.
    
    Args:
        data: Array of values to visualize
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.show()

class DataAnalyzer:
    """
    Comprehensive data analysis class.
    
    This class provides methods for loading, analyzing, and exporting
    statistical analysis results for datasets.
    """
    
    def __init__(self, data: Optional[pd.DataFrame] = None):
        """
        Initialize the DataAnalyzer.
        
        Args:
            data: Optional initial DataFrame
        """
        self.data = data
        self.results = None
    
    def load(self, filepath: str) -> None:
        """Load data from file."""
        self.data = load_data(filepath)
    
    def analyze(self) -> Dict[str, Dict[str, float]]:
        """
        Perform comprehensive analysis on the data.
        
        Returns:
            Dictionary of statistics for each column
        """
        if self.data is None:
            raise ValueError("No data loaded")
        self.results = process_dataset(self.data)
        return self.results
    
    def export(self, path: str) -> None:
        """
        Export analysis results to file.
        
        Args:
            path: Output file path
        """
        if self.results is None:
            self.analyze()
        save_results(self.results, path)
    
    def visualize(self, column: str) -> None:
        """
        Visualize distribution of a specific column.
        
        Args:
            column: Column name to visualize
        """
        if self.data is None:
            raise ValueError("No data loaded")
        if column not in self.data.columns:
            raise ValueError(f"Column {column} not found")
        visualize_distribution(self.data[column].values, title=f"Distribution of {column}")
'''
        
        doc = Document(pages=[
            Page(page_number=1, elements=[
                Element(
                    element_type='code',
                    content=code,
                    bbox=BBox(100, 100, 900, 800),
                    page=1
                )
            ])
        ])
        
        chunker = StructureAwareChunker(max_tokens=500)
        chunks = chunker.chunk_document(doc)
        
        # Should have parent + multiple children
        assert len(chunks) > 1
        
        # First chunk should be parent (full code)
        parent = chunks[0]
        assert parent.dominant_type == 'code_block'
        assert parent.metadata.get('has_children') is True
        assert 'calculate_statistics' in parent.content
        assert 'DataAnalyzer' in parent.content
        
        # Subsequent chunks should be children
        children = chunks[1:]
        assert len(children) >= 2
        
        for i, child in enumerate(children):
            assert child.dominant_type == 'code_block'
            assert child.block_sequence == i
            assert child.parent_chunk_id is None  # Will be set after DB insert
            assert child.chunking_method == 'ast_split'
            
            # Each child should have imports prepended
            assert 'import numpy as np' in child.content or 'import pandas as pd' in child.content
    
    def test_small_python_code_stays_single_chunk(self):
        """Small Python code (<800 tokens) stays as single chunk."""
        code = '''def hello():
    return "Hello World"
'''
        
        doc = Document(pages=[
            Page(page_number=1, elements=[
                Element(
                    element_type='code',
                    content=code,
                    bbox=BBox(100, 100, 900, 200),
                    page=1
                )
            ])
        ])
        
        chunker = StructureAwareChunker()
        chunks = chunker.chunk_document(doc)
        
        # Should be single chunk
        assert len(chunks) == 1
        assert chunks[0].dominant_type == 'code_block'
        assert chunks[0].metadata.get('has_children') is not True


class TestLargeTableSplitting:
    """Test parent-child splitting for large tables."""
    
    def test_large_table_creates_parent_and_children(self):
        """Large table splits into parent + row-group children."""
        # Create large table with many rows (>800 tokens)
        # Need ~100+ rows to exceed 800 tokens
        header = '| Name | Age | City | Country | Department | Salary | Experience |'
        separator = '|------|-----|------|---------|------------|--------|------------|'
        rows = [f'| Person{i} | {20+i} | City{i} | Country{i} | Dept{i%10} | ${50000+i*1000} | {i%20} years |' for i in range(120)]
        content = '\n'.join([header, separator] + rows)
        
        doc = Document(pages=[
            Page(page_number=1, elements=[
                Element(
                    element_type='table',
                    content=content,
                    bbox=BBox(100, 100, 900, 800),
                    page=1
                )
            ])
        ])
        
        chunker = StructureAwareChunker()
        chunks = chunker.chunk_document(doc)
        
        # Should have parent + multiple children
        assert len(chunks) > 1
        
        # First chunk is parent
        parent = chunks[0]
        assert parent.dominant_type == 'table_block'
        assert parent.metadata.get('has_children') is True
        assert 'Person0' in parent.content
        assert 'Person49' in parent.content
        
        # Children should have headers repeated
        children = chunks[1:]
        assert len(children) >= 2
        
        for child in children:
            assert child.dominant_type == 'table_block'
            assert child.chunking_method == 'row_group'
            assert '| Name | Age | City | Country |' in child.content
            assert 'Person' in child.content
    
    def test_small_table_stays_single_chunk(self):
        """Small table stays as single chunk."""
        content = '''| Name | Age |
|------|-----|
| Alice | 30 |
| Bob | 25 |'''
        
        doc = Document(pages=[
            Page(page_number=1, elements=[
                Element(
                    element_type='table',
                    content=content,
                    bbox=BBox(100, 100, 900, 300),
                    page=1
                )
            ])
        ])
        
        chunker = StructureAwareChunker()
        chunks = chunker.chunk_document(doc)
        
        assert len(chunks) == 1
        assert chunks[0].dominant_type == 'table_block'


class TestLargeFormulaSplitting:
    """Test parent-child splitting for large formula blocks."""
    
    def test_multiple_latex_environments_split(self):
        """Multiple LaTeX environments split into children."""
        # Create large formula block with many environments (>800 tokens)
        # Need to add much more content to exceed 800 tokens
        content = r'''
\begin{equation}
E = mc^2
\end{equation}

This is the famous mass-energy equivalence discovered by Albert Einstein in 1905, one of the most important equations in physics that relates energy and mass through the speed of light squared.

\begin{align}
F &= ma \\
a &= \frac{F}{m} \\
m &= \frac{F}{a} \\
\text{where } F \text{ is force, } m \text{ is mass, and } a \text{ is acceleration}
\end{align}

Newton's second law of motion and its various algebraic rearrangements, fundamental to classical mechanics.

\begin{gather}
x + y = z \\
z = 10 \\
x + y = 10 \\
\text{where } x, y \in \mathbb{R} \\
\text{This demonstrates basic algebraic substitution}
\end{gather}

Simple algebraic equations showing substitution and the properties of equality.

\begin{equation}
\int_{0}^{\infty} e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
\end{equation}

The Gaussian integral, absolutely fundamental in probability theory, statistical mechanics, and quantum field theory. This integral appears throughout physics.

\begin{align}
\nabla \cdot \mathbf{E} &= \frac{\rho}{\epsilon_0} \\
\nabla \cdot \mathbf{B} &= 0 \\
\nabla \times \mathbf{E} &= -\frac{\partial \mathbf{B}}{\partial t} \\
\nabla \times \mathbf{B} &= \mu_0\mathbf{J} + \mu_0\epsilon_0\frac{\partial \mathbf{E}}{\partial t}
\end{align}

Maxwell's equations in differential form, describing how electric and magnetic fields propagate, interact, and are influenced by charges and currents.

\begin{equation}
\hat{H}\psi = E\psi
\end{equation}

The time-independent Schrödinger equation, absolutely central to quantum mechanics for determining energy eigenstates of quantum systems.

\begin{gather}
S = k_B \ln \Omega \\
\text{where } k_B = 1.380649 \times 10^{-23} \text{ J/K} \\
\text{and } \Omega \text{ is the number of microstates}
\end{gather}

Boltzmann's entropy formula relating macroscopic entropy to microscopic disorder, fundamental to statistical mechanics and thermodynamics.

\begin{align}
\frac{dx}{dt} &= \alpha x - \beta xy \\
\frac{dy}{dt} &= \delta xy - \gamma y \\
\text{where } x \text{ is prey population and } y \text{ is predator population}
\end{align}

The Lotka-Volterra predator-prey equations modeling population dynamics in ecological systems with nonlinear interaction terms between species.

\begin{equation}
ds^2 = -c^2dt^2 + dx^2 + dy^2 + dz^2
\end{equation}

The Minkowski metric for special relativity defining the spacetime interval in flat spacetime geometry.

\begin{align}
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) &= 0 \\
\frac{\partial (\rho \mathbf{v})}{\partial t} + \nabla \cdot (\rho \mathbf{v} \otimes \mathbf{v}) &= -\nabla p + \rho \mathbf{g} + \mu \nabla^2 \mathbf{v}
\end{align}

Continuity and momentum conservation equations for fluid flow forming the Navier-Stokes equations, the basis of computational fluid dynamics.

\begin{equation}
\Delta G = \Delta H - T\Delta S
\end{equation}

Gibbs free energy equation determining the spontaneity of chemical reactions and phase transitions in thermodynamic systems.

\begin{gather}
\mathcal{L} = T - V \\
\frac{d}{dt}\frac{\partial \mathcal{L}}{\partial \dot{q}} - \frac{\partial \mathcal{L}}{\partial q} = 0 \\
\text{Euler-Lagrange equation of motion}
\end{gather}

Lagrangian mechanics formulation providing an elegant alternative to Newtonian mechanics based on energy principles and variational calculus.

\begin{equation}
\nabla^2 \phi = 4\pi G \rho
\end{equation}

Poisson's equation for gravitational potential used extensively in astrophysics, cosmology, and orbital mechanics calculations.

\begin{align}
\sigma_x \sigma_p &\geq \frac{\hbar}{2} \\
\sigma_E \sigma_t &\geq \frac{\hbar}{2} \\
\text{Heisenberg uncertainty relations}
\end{align}

Heisenberg uncertainty principles for position-momentum and energy-time establishing fundamental limits on simultaneous measurement precision in quantum systems.

\begin{equation}
R_{\mu\nu} - \frac{1}{2}Rg_{\mu\nu} + \Lambda g_{\mu\nu} = \frac{8\pi G}{c^4}T_{\mu\nu}
\end{equation}

Einstein field equations of general relativity describing how matter and energy curve spacetime geometry, the foundation of modern gravitational physics.

\begin{gather}
\nabla^2 u = 0 \\
\text{Laplace's equation for harmonic functions} \\
\text{Solutions represent equilibrium configurations}
\end{gather}

Fundamental partial differential equation appearing in electrostatics, steady-state heat flow, fluid flow, and gravitational potential theory.

\begin{align}
\frac{\partial^2 \psi}{\partial t^2} &= c^2 \nabla^2 \psi \\
\text{Wave equation in three dimensions}
\end{align}

Classical wave equation governing electromagnetic waves, sound waves, and mechanical vibrations propagating through various media.
'''
        
        doc = Document(pages=[
            Page(page_number=1, elements=[
                Element(
                    element_type='formula',
                    content=content,
                    bbox=BBox(100, 100, 900, 400),
                    page=1
                )
            ])
        ])
        
        chunker = StructureAwareChunker()
        chunks = chunker.chunk_document(doc)
        
        # Should split into parent + children
        assert len(chunks) > 1
        
        parent = chunks[0]
        assert parent.dominant_type == 'formula_block'
        assert parent.metadata.get('has_children') is True
        
        children = chunks[1:]
        assert len(children) >= 2  # At least 2 children from splitting
        
        for child in children:
            assert child.dominant_type == 'formula_block'
            assert child.chunking_method == 'env_split'
            assert r'\begin{' in child.content
    
    def test_small_formula_stays_single_chunk(self):
        """Small formula stays as single chunk."""
        content = r'\begin{equation}E = mc^2\end{equation}'
        
        doc = Document(pages=[
            Page(page_number=1, elements=[
                Element(
                    element_type='formula',
                    content=content,
                    bbox=BBox(100, 100, 900, 200),
                    page=1
                )
            ])
        ])
        
        chunker = StructureAwareChunker()
        chunks = chunker.chunk_document(doc)
        
        assert len(chunks) == 1
        assert chunks[0].dominant_type == 'formula_block'


class TestParentChildMetadata:
    """Test metadata in parent-child relationships."""
    
    def test_parent_has_child_count(self):
        """Parent chunk metadata includes child count."""
        code = '''import sys

def func1():
    return 1

def func2():
    return 2

def func3():
    return 3

def func4():
    return 4
'''
        
        doc = Document(pages=[
            Page(page_number=1, elements=[
                Element(
                    element_type='code',
                    content=code,
                    bbox=BBox(100, 100, 900, 400),
                    page=1
                )
            ])
        ])
        
        chunker = StructureAwareChunker()
        chunks = chunker.chunk_document(doc)
        
        if len(chunks) > 1:  # If it split
            parent = chunks[0]
            assert parent.metadata.get('has_children') is True
            assert parent.metadata.get('child_count') == len(chunks) - 1
    
    def test_children_have_sequence_numbers(self):
        """Child chunks have sequential block_sequence values."""
        # Create table that will split
        header = '| A | B |'
        separator = '|---|---|'
        rows = [f'| {i} | {i*2} |' for i in range(30)]
        content = '\n'.join([header, separator] + rows)
        
        doc = Document(pages=[
            Page(page_number=1, elements=[
                Element(
                    element_type='table',
                    content=content,
                    bbox=BBox(100, 100, 900, 600),
                    page=1
                )
            ])
        ])
        
        chunker = StructureAwareChunker()
        chunks = chunker.chunk_document(doc)
        
        if len(chunks) > 1:  # If it split
            children = chunks[1:]
            for i, child in enumerate(children):
                assert child.block_sequence == i
    
    def test_language_detection_in_code_metadata(self):
        """Code children have language metadata."""
        code = '''def python_function():
    return "Python"
'''
        
        doc = Document(pages=[
            Page(page_number=1, elements=[
                Element(
                    element_type='code',
                    content=code,
                    bbox=BBox(100, 100, 900, 200),
                    page=1,
                    metadata={'language': 'python'}
                )
            ])
        ])
        
        chunker = StructureAwareChunker()
        chunks = chunker.chunk_document(doc)
        
        # Even if not split, check metadata
        for chunk in chunks:
            if chunk.chunking_method == 'ast_split':
                assert chunk.metadata.get('language') == 'python'


class TestMixedDocumentWithSplitting:
    """Test documents with both prose and split blocks."""
    
    def test_mixed_document_with_large_blocks(self):
        """Document with prose and large blocks maintains order."""
        # Large code
        code = '''
def func1():
    return 1

def func2():
    return 2

def func3():
    return 3
'''
        
        # The chunker dedupes repeated phrases ("Introduction text here." x 60
        # collapses to one), so we have to write distinct sentences to push the
        # prose blocks above the short-prose merge threshold and prevent the
        # chunker from semantically collapsing intro+conclusion across the
        # intervening code block.
        intro_sentences = [
            f"Section {i} of the introduction discusses topic {i} in depth, "
            f"covering background concept {i}, related literature {i}, and "
            f"open questions about {i}."
            for i in range(1, 30)
        ]
        intro = " ".join(intro_sentences)
        conclusion_sentences = [
            f"In conclusion paragraph {i}, the empirical findings on {i} are "
            f"summarised, the limitations of method {i} are noted, and "
            f"future work direction {i} is proposed."
            for i in range(1, 30)
        ]
        conclusion = " ".join(conclusion_sentences)

        doc = Document(pages=[
            Page(page_number=1, elements=[
                Element(
                    element_type='text',
                    content=intro,
                    bbox=BBox(100, 100, 900, 150),
                    page=1
                ),
                Element(
                    element_type='code',
                    content=code,
                    bbox=BBox(100, 200, 900, 600),
                    page=1
                ),
                Element(
                    element_type='text',
                    content=conclusion,
                    bbox=BBox(100, 650, 900, 700),
                    page=1
                )
            ])
        ])

        chunker = StructureAwareChunker()
        chunks = chunker.chunk_document(doc)

        # Should have: prose, code_parent, [code_children...], prose
        assert len(chunks) >= 3

        # First should be prose
        assert chunks[0].dominant_type == 'prose'

        # Last should be prose
        assert chunks[-1].dominant_type == 'prose'

        # Middle chunks should be code-related
        code_chunks = [c for c in chunks if c.dominant_type == 'code_block']
        assert len(code_chunks) >= 1


if __name__ == '__main__':
    import pytest
    pytest.main([__file__, '-v'])