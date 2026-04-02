"""
Gene library utilities for accessing bundled gseapy gene sets
"""

from pathlib import Path
from typing import List

# Get the directory where gene libraries are stored
GENE_LIBRARY_DIR = Path(__file__).parent


def get_library_path(library_name: str) -> str:
    """
    Get the full path to a gene library file
    
    Args:
        library_name: Name of the library (e.g., 'GO_Biological_Process_2023')
                     Can be with or without .gmt extension
    
    Returns:
        Full path to the library file
    
    Raises:
        FileNotFoundError: If library doesn't exist
    """
    if not library_name.endswith('.gmt'):
        library_name += '.gmt'
    
    library_path = GENE_LIBRARY_DIR / library_name
    
    if not library_path.exists():
        raise FileNotFoundError(
            f"Gene library '{library_name}' not found. "
            f"Available libraries: {list_available_libraries()}"
        )
    
    return str(library_path)


def list_available_libraries() -> List[str]:
    """
    List all available gene libraries in the package
    
    Returns:
        List of library names (without .gmt extension)
    """
    return [
        f.stem for f in GENE_LIBRARY_DIR.glob('*.gmt')
    ]


__all__ = ["get_library_path", "list_available_libraries", "GENE_LIBRARY_DIR"]