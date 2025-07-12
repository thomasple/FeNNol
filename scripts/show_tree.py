#!/usr/bin/env python3
"""
Directory Tree Structure Display Script

This script displays the directory tree structure of a given path
in a clean, hierarchical format similar to the Unix 'tree' command.
"""

import os
import sys
import argparse
import tempfile
import shutil
import subprocess
from pathlib import Path


def should_ignore(path, ignore_patterns=None, include_patterns=None):
    """Check if a path should be ignored based on ignore and include patterns."""
    if ignore_patterns is None:
        # Default patterns when no specific patterns are provided
        ignore_patterns = {
            '__pycache__', '.git', '.vscode', '.idea', 
            'node_modules', '.DS_Store', '*.pyc', '.pytest_cache',
            '.mypy_cache', '.ruff_cache', 'venv', '.venv', 'env'
        }
    
    name = path.name
    
    # Check if file matches any include patterns (these override ignore patterns)
    if include_patterns:
        for pattern in include_patterns:
            if matches_pattern(name, pattern):
                return False  # Always include files that match include patterns
    
    # Check for ignore patterns
    for pattern in ignore_patterns:
        if matches_pattern(name, pattern):
            return True
    
    return False


def matches_pattern(name, pattern):
    """Check if a name matches a glob-style pattern."""
    # Handle exact matches
    if pattern == name:
        return True
    
    # Handle wildcard patterns
    if '*' in pattern:
        if pattern.startswith('*') and pattern.endswith('*'):
            # *pattern* - contains
            return pattern[1:-1] in name
        elif pattern.startswith('*'):
            # *pattern - ends with
            return name.endswith(pattern[1:])
        elif pattern.endswith('*'):
            # pattern* - starts with
            return name.startswith(pattern[:-1])
        else:
            # More complex patterns - use fnmatch for full glob support
            import fnmatch
            return fnmatch.fnmatch(name, pattern)
    
    return False


def get_tree_structure(root_path, max_depth=None, show_hidden=False, ignore_patterns=None, include_patterns=None):
    """
    Generate tree structure for a given directory.
    
    Args:
        root_path: Path to the root directory
        max_depth: Maximum depth to traverse (None for unlimited)
        show_hidden: Whether to show hidden files/directories
        ignore_patterns: Set of patterns to ignore
        include_patterns: Set of patterns to always include (overrides ignore patterns)
    
    Returns:
        List of tuples (depth, name, is_directory, full_path)
    """
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(f"Path '{root_path}' does not exist")

    if not root.is_dir():
        raise NotADirectoryError(f"Path '{root_path}' is not a directory")

    tree_items = []
    
    # Add the root directory as the first item
    root_name = root.name if root.name else str(root)
    tree_items.append((0, root_name, True, str(root)))

    def traverse(path, depth=1):  # Start at depth 1 since root is at depth 0
        if max_depth is not None and depth > max_depth:
            return

        try:
            # Get all items in the directory
            items = list(path.iterdir())

            # Filter items
            filtered_items = []
            for item in items:
                # Skip hidden files unless requested
                if not show_hidden and item.name.startswith('.'):
                    continue

                # Skip ignored patterns
                if should_ignore(item, ignore_patterns, include_patterns):
                    continue

                filtered_items.append(item)

            # Sort items: directories first, then files, both alphabetically
            filtered_items.sort(key=lambda x: (not x.is_dir(), x.name.lower()))

            for item in filtered_items:
                tree_items.append((depth, item.name, item.is_dir(), str(item)))

                if item.is_dir():
                    traverse(item, depth + 1)

        except PermissionError:
            # Skip directories we can't read
            pass

    traverse(root)
    return tree_items


def format_tree(tree_items, use_unicode=True):
    """
    Format tree items into a visual tree structure.
    
    Args:
        tree_items: List of tuples (depth, name, is_directory, full_path)
        use_unicode: Whether to use Unicode box-drawing characters
    
    Returns:
        Formatted string representation of the tree
    """
    if use_unicode:
        # Unicode box-drawing characters
        branch = "├── "
        last_branch = "└── "
        vertical = "│   "
        space = "    "
    else:
        # ASCII characters
        branch = "|-- "
        last_branch = "`-- "
        vertical = "|   "
        space = "    "
    
    if not tree_items:
        return ""
    
    lines = []
    depth_tracker = {}  # Track which depths still have siblings
    
    for i, (depth, name, is_dir, full_path) in enumerate(tree_items):
        # Determine if this is the last item at this depth
        is_last_at_depth = True
        for j in range(i + 1, len(tree_items)):
            next_depth, _, _, _ = tree_items[j]
            if next_depth == depth:
                is_last_at_depth = False
                break
            elif next_depth < depth:
                break
        
        # Build prefix for this line
        prefix_parts = []
        for d in range(depth):
            if d in depth_tracker and depth_tracker[d]:
                prefix_parts.append(vertical)
            else:
                prefix_parts.append(space)
        
        # Add the branch character
        if depth == 0:
            prefix = ""
        else:
            if is_last_at_depth:
                prefix = "".join(prefix_parts) + last_branch
                depth_tracker[depth] = False
            else:
                prefix = "".join(prefix_parts) + branch
                depth_tracker[depth] = True
        
        # Add directory indicator
        if is_dir:
            display_name = f"{name}/"
        else:
            display_name = name
        
        lines.append(f"{prefix}{display_name}")
    
    return "\n".join(lines)


def display_file_contents(tree_items, root_path, max_file_size=1024*1024, hidden_patterns=None):  # 1MB default limit
    """
    Display the contents of all files in the tree with descriptive headers.
    
    Args:
        tree_items: List of tuples (depth, name, is_directory, full_path)
        root_path: Root directory path for calculating relative paths
        max_file_size: Maximum file size to display (in bytes)
        hidden_patterns: Set of patterns for files whose content should not be displayed
    """
    files_to_show = [item for item in tree_items if not item[2]]  # Only files, not directories
    
    if not files_to_show:
        print("\nNo files to display.")
        return
    
    print(f"\n{'='*60}")
    print("FILE CONTENTS")
    print(f"{'='*60}")
    
    for depth, name, is_dir, full_path in files_to_show:
        file_path = Path(full_path)
        
        # Calculate relative path
        try:
            relative_path = file_path.relative_to(root_path)
        except ValueError:
            # Fallback to full path if relative calculation fails
            relative_path = file_path
        
        # Create header
        print(f"\n{'-'*50}")
        print(f"FILE: {name}")
        print(f"PATH: {relative_path}")
        
        # Check if content should be hidden for this file
        should_hide_content = False
        if hidden_patterns:
            for pattern in hidden_patterns:
                if matches_pattern(name, pattern):
                    should_hide_content = True
                    break
        
        if should_hide_content:
            try:
                file_size = file_path.stat().st_size
                print(f"SIZE: {file_size:,} bytes")
            except:
                print(f"SIZE: (Error accessing file)")
            print(f"{'-'*50}")
            print("(File content hidden)")
            continue
        
        try:
            # First, check if file is likely binary by examining extension and content
            if is_likely_binary_file(file_path):
                file_size = file_path.stat().st_size
                print(f"SIZE: {file_size:,} bytes")
                print(f"{'-'*50}")
                print("(Binary file - content not displayable as text)")
                continue
            
            # Check file size for text files
            file_size = file_path.stat().st_size
            if file_size > max_file_size:
                print(f"SIZE: {file_size:,} bytes (too large to display, limit: {max_file_size:,} bytes)")
                print(f"{'-'*50}")
                print("(File too large to display)")
                continue
            elif file_size == 0:
                print(f"SIZE: {file_size} bytes")
                print(f"{'-'*50}")
                print("(Empty file)")
                continue
            else:
                print(f"SIZE: {file_size:,} bytes")
            
            print(f"{'-'*50}")
            
            # Try to read and display file content
            try:
                
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    if content.strip():
                        print(content)
                    else:
                        print("(File appears to be empty or contains only whitespace)")
                        
            except UnicodeDecodeError:
                print("(Binary file - content not displayable as text)")
            
        except PermissionError:
            print(f"SIZE: (Permission denied)")
            print(f"{'-'*50}")
            print("(Permission denied - cannot read file)")
        except Exception as e:
            print(f"SIZE: (Error accessing file)")
            print(f"{'-'*50}")
            print(f"(Error reading file: {e})")


def is_likely_binary_file(file_path):
    """
    Check if a file is likely to be binary based on extension and content sampling.
    
    Args:
        file_path: Path object to the file
    
    Returns:
        bool: True if file is likely binary, False otherwise
    """
    # Common binary file extensions
    binary_extensions = {
        # Images
        '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp', '.ico',
        '.svg', '.psd', '.ai', '.eps',
        # Videos
        '.mp4', '.avi', '.mov', '.wmv', '.flv', '.webm', '.mkv', '.m4v',
        # Audio
        '.mp3', '.wav', '.flac', '.aac', '.ogg', '.wma', '.m4a',
        # Archives
        '.zip', '.rar', '.7z', '.tar', '.gz', '.bz2', '.xz', '.lz', '.lzma',
        # Executables
        '.exe', '.dll', '.so', '.dylib', '.app', '.deb', '.rpm', '.msi',
        # Documents
        '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx', '.odt', '.ods', '.odp',
        # Databases
        '.db', '.sqlite', '.sqlite3', '.mdb',
        # Fonts
        '.ttf', '.otf', '.woff', '.woff2', '.eot',
        # Other binary formats
        '.bin', '.dat', '.pickle', '.pkl', '.npy', '.npz', '.h5', '.hdf5',
        '.pyc', '.pyo', '.class', '.jar', '.war',
    }
    
    # Check file extension
    if file_path.suffix.lower() in binary_extensions:
        return True
    
    # For files without clear extensions or unknown extensions,
    # sample the beginning of the file to check for binary content
    try:
        with open(file_path, 'rb') as f:
            # Read first 8192 bytes to check for binary content
            chunk = f.read(8192)
            if not chunk:
                return False
            
            # Check for null bytes (common in binary files)
            if b'\x00' in chunk:
                return True
            
            # Check for high percentage of non-printable characters
            printable_chars = sum(1 for byte in chunk if 32 <= byte <= 126 or byte in (9, 10, 13))
            total_chars = len(chunk)
            
            if total_chars > 0:
                printable_ratio = printable_chars / total_chars
                # If less than 70% of characters are printable, consider it binary
                if printable_ratio < 0.7:
                    return True
                    
    except (OSError, IOError):
        # If we can't read the file, assume it might be binary
        return True
    
    return False


def is_git_url(path_str):
    """Check if a string appears to be a git repository URL."""
    git_patterns = [
        'https://github.com/',
        'https://gitlab.com/',
        'https://bitbucket.org/',
        'git@github.com:',
        'git@gitlab.com:',
        'git@bitbucket.org:',
        '.git'  # Any URL ending with .git
    ]
    
    # Check for common git hosting patterns
    for pattern in git_patterns:
        if pattern in path_str.lower():
            return True
    
    # Check for SSH git URLs (user@host:path format)
    if '@' in path_str and ':' in path_str and not path_str.startswith('http'):
        return True
    
    return False


def clone_git_repository(git_url, temp_dir=None):
    """
    Clone a git repository to a temporary directory.
    
    Args:
        git_url: The git repository URL
        temp_dir: Optional temporary directory to clone into
    
    Returns:
        Path to the cloned repository directory
    
    Raises:
        subprocess.CalledProcessError: If git clone fails
        FileNotFoundError: If git is not installed
    """
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp(prefix="show_tree_git_")
    
    try:
        # Extract repository name from URL for the directory name
        repo_name = git_url.rstrip('/').split('/')[-1]
        if repo_name.endswith('.git'):
            repo_name = repo_name[:-4]
        
        clone_path = Path(temp_dir) / repo_name
        
        print(f"Cloning repository: {git_url}", file=sys.stderr)
        print(f"Destination: {clone_path}", file=sys.stderr)
        
        # Run git clone
        result = subprocess.run(
            ['git', 'clone', '--depth', '1', git_url, str(clone_path)],
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(
                result.returncode, 
                ['git', 'clone', git_url],
                output=result.stdout,
                stderr=result.stderr
            )
        
        print("Repository cloned successfully!", file=sys.stderr)
        return clone_path
        
    except FileNotFoundError:
        raise FileNotFoundError(
            "Git is not installed or not found in PATH. "
            "Please install git to clone repositories."
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Git clone timed out after 5 minutes. "
            f"The repository {git_url} might be too large or the connection is slow."
        )
    except Exception as e:
        print(f"Error cloning repository: {e}", file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Display directory tree structure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python show_tree.py                    # Show tree for current directory
  python show_tree.py /path/to/dir       # Show tree for specific directory
  python show_tree.py https://github.com/user/repo.git  # Clone and analyze git repository
  python show_tree.py git@github.com:user/repo.git      # Clone via SSH and analyze
  python show_tree.py -d 3               # Limit depth to 3 levels
  python show_tree.py -a                 # Show hidden files
  python show_tree.py --ascii            # Use ASCII characters instead of Unicode
  python show_tree.py -c                 # Show file contents after tree
  python show_tree.py --ignore "*.log"   # Ignore all .log files
  python show_tree.py --ignore "temp*" --ignore "*.tmp"  # Multiple ignore patterns
  python show_tree.py --include "*.py" # Always show Python files, even if they match ignore patterns
  python show_tree.py --include "*.py" --ignore "test*"  # Show Python files, ignore test files
  python show_tree.py -c --hidden "*.min.js"  # Show content but hide minified JS files
  python show_tree.py -c --hidden "*.log" --hidden "*.cache"  # Hide multiple file types
        """
    )
    
    parser.add_argument(
        "path", 
        nargs="?", 
        default=".", 
        help="Path to display or git repository URL to clone and analyze (default: current directory)"
    )
    
    parser.add_argument(
        "-d", "--depth", 
        type=int, 
        help="Maximum depth to traverse"
    )
    
    parser.add_argument(
        "-a", "--all", 
        action="store_true", 
        help="Show hidden files and directories"
    )
    
    parser.add_argument(
        "--ascii", 
        action="store_true", 
        help="Use ASCII characters instead of Unicode"
    )
    
    parser.add_argument(
        "--no-ignore", 
        action="store_true", 
        help="Don't ignore common files/directories like __pycache__, .git, etc."
    )
    
    parser.add_argument(
        "-c", "--show-content", 
        action="store_true", 
        help="Show the content of all files after the tree structure"
    )
    
    parser.add_argument(
        "--ignore", 
        action="append",
        metavar="PATTERN",
        help="Additional patterns to ignore (can be used multiple times). Supports wildcards like *.log"
    )
    
    parser.add_argument(
        "--include", 
        action="append",
        metavar="PATTERN",
        help="Patterns to always include (can be used multiple times). Files matching these patterns are shown even if they match ignore patterns. Supports wildcards like *.py"
    )
    
    parser.add_argument(
        "--hide", 
        action="append",
        metavar="PATTERN",
        help="Patterns for files whose content should not be displayed when using --show-content (can be used multiple times). Supports wildcards like *.min.js"
    )
    
    args = parser.parse_args()
    
    try:
        # Check if path is a git repository URL
        if is_git_url(args.path):
            # Clone the repository to a temporary directory
            temp_dir = None
            cloned_path = None
            try:
                cloned_path = clone_git_repository(args.path)
                target_path = cloned_path
                print(file=sys.stderr)  # Add blank line after clone messages
            except (subprocess.CalledProcessError, FileNotFoundError, RuntimeError) as e:
                print(f"Error cloning repository: {e}", file=sys.stderr)
                return 1
        else:
            # Resolve the local path
            target_path = Path(args.path).resolve()
            cloned_path = None
        
        # Set ignore patterns
        if args.no_ignore:
            # Only use user-specified patterns if --no-ignore is used
            ignore_patterns = set(args.ignore) if args.ignore else set()
        else:
            # Use default patterns and add user-specified ones
            ignore_patterns = {
                '__pycache__', '.git', '.vscode', '.idea', 
                'node_modules', '.DS_Store', '*.pyc', '.pytest_cache',
                '.mypy_cache', '.ruff_cache', 'venv', '.venv', 'env'
            }
            if args.ignore:
                ignore_patterns.update(args.ignore)
        
        # Set include patterns
        include_patterns = set(args.include) if args.include else None
        
        # Set hidden content patterns
        hidden_patterns = set(args.hide) if args.hide else None
        
        # Generate tree structure
        tree_items = get_tree_structure(
            target_path, 
            max_depth=args.depth,
            show_hidden=args.all,
            ignore_patterns=ignore_patterns,
            include_patterns=include_patterns
        )
        
        # Format and display
        if not tree_items:
            print("(empty directory)")
        else:
            tree_output = format_tree(tree_items, use_unicode=not args.ascii)
            print(tree_output)
            
            # Show summary
            dirs = sum(1 for _, _, is_dir, _ in tree_items if is_dir)
            files = sum(1 for _, _, is_dir, _ in tree_items if not is_dir)
            print(f"\n{dirs} directories, {files} files")
            
            # Show file contents if requested
            if args.show_content:
                display_file_contents(tree_items, target_path, hidden_patterns=hidden_patterns)
    
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        return 1
    
    finally:
        # Clean up temporary directory if we cloned a repository
        if 'cloned_path' in locals() and cloned_path and cloned_path.exists():
            try:
                shutil.rmtree(cloned_path.parent)
                print(f"\nCleaned up temporary directory: {cloned_path.parent}", file=sys.stderr)
            except Exception as e:
                print(f"\nWarning: Could not clean up temporary directory: {e}", file=sys.stderr)
    
    return 0


if __name__ == "__main__":
    exit(main())
