import os
from pathlib import Path
import shutil

def rename_files_with_pattern(old_pattern="Apply halt: 2s", new_pattern="Apply_halt_2s", 
                             file_extensions=['.csv', '.pdf'], dry_run=True):
    """
    Rename files containing a specific pattern in their filename.
    
    Parameters:
    -----------
    data_directory : str or Path
        Root directory to search for files
    old_pattern : str
        Pattern to search for in filenames
    new_pattern : str
        Pattern to replace with
    file_extensions : list
        List of file extensions to process (e.g., ['.csv', '.pdf'])
    dry_run : bool
        If True, only shows what would be renamed without actually renaming
    
    Returns:
    --------
    dict : Summary of operations
    """
    
    data_path = Path("~/RANCZLAB-NAS/data/ONIX/20250409_Cohort3_rotation/Open_loop_day1").expanduser()
    
    if not data_path.exists():
        print(f"❌ Directory does not exist: {data_path}")
        return {'error': 'Directory not found'}
    
    results = {
        'found_files': [],
        'renamed_files': [],
        'errors': [],
        'total_found': 0,
        'total_renamed': 0
    }
    
    print(f"🔍 Searching in: {data_path}")
    print(f"📋 Looking for pattern: '{old_pattern}'")
    print(f"🔄 Will replace with: '{new_pattern}'")
    print(f"📁 File types: {file_extensions}")
    print(f"🧪 Dry run mode: {'ON' if dry_run else 'OFF'}")
    print("-" * 60)
    
    # Search for files recursively
    for file_path in data_path.rglob('*'):
        if file_path.is_file():
            # Check if file has the right extension
            if any(file_path.name.lower().endswith(ext.lower()) for ext in file_extensions):
                # Check if filename contains the pattern
                if old_pattern in file_path.name:
                    results['found_files'].append(str(file_path))
                    results['total_found'] += 1
                    
                    # Create new filename
                    new_filename = file_path.name.replace(old_pattern, new_pattern)
                    new_file_path = file_path.parent / new_filename
                    
                    print(f"📄 Found: {file_path.name}")
                    print(f"   ➡️  New: {new_filename}")
                    print(f"   📁 In: {file_path.parent}")
                    
                    if not dry_run:
                        try:
                            # Rename the file
                            file_path.rename(new_file_path)
                            results['renamed_files'].append({
                                'old_path': str(file_path),
                                'new_path': str(new_file_path),
                                'old_name': file_path.name,
                                'new_name': new_filename
                            })
                            results['total_renamed'] += 1
                            print(f"   ✅ Renamed successfully!")
                            
                        except Exception as e:
                            error_info = {
                                'file': str(file_path),
                                'error': str(e),
                                'operation': 'rename'
                            }
                            results['errors'].append(error_info)
                            print(f"   ❌ Error renaming: {str(e)}")
                    else:
                        print(f"   🧪 [DRY RUN] Would rename to: {new_filename}")
                    
                    print()  # Empty line for readability
    
    # Print summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total files found with pattern: {results['total_found']}")
    
    if dry_run:
        print(f"Files that WOULD be renamed: {results['total_found']}")
        print("\n🧪 This was a DRY RUN - no files were actually renamed.")
        print("   Set dry_run=False to perform actual renaming.")
    else:
        print(f"Files successfully renamed: {results['total_renamed']}")
        print(f"Errors encountered: {len(results['errors'])}")
        
        if results['errors']:
            print("\nErrors:")
            for error in results['errors']:
                print(f"  - {Path(error['file']).name}: {error['error']}")
    
    return results


def rename_files_in_multiple_directories(data_directories, old_pattern="Apply halt: 2s", 
                                       new_pattern="Apply_halt_2s", file_extensions=['.csv', '.pdf'], 
                                       dry_run=True):
    """
    Rename files in multiple directories.
    
    Parameters:
    -----------
    data_directories : list
        List of directory paths to process
    """
    
    all_results = {
        'directories_processed': 0,
        'total_found': 0,
        'total_renamed': 0,
        'total_errors': 0,
        'directory_results': []
    }
    
    for data_dir in data_directories:
        print(f"\n{'='*80}")
        print(f"PROCESSING DIRECTORY: {data_dir}")
        print(f"{'='*80}")
        
        results = rename_files_with_pattern(
            data_directory=data_dir,
            old_pattern=old_pattern,
            new_pattern=new_pattern,
            file_extensions=file_extensions,
            dry_run=dry_run
        )
        
        if 'error' not in results:
            all_results['directories_processed'] += 1
            all_results['total_found'] += results['total_found']
            all_results['total_renamed'] += results['total_renamed']
            all_results['total_errors'] += len(results['errors'])
            all_results['directory_results'].append({
                'directory': str(data_dir),
                'results': results
            })
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    print(f"Directories processed: {all_results['directories_processed']}")
    print(f"Total files found: {all_results['total_found']}")
    
    if dry_run:
        print(f"Total files that WOULD be renamed: {all_results['total_found']}")
    else:
        print(f"Total files renamed: {all_results['total_renamed']}")
        print(f"Total errors: {all_results['total_errors']}")
    
    return all_results


# Example usage:
if __name__ == "__main__":    
    # First, run in dry-run mode to see what would be changed
    print("🧪 RUNNING IN DRY-RUN MODE FIRST...")
    results = rename_files_with_pattern(
        old_pattern="Apply halt: 2s",
        new_pattern="Apply_halt_2s",
        file_extensions=['.csv', '.pdf'],
        dry_run=False  # Set to False to actually rename files
    )
