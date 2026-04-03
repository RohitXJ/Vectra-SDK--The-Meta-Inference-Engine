import os
import shutil
import tempfile
import re

# Roots for temporary data storage
BASE_TEMP_DIR = os.path.join(tempfile.gettempdir(), "vectra_sdk")
os.makedirs(BASE_TEMP_DIR, exist_ok=True)

def sanitize_name(name):
    """Sanitize class or file names for filesystem safety."""
    name = re.sub(r'[^a-zA-Z0-9_\-.]', '_', name)
    return name.strip()

def get_token_root(token):
    """Get the root temp folder for a specific token."""
    path = os.path.join(BASE_TEMP_DIR, sanitize_name(token))
    os.makedirs(path, exist_ok=True)
    return path

def get_session_paths(token):
    """Get support and query directory paths for a session."""
    root = get_token_root(token)
    support_path = os.path.join(root, "support")
    query_path = os.path.join(root, "query")
    export_path = os.path.join(root, "export")
    
    os.makedirs(support_path, exist_ok=True)
    os.makedirs(query_path, exist_ok=True)
    os.makedirs(export_path, exist_ok=True)
    
    return {
        "root": root,
        "support": support_path,
        "query": query_path,
        "export": export_path
    }

def save_upload_image(token, category, class_name, file_name, content):
    """
    Saves an uploaded image into the token-specific class folder.
    
    Args:
        token: User session token.
        category: "support" or "query".
        class_name: Name of the class the image belongs to.
        file_name: Original file name.
        content: Binary image content.
    """
    paths = get_session_paths(token)
    target_dir = os.path.join(paths[category], sanitize_name(class_name))
    os.makedirs(target_dir, exist_ok=True)
    
    safe_file_name = sanitize_name(file_name)
    save_path = os.path.join(target_dir, safe_file_name)
    
    with open(save_path, "wb") as f:
        f.write(content)
    return save_path

def clear_session_data(token):
    """Clears all temporary data associated with a token."""
    root = get_token_root(token)
    if os.path.exists(root):
        shutil.rmtree(root)
        return True
    return False

def reset_category_data(token, category):
    """Clears either support or query data for a token."""
    paths = get_session_paths(token)
    if os.path.exists(paths[category]):
        shutil.rmtree(paths[category])
        os.makedirs(paths[category], exist_ok=True)
