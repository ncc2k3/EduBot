import os
def validate_file_exists(file_path):
    """
    Kiểm tra file có tồn tại hay không.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} không tồn tại.")
    return True
