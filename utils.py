import json
def read_json_file(file_path):
    """读取JSON文件并返回其内容
    
    Args:
        file_path (str): JSON文件的路径
    
    Returns:
        dict or list: JSON文件的内容，如果出错则返回None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到")
        return None
    except json.JSONDecodeError as e:
        print(f"错误: 文件 '{file_path}' 不是有效的JSON格式")
        print(f"详细错误: {e}")
        return None
    except PermissionError:
        print(f"错误: 没有权限读取文件 '{file_path}'")
        return None
    except Exception as e:
        print(f"未知错误: {e}")
        return None
    
def save_json_file(file_path, file):
    """存储json文件并将file存储到file_path路径"""
    with open(file_path, 'w') as f:
        json.dump(file, f, indent=4)