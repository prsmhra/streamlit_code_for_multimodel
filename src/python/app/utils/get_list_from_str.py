import ast
from src.python.app.constants.constants import Constants
def get_list_from_str(input_str):
    """
    Converts a comma-separated string into a list of trimmed strings.
    
    Args:
        input_str (str): Comma-separated string.
    
    Returns: list
    """
    # Find the first occurrence of '[[' or '['
    start_index = input_str.find('[[')
    if start_index == -Constants.ONE:
        start_index = input_str.find('[')
    # Find the last occurrence of ']]' or ']'
    end_index = input_str.rfind(']]')
    if end_index == -Constants.ONE:
        end_index = input_str.rfind(']')
    if start_index == -Constants.ONE or end_index == -Constants.ONE:
        print("Brackets not found in the string.")
        return None

    end_index += 2 if input_str[end_index:end_index+2] == ']]' else Constants.ONE
    list_string = input_str[start_index:end_index]

    try:
        result_list = ast.literal_eval(list_string)
        return result_list
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing the string: {e}")
        return None
