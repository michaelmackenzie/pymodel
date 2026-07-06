import os
import ast

def extract_skeleton(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        try:
            node = ast.parse(f.read(), filename=file_path)
        except SyntaxError:
            return "  # [Syntax Error Parsing File]"
            
    lines = []
    for element in node.body:
        # Extract imports to show statistical package dependencies
        if isinstance(element, (ast.Import, ast.ImportFrom)):
            lines.append(ast.unparse(element))
        # Extract classes, methods, and functions
        elif isinstance(element, (ast.FunctionDef, ast.ClassDef)):
            lines.append(_parse_node(element))
    return "\n".join(lines)

def _parse_node(node, depth=0):
    indent = "    " * depth
    result = []
    
    if isinstance(node, ast.ClassDef):
        result.append(f"{indent}class {node.name}:")
        docstring = ast.get_docstring(node)
        if docstring:
            result.append(f'{indent}    """{docstring}"""')
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                result.append(_parse_node(item, depth + 1))
                
    elif isinstance(node, ast.FunctionDef):
        # Reconstruct function definition line with arguments
        args = ast.unparse(node.args)
        result.append(f"{indent}def {node.name}({args}):")
        docstring = ast.get_docstring(node)
        if docstring:
            result.append(f'{indent}    """{docstring}"""')
        result.append(f"{indent}    ... # [Logic Trimming]")
        
    return "\n".join(result)

def build_repo_map(root_dir):
    repo_map = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".py") and file != "generate_ai_context.py":
                rel_path = os.path.relpath(os.path.join(root, file), root_dir)
                repo_map.append(f"\n### FILE: {rel_path}\n" + "="*40)
                repo_map.append(extract_skeleton(os.path.join(root, file)))
    return "\n".join(repo_map)

if __name__ == "__main__":
    with open("ai_repo_skeleton.txt", "w", encoding="utf-8") as out:
        out.write(build_repo_map("."))
    print("Project skeleton mapped successfully into 'ai_repo_skeleton.txt'")
