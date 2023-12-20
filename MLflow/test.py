import ast
import pkg_resources

def get_imports(path):
    """
    Parses a Python file and returns a set of imported modules.
    """
    with open(path, 'r') as file:
        tree = ast.parse(file.read())

        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                imports.add(node.module if node.module else '')

        return imports

def get_module_versions(modules):
    """
    Takes a set of module names and returns a dictionary with their versions.
    """
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    return {module: installed_packages.get(module) for module in modules}

# Replace 'your_script.py' with the path to your Python script
imports = get_imports('/Users/hirad/Chatbot_Cognitivo-2/MLflow.py')
versions = get_module_versions(imports)

for module, version in versions.items():
    print(f"{module}: {version if version else 'Version not found'}")
