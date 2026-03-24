from langchain_core.tools import tool
import subprocess
import sys
import io
import traceback


@tool
def execute_python(code: str) -> str:
    """Execute Python code and return the output. Use for calculations, data processing, and analysis."""
    try:
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        # Execute code in a restricted namespace
        namespace = {}
        exec(code, namespace)

        output = sys.stdout.getvalue()
        sys.stdout = old_stdout

        return f"✅ Code executed successfully!\n\nOutput:\n{output}" if output else "✅ Code executed successfully! (No output)"
    except Exception as e:
        sys.stdout = old_stdout
        return f"❌ Execution Error:\n{traceback.format_exc()}"


@tool
def write_code(task_description: str, language: str = "python") -> str:
    """Generate clean, well-commented code for a given task description."""
    # This tool provides a structured prompt for code generation
    return f"""
Code Writing Task:
- Language: {language}
- Task: {task_description}

Please provide complete, production-ready code with:
1. Clear comments explaining logic
2. Error handling
3. Example usage
4. Any required imports
"""


@tool
def debug_code(code: str, error_message: str) -> str:
    """Analyze code and an error message to suggest fixes."""
    return f"""
Debug Request:
Code:
```
{code}
```
Error:
{error_message}

Please analyze and provide:
1. Root cause of the error
2. Fixed code
3. Explanation of the fix
"""


@tool
def run_bash_command(command: str) -> str:
    """Run a safe bash command (read-only operations like ls, pwd, echo, cat)."""
    # Whitelist safe commands only
    safe_commands = ["ls", "pwd", "echo", "cat", "head", "tail", "wc", "find", "grep", "date", "whoami"]
    cmd_name = command.strip().split()[0]

    if cmd_name not in safe_commands:
        return f"⚠️ Command '{cmd_name}' is not allowed for safety reasons. Allowed: {', '.join(safe_commands)}"

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=10
        )
        output = result.stdout or result.stderr
        return f"Command: `{command}`\n\nOutput:\n{output}"
    except subprocess.TimeoutExpired:
        return "Command timed out after 10 seconds."
    except Exception as e:
        return f"Error: {str(e)}"


def get_code_tools():
    return [execute_python, write_code, debug_code, run_bash_command]
