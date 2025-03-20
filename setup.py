import os
import subprocess
import sys
import venv
import platform

def get_python_command():
    """Get the appropriate Python command based on the system."""
    if platform.system() == 'Windows':
        # Try different Python commands
        python_commands = ['python', 'py', 'python3']
        for cmd in python_commands:
            try:
                subprocess.run([cmd, '--version'], capture_output=True, check=True)
                return cmd
            except subprocess.CalledProcessError:
                continue
        raise RuntimeError("Python not found. Please install Python and try again.")
    return 'python3'

def create_venv():
    """Create a virtual environment if it doesn't exist."""
    venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv')
    
    if not os.path.exists(venv_path):
        print("Creating virtual environment...")
        python_cmd = get_python_command()
        subprocess.run([python_cmd, '-m', 'venv', venv_path], check=True)
        print(f"Virtual environment created at {venv_path}")
    else:
        print("Virtual environment already exists")
    
    return venv_path

def get_python_path():
    """Get the Python executable path based on the operating system."""
    if platform.system() == 'Windows':
        return os.path.join('venv', 'Scripts', 'python.exe')
    return os.path.join('venv', 'bin', 'python')

def install_requirements():
    """Install project requirements."""
    python_path = get_python_path()
    requirements_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'requirements.txt')
    
    print("Installing requirements...")
    try:
        subprocess.run([python_path, '-m', 'pip', 'install', '-r', requirements_path], check=True)
        print("Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        print("Please make sure you have internet connection and try again.")
        sys.exit(1)

def main():
    try:
        # Create virtual environment
        create_venv()
        
        # Install requirements
        install_requirements()
        
        print("\nSetup completed successfully!")
        print("\nTo activate the virtual environment:")
        if platform.system() == 'Windows':
            print("    .\\venv\\Scripts\\activate")
        else:
            print("    source ./venv/bin/activate")
        print("\nTo deactivate the virtual environment:")
        print("    deactivate")
        
    except Exception as e:
        print(f"\nError during setup: {e}")
        print("\nPlease make sure you have Python installed and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main() 