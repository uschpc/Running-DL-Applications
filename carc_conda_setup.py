#!/usr/bin/env python3
"""
CARC Conda Environment Setup Script
Based on CARC documentation: https://www.carc.usc.edu/user-guides/hpc-systems/software/conda
"""

import subprocess
import sys
import os
import platform
import shutil
import argparse
from pathlib import Path
import time
import getpass
import json

class CARCCondaSetup:
    def __init__(self):
        self.current_step = 0
        self.total_steps = 6
        self.status = {}
        self.is_first_time = True
        self.env_name = "torch-env"
        self.python_version = "3.12"  # Using 3.12 
        self.home_dir = os.path.expanduser("~")
        self.config_file = os.path.join(self.home_dir, ".carc_conda_config.json")

    def print_status(self, message, status="info"):
        """Print status message with color indicators."""
        if status == "success":
            print(f"✓ {message}")
        elif status == "error":
            print(f"✗ {message}")
        elif status == "warning":
            print(f"! {message}")
        elif status == "info":
            print(f"  {message}")
        elif status == "highlight":
            print(f"→ {message}")

    def get_user_input(self, prompt, default=None, required=False):
        """Get user input with a prompt."""
        while True:
            print(f"{prompt} ", end='')
            if default:
                print(f"[{default}] ", end='')
            user_input = input().strip()
            
            if not user_input and default:
                return default
            elif not user_input and required:
                self.print_status("This field is required. Please try again.", "warning")
                continue
            return user_input

    def check_bash_configs(self):
        """Check if user's bash configuration files exist and copy from /etc/skel if they don't."""
        home_dir = os.path.expanduser("~")
        bash_profile = os.path.join(home_dir, ".bash_profile")
        bashrc = os.path.join(home_dir, ".bashrc")
        
        print("\nChecking bash configuration files...")
        
        # Check .bash_profile
        print(f"\nChecking .bash_profile at: {bash_profile}")
        if os.path.exists(bash_profile):
            print("✓ .bash_profile exists")
        else:
            print("✗ .bash_profile not found")
            print("Creating .bash_profile...")
            try:
                shutil.copy("/etc/skel/.bash_profile", bash_profile)
                print("✓ Successfully created .bash_profile")
            except Exception as e:
                print(f"✗ Error creating .bash_profile: {e}")
                return False
        
        # Check .bashrc
        print(f"\nChecking .bashrc at: {bashrc}")
        if os.path.exists(bashrc):
            print("✓ .bashrc exists")
        else:
            print("✗ .bashrc not found")
            print("Creating .bashrc...")
            try:
                shutil.copy("/etc/skel/.bashrc", bashrc)
                print("✓ Successfully created .bashrc")
            except Exception as e:
                print(f"✗ Error creating .bashrc: {e}")
                return False
        
        print("\n✓ Bash configuration check completed successfully")
        return True

    def load_conda_module(self):
        """Load conda module as per CARC documentation."""
        print("\nStep 1: Loading conda module...")
        try:
            # Module purge and load conda as per CARC docs
            module_cmd = "module purge && module load conda"
            subprocess.run(module_cmd, shell=True, check=True, executable='/bin/bash')
            self.print_status("Successfully loaded conda module", "success")
            return True
        except subprocess.CalledProcessError as e:
            self.print_status(f"Error loading conda module: {e}", "error")
            return False

    def initialize_conda(self):
        """Initialize conda in bash shell as per CARC documentation."""
        print("\nStep 2: Initializing conda...")
        try:
            # Initialize conda in bash as per CARC docs
            init_cmd = "conda init bash"
            subprocess.run(init_cmd, shell=True, check=True, executable='/bin/bash')
            
            # Source bashrc to make conda available
            source_cmd = "source ~/.bashrc"
            subprocess.run(source_cmd, shell=True, check=True, executable='/bin/bash')
            
            self.print_status("Successfully initialized conda", "success")
            return True
        except subprocess.CalledProcessError as e:
            self.print_status(f"Error initializing conda: {e}", "error")
            return False

    def create_conda_environment(self):
        """Create conda environment as per CARC documentation."""
        print(f"\nStep 3: Creating conda environment '{self.env_name}'...")
        try:
            # Create environment with Python as per CARC docs
            create_cmd = f"conda create --name {self.env_name} python=3.12 -y"
            subprocess.run(create_cmd, shell=True, check=True, executable='/bin/bash')
            self.print_status(f"Successfully created environment '{self.env_name}'", "success")
            return True
        except subprocess.CalledProcessError as e:
            self.print_status(f"Error creating conda environment: {e}", "error")
            return False

    def install_pytorch(self):
        """Install PyTorch as per CARC PyTorch installation guide."""
        print(f"\nStep 4: Installing PyTorch in environment '{self.env_name}'...")
        try:
            # Initialize conda shell hook
            init_cmd = "eval \"$(conda shell.bash hook)\""
            activate_cmd = f"conda activate {self.env_name}"
            
            # Install PyTorch using pip as recommended
            install_cmd = f"{init_cmd} && {activate_cmd} && pip install torch torchvision torchaudio"
            subprocess.run(install_cmd, shell=True, check=True, executable='/bin/bash')
            
            self.print_status("Successfully installed PyTorch", "success")
            return True
        except subprocess.CalledProcessError as e:
            self.print_status(f"Error installing PyTorch: {e}", "error")
            return False

    def install_additional_packages(self):
        """Install additional data science packages."""
        print(f"\nStep 5: Installing additional packages in environment '{self.env_name}'...")
        packages = ["numpy", "pandas", "scikit-learn", "matplotlib", "line_profiler","tensorboard"]
        
        try:
            init_cmd = "eval \"$(conda shell.bash hook)\""
            activate_cmd = f"conda activate {self.env_name}"
            
            for package in packages:
                print(f"Installing {package}...")
                install_cmd = f"{init_cmd} && {activate_cmd} && conda install -y {package}"
                subprocess.run(install_cmd, shell=True, check=True, executable='/bin/bash')
                self.print_status(f"Successfully installed {package}", "success")
            
            return True
        except subprocess.CalledProcessError as e:
            self.print_status(f"Error installing packages: {e}", "error")
            return False

    def install_jupyter_kernel(self):
        """Install Jupyter kernel for the conda environment."""
        print(f"\nStep 6: Installing Jupyter kernel for '{self.env_name}'...")
        try:
            init_cmd = "eval \"$(conda shell.bash hook)\""
            activate_cmd = f"conda activate {self.env_name}"
            
            # Install ipykernel
            install_cmd = f"{init_cmd} && {activate_cmd} && conda install -c conda-forge ipykernel -y"
            subprocess.run(install_cmd, shell=True, check=True, executable='/bin/bash')
            
            # Install the kernel
            kernel_cmd = f"{init_cmd} && {activate_cmd} && python -m ipykernel install --user --name {self.env_name} --display-name \"{self.env_name}\""
            subprocess.run(kernel_cmd, shell=True, check=True, executable='/bin/bash')
            
            self.print_status(f"Successfully installed Jupyter kernel for '{self.env_name}'", "success")
            return True
        except subprocess.CalledProcessError as e:
            self.print_status(f"Error installing Jupyter kernel: {e}", "error")
            return False

    def save_config(self):
        """Save configuration to mark setup as complete."""
        config = {
            "setup_completed": True,
            "env_name": self.env_name,
            "python_version": "3.12",
            "setup_date": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            self.print_status("Configuration saved", "success")
        except Exception as e:
            self.print_status(f"Error saving configuration: {e}", "warning")

    def show_usage_instructions(self):
        """Show instructions for using the conda environment."""
        print("\n" + "="*60)
        print("CARC CONDA SETUP COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        print(f"\nYour conda environment '{self.env_name}' is ready to use!")
        
        print("\nTo activate your environment:")
        print(f"  conda activate {self.env_name}")
        
        print("\nTo test PyTorch installation:")
        print(f"  conda activate {self.env_name}")
        print("  python -c 'import torch; print(torch.__version__)'")
        

        
        print("\nTo clean up conda cache and free up space:")
        print("  conda clean --all")
        
        print("\nFor more information:")
        print("  https://www.carc.usc.edu/user-guides/hpc-systems/software/conda")
        print("  https://www.carc.usc.edu/user-guides/data-science/pytorch-installation")
        
        print("\n" + "="*60)

    def run_interactive_setup(self):
        """Run the interactive setup process."""
        print("Welcome to CARC Conda Environment Setup!")
        print("This script will set up conda and install PyTorch according to CARC documentation.\n")
        
        # Get user preferences
        self.env_name = self.get_user_input("Enter environment name", self.env_name)
        
        print(f"\nSetup configuration:")
        print(f"  Environment name: {self.env_name}")
        print(f"  Python version: 3.12")
        
        proceed = self.get_user_input("Proceed with setup? (y/n)", "y")
        if proceed.lower() not in ['y', 'yes']:
            print("Setup cancelled.")
            return False

        # Run setup steps
        steps = [
            ("Checking bash configuration files", self.check_bash_configs),
            ("Loading conda module", self.load_conda_module),
            ("Initializing conda", self.initialize_conda),
            ("Creating conda environment", self.create_conda_environment),
            ("Installing PyTorch", self.install_pytorch),
            ("Installing additional packages", self.install_additional_packages),
            ("Installing Jupyter kernel", self.install_jupyter_kernel)
        ]

        for step_name, step_func in steps:
            print(f"\n{step_name}...")
            if not step_func():
                self.print_status(f"Setup failed at: {step_name}", "error")
                return False

        # Save configuration and show instructions
        self.save_config()
        self.show_usage_instructions()
        return True

    def run_automated_setup(self):
        """Run the automated setup process with default values."""
        print("Running automated CARC conda setup...")
        
        # Run setup steps
        steps = [
            ("Checking bash configuration files", self.check_bash_configs),
            ("Loading conda module", self.load_conda_module),
            ("Initializing conda", self.initialize_conda),
            ("Creating conda environment", self.create_conda_environment),
            ("Installing PyTorch", self.install_pytorch),
            ("Installing additional packages", self.install_additional_packages),
            ("Installing Jupyter kernel", self.install_jupyter_kernel)
        ]

        for step_name, step_func in steps:
            print(f"\n{step_name}...")
            if not step_func():
                self.print_status(f"Setup failed at: {step_name}", "error")
                return False

        # Save configuration and show instructions
        self.save_config()
        self.show_usage_instructions()
        return True

def main():
    parser = argparse.ArgumentParser(
        description='CARC Conda Environment Setup Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python carc_conda_setup_improved.py                    # Automated setup with defaults
  python carc_conda_setup_improved.py --interactive     # Interactive setup
  python carc_conda_setup_improved.py --env-name myenv  # Custom environment name
        """
    )
    
    parser.add_argument('--env-name', 
                       help='Name of the conda environment (default: pytorch_env)')
    parser.add_argument('--interactive', action='store_true', 
                       help='Run in interactive mode (requires user input)')
    
    args = parser.parse_args()

    setup = CARCCondaSetup()
    
    # Override defaults with command line arguments
    if args.env_name:
        setup.env_name = args.env_name

    try:
        if args.interactive:
            success = setup.run_interactive_setup()
        else:
            success = setup.run_automated_setup()
        
        if not success:
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
