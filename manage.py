#!/usr/bin/env python3
"""
manage.py - Command Line Interface for Project Bard
"""
import sys
import argparse
import subprocess
import os

# Prevent CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def run_script(script_name):
    print(f"\n[*] Running {script_name}...")
    try:
        subprocess.run([sys.executable, script_name], check=True)
    except subprocess.CalledProcessError:
        print(f"\n[!] Error running {script_name}.")
    except KeyboardInterrupt:
        print("\n[!] Stopped by user.")

def interactive_menu():
    while True:
        print("\n" + "="*50)
        print("[Project Bard - Main Menu]")
        print("="*50)
        print("1. Run Data Pipeline (Download & Clean)")
        print("2. Train Tokenizer (Build Vocab)")
        print("3. Train Model (Pre-Training, Start/Resume)")
        print("4. Fine-Tune Model (SFT + DPO)")
        print("5. Start API Server (FastAPI)")
        print("6. Launch Interactive Chat (CLI)")
        print("7. Run Model Evaluation")
        print("8. Exit")
        
        choice = input("\nEnter your choice (1-8): ").strip()
        
        if choice == '1':
            run_script("data_pipeline.py")
        elif choice == '2':
            run_script("tokenizer.py")
        elif choice == '3':
            run_script("train.py")
        elif choice == '4':
            run_script("sft.py")
        elif choice == '5':
            print("\n[*] Starting API Server on http://0.0.0.0:8000")
            try:
                subprocess.run([sys.executable, "api_server.py"])
            except KeyboardInterrupt:
                pass
        elif choice == '6':
            run_script("chat.py")
        elif choice == '7':
            run_script("evaluate.py")
        elif choice == '8':
            print("Goodbye!")
            break
        else:
            print("Invalid choice, please try again.")

def main():
    parser = argparse.ArgumentParser(description="Project Bard CLI")
    parser.add_argument('command', nargs='?', choices=['data', 'tokenize', 'train', 'sft', 'serve', 'chat', 'eval', 'menu'], 
                        help="Command to run (leave blank for interactive menu)")
    args = parser.parse_args()

    if not args.command or args.command == 'menu':
        interactive_menu()
    elif args.command == 'data':
        run_script("data_pipeline.py")
    elif args.command == 'tokenize':
        run_script("tokenizer.py")
    elif args.command == 'train':
        run_script("train.py")
    elif args.command == 'sft':
        run_script("sft.py")
    elif args.command == 'serve':
        subprocess.run([sys.executable, "api_server.py"])
    elif args.command == 'chat':
        run_script("chat.py")
    elif args.command == 'eval':
        run_script("evaluate.py")

if __name__ == "__main__":
    main()
