"""Project 2: knowledge distillation. Code package `kd`.

Importing `kd` makes the Project 1 teacher package importable, so its model code can be reused:
    import kd                      # adds ../project1/teacher to sys.path
    from src.model import GPT      # the teacher's GPT class (the student uses the same class, smaller)

Project 1 is only read, never changed. Bytecode caching is turned off so no __pycache__ files are written there.
"""
import os
import sys

sys.dont_write_bytecode = True

PROJECT2 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT1 = os.path.join(os.path.dirname(PROJECT2), "project1")
TEACHER_DIR = os.path.join(PROJECT1, "teacher")
DATA_DIR = os.path.join(PROJECT1, "data", "full")  # all cleaned TinyStories, tokenized (train/val/test .bin)
TOKENIZER = os.path.join(TEACHER_DIR, "tokenizer.json")

if TEACHER_DIR not in sys.path:
    sys.path.insert(0, TEACHER_DIR)
