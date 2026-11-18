"""PyInstaller entry point (a script, not `-m course_assistant`)."""
import multiprocessing

from course_assistant.cli import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
