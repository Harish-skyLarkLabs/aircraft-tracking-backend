#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys
import multiprocessing

# Set multiprocessing start method to 'spawn' right at the beginning
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    multiprocessing.set_start_method('spawn', force=True)


def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
    try:
        from django.core.management import execute_from_command_line
        if len(sys.argv) > 1 and sys.argv[1] == 'celery':
            # Set environment variable for Celery to use spawn method
            os.environ['FORKED_BY_MULTIPROCESSING'] = '1'

            # Ensure any CUDA initialization happens before forking
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.init()
            except ImportError:
                pass

            from backend.celery import celery
            celery.start(argv=sys.argv[2:])
        else:
            execute_from_command_line(sys.argv)
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc


if __name__ == '__main__':
    main()

