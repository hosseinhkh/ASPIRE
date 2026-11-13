from cap_harness.trace2vla.cli import main

# The guard matters: worker processes (convert/teacher-grade --workers) start by importing
# this module, and must not run the command again.
if __name__ == "__main__":
    raise SystemExit(main())
