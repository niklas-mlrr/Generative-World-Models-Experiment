import os

if "MPLCONFIGDIR" not in os.environ:
    os.environ["MPLCONFIGDIR"] = "/tmp/mplconfig_reliability_paradox"

from toywm.cli import main


if __name__ == "__main__":
    main()
