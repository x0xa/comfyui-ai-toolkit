"""Loads the shared Fantasio library from the sibling custom node package.

S3 access, WebP encoding and archive handling live in the ComfyUI-Fantasio-Nodes
package (`custom_nodes/fantasio/lib.py`). The training integration here loads that
single implementation by path instead of duplicating it.
"""

import os
import importlib.util

FANTASIO_DIR_NAME = "fantasio"


def load_fantasio_lib():
    here = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.dirname(here)
    custom_nodes_dir = os.path.dirname(package_root)
    lib_path = os.path.join(custom_nodes_dir, FANTASIO_DIR_NAME, "lib.py")

    if not os.path.isfile(lib_path):
        raise RuntimeError(
            f"Fantasio shared library not found at {lib_path}. "
            f"The {FANTASIO_DIR_NAME} custom node package must be installed alongside this one."
        )

    spec = importlib.util.spec_from_file_location("fantasio_shared_lib", lib_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
