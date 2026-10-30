"""OneData defaults for the UC3.3 SoMES pieces.

No credentials are built in: set ``onedata_onezone_host``, ``onedata_token`` and
``onedata_output_dir`` as Domino repository secrets (P9 / GEN-E01-US07). With the
token secret empty, results stay on local Domino shared storage; on SPICE the
token is injected as a secret and never committed to the repo or the import.
"""

DEFAULT_ONEZONE_HOST = "data.spice-platform.eu"
DEFAULT_INPUT_DIR = "onedata:///SCDI/UC3.3_SOMES/inputs"
DEFAULT_OUTPUT_DIR = "onedata:///SCDI/UC3.3_SOMES/outputs"
DEFAULT_ONEDATA_TOKEN = ""
