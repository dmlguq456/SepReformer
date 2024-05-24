import os
import logging
import inspect

# Call logger for monitoring
logger = logging.getLogger("sys_log")

def parse_scps(scp_path):
    scp_dict = dict()
    try:
        if not os.path.exists(scp_path): raise FileNotFoundError(f"File not found: {scp_path}")
        with open(scp_path, 'r') as f:
            for scp in f:
                scp_tokens = scp.strip().split()
                if len(scp_tokens) != 2: raise RuntimeError(f"Error format of context '{scp}'")
                key, addr = scp_tokens
                if key in scp_dict: raise ValueError(f"Duplicate key '{key}' exists!")
                scp_dict[key] = addr
    except (RuntimeError, ValueError) as e:
        logger.error(e)
        raise
    finally:
        func_name = inspect.currentframe().f_code.co_name
        logger.debug(f"Complete {__name__}.{func_name}")
    return scp_dict