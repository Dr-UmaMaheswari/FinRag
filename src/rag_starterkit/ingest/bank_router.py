from __future__ import annotations
import os
import re
from typing import Tuple

def infer_bank_and_collection(file_path: str) -> Tuple[str, str]:
    name = os.path.basename(file_path).lower()

    if "rbi" in name:
        return "rbi", "rbi_faq_chunks"
    if "axis" in name:
        return "axis_bank", "axis_bank_policy_chunks"

    if "boi" in name:
        return "boi_bank", "boi_bank_policy_chunks"
    if "barado" in name:
        return "bank_of_barado_bank", "bank_of_barado_bank_policy_chunks"

    if "canara" in name:
        return "canara_bank", "canara_bank_policy_chunks"
    if "csb" in name:
        return "csb_bank", "csb_bank_policy_chunks"
    if "cub" in name:
        return "cub_bank", "cub_bank_policy_chunks"
    if "central" in name:
        return "central_bank", "central_bank_policy_chunks"
    
    if "cscm" in name:
        return "cscm_bank", "cscm_bank_policy_chunks"

    if "hdfc" in name:
        return "hdfc_bank", "hdfc_bank_policy_chunks"
    if "hpscb" in name:
        return "hpscb_bank", "hpscb_bank_policy_chunks"
    
    if "icici" in name:
        return "icici_bank", "icici_bank_policy_chunks"
    
    if "indian" in name:
        return "indian_bank", "indian_bank_policy_chunks"
    
    if "idbi" in name:
        return "idbi", "idbi_policy_chunks"
    
    if "idfc" in name:
        return "idfc", "idfc_policy_chunks"
    
    if "indusind" in name:
        return "indusind_bank", "indusind_bank_policy_chunks"
    if "jk" in name:
        return "jk_bank", "jk_bank_policy_chunks"
    if "dcb" in name:
        return "dcb_bank", "dcb_bank_policy_chunks"
    if "kodak" in name:
        return "kodak_bank", "kodak_bank_policy_chunks"
    if "kvb" in name:
        return "kvb_bank", "kvb_bank_policy_chunks"
    if "maharastra" in name:
        return "maharastra_bank", "maharastra_bank_policy_chunks"
    if "mcb" in name:
        return "mcb_bank", "mcb_bank_policy_chunks"
    if "punjab" in name:
        return "punjab_bank", "punjab_bank_policy_chunks"
    if "tmb" in name:
        return "tmb", "tmb_policy_chunks"
    if "union" in name:
        return "union", "union_policy_chunks"
    if "uco" in name:
        return "uco", "uco_policy_chunks"
    if "yes" in name:
        return "yes_bank", "yes_bank_policy_chunks"
    if "npci" in name:
        return "npci_bank", "npci_bank_policy_chunks"
    return "generic", "generic_policy_chunks"
