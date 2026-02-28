# class_map.py — UPDATED with exact class names from your output
# ─────────────────────────────────────────────────────────────────
# 9 canonical GHS classes we care about
CANONICAL_CLASSES = [
    "explosive",       # GHS01
    "flammable",       # GHS02
    "oxidizer",        # GHS03
    "compressed_gas",  # GHS04
    "corrosive",       # GHS05
    "toxic",           # GHS06
    "harmful",         # GHS07
    "health_hazard",   # GHS08
    "environmental",   # GHS09
]
CANONICAL_TO_IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}

# ─────────────────────────────────────────────────────────────────
# EXACT class names from your data.yaml files → canonical names
# None = skip this class entirely (non-GHS safety equipment etc.)
# ─────────────────────────────────────────────────────────────────
CLASS_NORMALIZATION = {

    # ══════════════════════════════════════════
    # DATASET 1 — Gefahrensymbole (16 classes)
    # ══════════════════════════════════════════
    "No access for unauthorized persons": None,   # class 0  — skip
    "acute toxicity":                     "toxic",          # class 1  — GHS06
    "corrosive":                          "corrosive",      # class 2  — GHS05
    "explosive":                          "explosive",      # class 3  — GHS01
    "eye shower":                         None,             # class 4  — skip
    "fire extinguisher":                  None,             # class 5  — skip
    "fire hazard":                        "flammable",      # class 6  — GHS02
    "first aid kit":                      None,             # class 7  — skip
    "flammable":                          "flammable",      # class 8  — GHS02
    "gas under pressure":                 "compressed_gas", # class 9  — GHS04
    "hazardous to the environment":       "environmental",  # class 10 — GHS09
    "health-ozone layer hazard":          "harmful",        # class 11 — GHS07
    "no GHS symbol":                      None,             # class 12 — skip
    "oxidising":                          "oxidizer",       # class 13 — GHS03
    "serious health hazard":              "health_hazard",  # class 14 — GHS08
    "shower":                             None,             # class 15 — skip

    # ══════════════════════════════════════════
    # DATASET 2 — AI Pictogram DocExtract (4 classes)
    # ══════════════════════════════════════════
    "GHS_Symbol_CORROSION":       "corrosive",   # class 0 — GHS05
    "GHS_Symbol_ENVIRONMENT":     "environmental", # class 1 — GHS09
    "GHS_Symbol_EXCLAMATION_MARK":"harmful",     # class 2 — GHS07
    "GHS_Symbol_FLAME":           "flammable",   # class 3 — GHS02

    # ══════════════════════════════════════════
    # DATASET 3 — GHS hze4d (27 classes)
    # ══════════════════════════════════════════

    # — DOT Transport classes (0-17) — map to nearest GHS or skip —
    "Class 3":              "flammable",      # flammable liquids
    "Class 6-2":            "toxic",          # infectious substances → closest is toxic
    "Class 7":              None,             # radioactive → no GHS equivalent, skip
    "Class 8":              "corrosive",      # corrosive materials
    "Class 9":              None,             # miscellaneous → skip
    "Division 1-4":         "explosive",      # explosive Div 1.4
    "Division 1-5":         "explosive",      # explosive Div 1.5
    "Division 1-6":         "explosive",      # explosive Div 1.6
    "Division 2-1":         "flammable",      # flammable gas
    "Division 2-2":         "compressed_gas", # non-flammable gas
    "Division 2-3":         "toxic",          # toxic gas
    "Division 4-1":         "flammable",      # flammable solid
    "Division 4-2":         "flammable",      # spontaneously combustible
    "Division 4-3":         "flammable",      # dangerous when wet
    "Division 5-1":         "oxidizer",       # oxidizing substances
    "Division 5-2":         "oxidizer",       # organic peroxide
    "Division 6-1":         "toxic",          # toxic substances
    "Divisions 1-1-C1-3":   "explosive",      # explosive Div 1.1

    # — Pure GHS classes (18-26) — clean 1:1 mapping —
    "GHS01- Explosive":                         "explosive",
    "GHS02- Flammable":                         "flammable",
    "GHS03- Oxidizing":                         "oxidizer",
    "GHS04- Compressed Gas":                    "compressed_gas",
    "GHS05- Corrosive":                         "corrosive",
    "GHS06- Toxic":                             "toxic",
    "GHS07- Health Hazard-Hazardous to Ozone Layer": "harmful",
    "GHS08- Serious Health hazard":             "health_hazard",
    "GHS09- Hazardous to the Environment":      "environmental",
}