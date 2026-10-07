"""Clinical terminology used in the EHR stream (aligned with ABDM's FHIR R4 + SNOMED CT + LOINC stack).

Medications use WHO ATC codes, which are international and better suited to the Indian
formulary (e.g. teneligliptin, gliclazide) than US-centric RxNorm.
"""

from __future__ import annotations

SNOMED = "http://snomed.info/sct"
LOINC = "http://loinc.org"
ATC = "http://www.whocc.no/atc"
UCUM = "http://unitsofmeasure.org"
DBSNP = "http://www.ncbi.nlm.nih.gov/projects/SNP"
LOCAL = "https://madhutwin.dev/fhir/CodeSystem/local"

CONDITIONS = {
    "t2d": ("44054006", "Type 2 diabetes mellitus"),
    "prediabetes": ("714628002", "Prediabetes"),
    "hypertension": ("38341003", "Hypertensive disorder"),
    "dyslipidemia": ("370992007", "Dyslipidemia"),
    "obesity": ("414916001", "Obesity"),
    "nafld": ("197321007", "Steatosis of liver"),
    "ckd": ("709044004", "Chronic kidney disease"),
    "diabetic_nephropathy": ("127013003", "Disorder of kidney due to diabetes mellitus"),
    "retinopathy": ("4855003", "Retinopathy due to diabetes mellitus"),
    "neuropathy": ("230572002", "Neuropathy due to diabetes mellitus"),
    "cad": ("53741008", "Coronary arteriosclerosis"),
    "hypothyroidism": ("40930008", "Hypothyroidism"),
    "osa": ("78275009", "Obstructive sleep apnea syndrome"),
}

# key: (LOINC, display, UCUM unit)
LABS = {
    "hba1c_pct": ("4548-4", "Hemoglobin A1c/Hemoglobin.total in Blood", "%"),
    "fpg_mgdl": ("1558-6", "Fasting glucose [Mass/volume] in Serum or Plasma", "mg/dL"),
    "fasting_insulin_uU": ("20448-7", "Insulin [Units/volume] in Serum or Plasma", "u[IU]/mL"),
    "chol_mgdl": ("2093-3", "Cholesterol [Mass/volume] in Serum or Plasma", "mg/dL"),
    "hdl_mgdl": ("2085-9", "Cholesterol in HDL [Mass/volume] in Serum or Plasma", "mg/dL"),
    "ldl_mgdl": ("13457-7", "Cholesterol in LDL [Mass/volume] in Serum or Plasma by calculation", "mg/dL"),
    "tg_mgdl": ("2571-8", "Triglyceride [Mass/volume] in Serum or Plasma", "mg/dL"),
    "creatinine_mgdl": ("2160-0", "Creatinine [Mass/volume] in Serum or Plasma", "mg/dL"),
    "egfr": ("98979-8", "Glomerular filtration rate/1.73 sq M.predicted [Volume Rate/Area] by CKD-EPI 2021", "mL/min/{1.73_m2}"),
    "alt": ("1742-6", "Alanine aminotransferase [Enzymatic activity/volume] in Serum or Plasma", "U/L"),
    "uacr": ("9318-7", "Albumin/Creatinine [Mass Ratio] in Urine", "mg/g"),
    "b12": ("2132-9", "Cobalamin (Vitamin B12) [Mass/volume] in Serum or Plasma", "pg/mL"),
    "tsh": ("3016-3", "Thyrotropin [Units/volume] in Serum or Plasma", "m[IU]/L"),
}

VITALS = {
    "bmi": ("39156-5", "Body mass index (BMI) [Ratio]", "kg/m2"),
    "weight_kg": ("29463-7", "Body weight", "kg"),
    "height_cm": ("8302-2", "Body height", "cm"),
    "sbp": ("8480-6", "Systolic blood pressure", "mm[Hg]"),
    "dbp": ("8462-4", "Diastolic blood pressure", "mm[Hg]"),
    "waist_cm": ("8280-0", "Waist Circumference at umbilicus by Tape measure", "cm"),
}

WEARABLE = {
    "cgm": ("99504-3", "Glucose [Mass/volume] in Interstitial fluid", "mg/dL"),
    "hr": ("8867-4", "Heart rate", "/min"),
    "steps": ("41950-7", "Number of steps in 24 hour Measured", "/(24.h)"),
    "sleep_hours": ("93832-4", "Sleep duration", "h"),
}

GENETIC = {
    "variant_assessment": ("69548-6", "Genetic variant assessment"),
    "dbsnp_id": ("81255-2", "dbSNP [ID]"),
    "allelic_state": ("53034-5", "Allelic state"),
}

# key: (ATC code, display, class, unit)
DRUGS = {
    "metformin": ("A10BA02", "Metformin", "biguanide", "mg"),
    "glimepiride": ("A10BB12", "Glimepiride", "sulfonylurea", "mg"),
    "gliclazide": ("A10BB09", "Gliclazide", "sulfonylurea", "mg"),
    "teneligliptin": ("A10BH08", "Teneligliptin", "dpp4", "mg"),
    "sitagliptin": ("A10BH01", "Sitagliptin", "dpp4", "mg"),
    "vildagliptin": ("A10BH02", "Vildagliptin", "dpp4", "mg"),
    "dapagliflozin": ("A10BK01", "Dapagliflozin", "sglt2", "mg"),
    "empagliflozin": ("A10BK03", "Empagliflozin", "sglt2", "mg"),
    "voglibose": ("A10BF03", "Voglibose", "agi", "mg"),
    "pioglitazone": ("A10BG03", "Pioglitazone", "tzd", "mg"),
    "insulin_premix_30_70": ("A10AD01", "Insulin (human) biphasic 30/70", "insulin", "IU"),
    "insulin_glargine": ("A10AE04", "Insulin glargine", "insulin", "IU"),
    "atorvastatin": ("C10AA05", "Atorvastatin", "statin", "mg"),
    "rosuvastatin": ("C10AA07", "Rosuvastatin", "statin", "mg"),
    "telmisartan": ("C09CA07", "Telmisartan", "arb", "mg"),
    "amlodipine": ("C08CA01", "Amlodipine", "ccb", "mg"),
    "aspirin": ("B01AC06", "Acetylsalicylic acid", "antiplatelet", "mg"),
    "levothyroxine": ("H03AA01", "Levothyroxine sodium", "thyroid", "mcg"),
}

STATUS_CONDITION = {"t2d": "t2d", "prediabetes": "prediabetes"}
