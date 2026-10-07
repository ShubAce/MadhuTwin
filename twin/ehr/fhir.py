"""Export a synthetic patient's EHR (and wearable summaries) as a FHIR R4 Bundle.

Resources are built as plain dicts and validated against the `fhir.resources` R4B models
(wire-compatible with R4 for every resource used here). Terminology follows the ABDM stack:
SNOMED CT for conditions, LOINC for observations, UCUM units, WHO ATC for medicines. CGM
traces are carried as `valueSampledData` (one Observation per day, 5-minute period), and
daily activity / sleep summaries map naturally onto ABDM's WellnessRecord content.

Every resource is tagged as synthetic test data.
"""

from __future__ import annotations

import uuid
import zlib
from datetime import date

import numpy as np
import pandas as pd
from fhir.resources.R4B.bundle import Bundle

from .codes import ATC, CONDITIONS, DBSNP, DRUGS, LABS, LOCAL, LOINC, SNOMED, UCUM, VITALS, WEARABLE
from .population import Profile

SYNTHETIC_TAG = {"system": "http://terminology.hl7.org/CodeSystem/v3-ActReason", "code": "HTEST", "display": "test health data"}
LANG = {"Kannada": "kn", "Tamil": "ta", "Telugu": "te", "Malayalam": "ml", "Hindi": "hi", "Punjabi": "pa",
        "Marathi": "mr", "Gujarati": "gu", "Bengali": "bn", "Odia": "or", "Assamese": "as"}
STATE = {"Bengaluru": "Karnataka", "Mysuru": "Karnataka", "Mangaluru": "Karnataka", "Chennai": "Tamil Nadu",
         "Coimbatore": "Tamil Nadu", "Hyderabad": "Telangana", "Visakhapatnam": "Andhra Pradesh", "Kochi": "Kerala",
         "Delhi": "Delhi", "Lucknow": "Uttar Pradesh", "Kanpur": "Uttar Pradesh", "Varanasi": "Uttar Pradesh",
         "Jaipur": "Rajasthan", "Chandigarh": "Chandigarh", "Ludhiana": "Punjab", "Dehradun": "Uttarakhand",
         "Mumbai": "Maharashtra", "Pune": "Maharashtra", "Nagpur": "Maharashtra", "Nashik": "Maharashtra",
         "Ahmedabad": "Gujarat", "Surat": "Gujarat", "Vadodara": "Gujarat", "Goa": "Goa", "Kolkata": "West Bengal",
         "Siliguri": "West Bengal", "Durgapur": "West Bengal", "Bhubaneswar": "Odisha", "Cuttack": "Odisha",
         "Guwahati": "Assam", "Patna": "Bihar", "Ranchi": "Jharkhand"}
RELATION = {"father": ("FTH", "father"), "mother": ("MTH", "mother"), "both parents": ("PRN", "parent")}
FAMILY_CONDITION = {"type 2 diabetes": "t2d", "coronary artery disease": "cad", "hypertension": "hypertension"}


def _id(*parts: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, "madhutwin/" + "/".join(parts)))


def _meta() -> dict:
    return {"tag": [SYNTHETIC_TAG]}


def _ref(pid: str) -> dict:
    return {"reference": f"Patient/{_id(pid)}"}


def _quantity(value: float, unit: str) -> dict:
    return {"value": round(float(value), 2), "unit": unit, "system": UCUM, "code": unit}


def _obs(pid: str, key: str, code: tuple, value: float, when: str, category: str) -> dict:
    loinc, display, unit = code
    return {
        "resourceType": "Observation", "id": _id(pid, key, when), "meta": _meta(), "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": category}]}],
        "code": {"coding": [{"system": LOINC, "code": loinc, "display": display}], "text": display},
        "subject": _ref(pid), "effectiveDateTime": when, "valueQuantity": _quantity(value, unit),
    }


def patient_bundle(profile: Profile, static_row: pd.Series, lab_history: pd.DataFrame | None = None,
                   series: pd.DataFrame | None = None, ref_date: str = "2026-09-01") -> dict:
    pid = profile.patient_id
    ref = pd.Timestamp(ref_date)
    h = zlib.crc32(pid.encode())
    birth = date(ref.year - profile.age, 1 + h % 12, 1 + (h // 12) % 28)
    given, family = profile.name.split(" ", 1)
    entries: list[dict] = [{
        "resourceType": "Patient", "id": _id(pid), "meta": _meta(),
        "identifier": [{"system": "https://madhutwin.dev/fhir/sid/synthetic-patient", "value": pid}],
        "name": [{"use": "official", "family": family, "given": [given]}],
        "gender": "female" if profile.sex == "F" else "male", "birthDate": birth.isoformat(),
        "address": [{"city": profile.city, "state": STATE.get(profile.city), "country": "IN"}],
        "communication": [{"language": {"coding": [{"system": "urn:ietf:bcp:47", "code": LANG.get(profile.language, "en")}],
                                        "text": profile.language}, "preferred": True}],
    }]

    for key, years in profile.conditions.items():
        code, display = CONDITIONS[key]
        onset = (ref - pd.Timedelta(days=int(years * 365.25))).date().isoformat()
        entries.append({
            "resourceType": "Condition", "id": _id(pid, "cond", key), "meta": _meta(),
            "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
            "verificationStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]},
            "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category", "code": "problem-list-item"}]}],
            "code": {"coding": [{"system": SNOMED, "code": code, "display": display}], "text": display},
            "subject": _ref(pid), "onsetDateTime": onset,
        })

    today = ref.date().isoformat()
    labs_now = {**profile.labs_latent, **{k: static_row[k] for k in ("hba1c_pct", "fpg_mgdl", "fasting_insulin_uU")}}
    for key, code in LABS.items():
        if key in labs_now and pd.notna(labs_now[key]):
            entries.append(_obs(pid, key, code, labs_now[key], today, "laboratory"))
    if lab_history is not None:
        for _, r in lab_history[lab_history.patient_id == pid].iterrows():
            d = pd.Timestamp(r["date"]).date().isoformat()
            entries.append(_obs(pid, "hba1c_pct", LABS["hba1c_pct"], r["hba1c_pct"], d, "laboratory"))
            entries.append(_obs(pid, "fpg_mgdl", LABS["fpg_mgdl"], r["fpg_mgdl"], d, "laboratory"))
            entries.append(_obs(pid, "weight_kg", VITALS["weight_kg"], r["weight_kg"], d, "vital-signs"))

    for key in ("bmi", "weight_kg", "height_cm", "waist_cm"):
        entries.append(_obs(pid, key, VITALS[key], getattr(profile, key), today, "vital-signs"))
    entries.append({
        "resourceType": "Observation", "id": _id(pid, "bp", today), "meta": _meta(), "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "vital-signs"}]}],
        "code": {"coding": [{"system": LOINC, "code": "85354-9", "display": "Blood pressure panel"}]},
        "subject": _ref(pid), "effectiveDateTime": today,
        "component": [
            {"code": {"coding": [{"system": LOINC, "code": VITALS["sbp"][0], "display": VITALS["sbp"][1]}]},
             "valueQuantity": _quantity(profile.cardio["sbp"], "mm[Hg]")},
            {"code": {"coding": [{"system": LOINC, "code": VITALS["dbp"][0], "display": VITALS["dbp"][1]}]},
             "valueQuantity": _quantity(profile.cardio["dbp"], "mm[Hg]")},
        ],
    })

    for k, med in enumerate(profile.meds):
        atc, display, _cls, unit = DRUGS[med.drug]
        times = ", ".join(f"{t // 60:02d}:{t % 60:02d}" for t in med.times)
        entries.append({
            "resourceType": "MedicationStatement", "id": _id(pid, "med", str(k)), "meta": _meta(), "status": "active",
            "medicationCodeableConcept": {"coding": [{"system": ATC, "code": atc, "display": display}], "text": display},
            "subject": _ref(pid),
            "effectivePeriod": {"start": (ref - pd.Timedelta(days=int(med.started_years_ago * 365.25))).date().isoformat()},
            "dosage": [{"text": f"{med.dose:g} {unit} at {times}",
                        "timing": {"repeat": {"frequency": len(med.times), "period": 1, "periodUnit": "d",
                                              "timeOfDay": [f"{t // 60:02d}:{t % 60:02d}:00" for t in med.times]}},
                        "doseAndRate": [{"doseQuantity": _quantity(med.dose, "[IU]" if unit == "IU" else unit)}]}],
        })

    for k, item in enumerate(profile.family_history):
        rel, cond = item.split(": ")
        rcode, rdisp = RELATION[rel]
        ccode, cdisp = CONDITIONS[FAMILY_CONDITION[cond]]
        entries.append({
            "resourceType": "FamilyMemberHistory", "id": _id(pid, "fmh", str(k)), "meta": _meta(), "status": "completed",
            "patient": _ref(pid),
            "relationship": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/v3-RoleCode", "code": rcode, "display": rdisp}]},
            "condition": [{"code": {"coding": [{"system": SNOMED, "code": ccode, "display": cdisp}]}}],
        })

    alleles = int(profile.genetics["TCF7L2_rs7903146_T_alleles"])
    comps = [{"code": {"coding": [{"system": LOINC, "code": "81255-2", "display": "dbSNP [ID]"}]},
              "valueCodeableConcept": {"coding": [{"system": DBSNP, "code": "rs7903146"}], "text": "TCF7L2 rs7903146 (T risk allele)"}}]
    if alleles:
        comps.append({"code": {"coding": [{"system": LOINC, "code": "53034-5", "display": "Allelic state"}]},
                      "valueCodeableConcept": {"coding": [{"system": LOINC, "code": "LA6706-1" if alleles == 1 else "LA6705-3",
                                                           "display": "Heterozygous" if alleles == 1 else "Homozygous"}]}})
    entries.append({
        "resourceType": "Observation", "id": _id(pid, "tcf7l2"), "meta": _meta(), "status": "final",
        "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory"}]}],
        "code": {"coding": [{"system": LOINC, "code": "69548-6", "display": "Genetic variant assessment"}]},
        "subject": _ref(pid), "effectiveDateTime": today,
        "valueCodeableConcept": {"coding": [{"system": LOINC, "code": "LA9633-4" if alleles else "LA9634-2",
                                             "display": "Present" if alleles else "Absent"}]},
        "component": comps,
    })
    entries.append({
        "resourceType": "Observation", "id": _id(pid, "prs"), "meta": _meta(), "status": "final",
        "code": {"coding": [{"system": LOCAL, "code": "prs-t2d", "display": "Type 2 diabetes polygenic risk score (z)"}]},
        "subject": _ref(pid), "effectiveDateTime": today,
        "valueQuantity": {"value": float(profile.genetics["prs_t2d_z"]), "unit": "SD", "system": UCUM, "code": "1"},
    })

    if series is not None:
        s = series[series.patient_id == pid].copy()
        s["day"] = s["ts"].dt.date
        for day, g in s.groupby("day"):
            d = day.isoformat()
            vals = " ".join("E" if not np.isfinite(v) else str(int(round(v))) for v in g["cgm"].to_numpy())
            entries.append({
                "resourceType": "Observation", "id": _id(pid, "cgm", d), "meta": _meta(), "status": "final",
                "category": [{"coding": [{"system": "http://terminology.hl7.org/CodeSystem/observation-category", "code": "laboratory"}]}],
                "code": {"coding": [{"system": LOINC, "code": WEARABLE["cgm"][0], "display": WEARABLE["cgm"][1]}]},
                "subject": _ref(pid), "effectivePeriod": {"start": f"{d}T00:00:00+05:30", "end": f"{d}T23:59:59+05:30"},
                "valueSampledData": {"origin": _quantity(0, "mg/dL"), "period": 300000, "dimensions": 1,
                                     "lowerLimit": 40, "upperLimit": 400, "data": vals},
            })
            entries.append(_obs(pid, "steps", WEARABLE["steps"], float(np.nansum(g["steps"])), d, "activity"))
        for night_day, hours in s.groupby("day")["sleep_stage"].apply(lambda x: (x > 0).sum() * 5 / 60).items():
            entries.append(_obs(pid, "sleep", WEARABLE["sleep_hours"], hours, night_day.isoformat(), "activity"))

    bundle = {"resourceType": "Bundle", "id": _id(pid, "bundle"), "meta": _meta(), "type": "collection",
              "timestamp": f"{today}T09:00:00+05:30",
              "entry": [{"fullUrl": f"urn:uuid:{e['id']}", "resource": e} for e in entries]}
    Bundle.model_validate(bundle)  # raises on any schema violation
    return bundle


def _dt(ts, tz: str = "+05:30") -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%S") + tz


def minimal_bundle(pid: str, static: dict, series: pd.DataFrame | None = None, ref_date: str | None = None,
                   tz: str = "+00:00") -> dict:
    """FHIR R4 Bundle for a de-identified real-world recording (no name/address: only coded data)."""
    today = (ref_date or pd.Timestamp.now().date().isoformat())[:10]
    entries: list[dict] = [{
        "resourceType": "Patient", "id": _id(pid), "meta": _meta(),
        "identifier": [{"system": "https://madhutwin.dev/fhir/sid/research-recording", "value": pid}],
        "gender": {"F": "female", "M": "male"}.get(static.get("sex"), "unknown"),
    }]
    status = static.get("status")
    if status in ("t2d", "prediabetes"):
        code, display = CONDITIONS[status]
        entries.append({"resourceType": "Condition", "id": _id(pid, "cond", status), "meta": _meta(),
                        "clinicalStatus": {"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
                        "code": {"coding": [{"system": SNOMED, "code": code, "display": display}]}, "subject": _ref(pid)})
    for key, code in LABS.items():
        v = static.get(key)
        if v is not None and pd.notna(v):
            entries.append(_obs(pid, key, code, float(v), today, "laboratory"))
    if series is not None and len(series):
        s = series.copy()
        s["day"] = s["ts"].dt.date
        for day, g in s.groupby("day"):
            d = day.isoformat()
            vals = " ".join("E" if not np.isfinite(v) else str(int(round(v))) for v in g["cgm"].to_numpy(dtype=float))
            entries.append({
                "resourceType": "Observation", "id": _id(pid, "cgm", d), "meta": _meta(), "status": "final",
                "code": {"coding": [{"system": LOINC, "code": WEARABLE["cgm"][0], "display": WEARABLE["cgm"][1]}]},
                "subject": _ref(pid), "effectivePeriod": {"start": f"{d}T00:00:00{tz}", "end": f"{d}T23:59:59{tz}"},
                "valueSampledData": {"origin": _quantity(0, "mg/dL"), "period": 300000, "dimensions": 1,
                                     "lowerLimit": 40, "upperLimit": 400, "data": vals},
            })
    bundle = {"resourceType": "Bundle", "id": _id(pid, "bundle"), "meta": _meta(), "type": "collection",
              "entry": [{"fullUrl": f"urn:uuid:{e['id']}", "resource": e} for e in entries]}
    Bundle.model_validate(bundle)
    return bundle


def risk_assessment(pid: str, when: str, p_spike: float, p_hypo: float, q50: list[float], reasons: list[str] | None = None,
                    model: str = "MadhuTwin hybrid digital twin 0.1", tz: str = "+05:30") -> dict:
    """FHIR RiskAssessment carrying the twin's 2-hour glycaemic event forecast."""
    from fhir.resources.R4B.riskassessment import RiskAssessment

    start, end = _dt(when, tz), _dt(pd.Timestamp(when) + pd.Timedelta(hours=2), tz)
    ra = {
        "resourceType": "RiskAssessment", "id": _id(pid, "risk", when), "meta": _meta(), "status": "preliminary",
        "subject": _ref(pid), "occurrenceDateTime": start,
        "method": {"text": model},
        "prediction": [
            {"outcome": {"coding": [{"system": LOCAL, "code": "hyperglycaemia-180-15min", "display": "Glucose >180 mg/dL for >=15 min"}]},
             "probabilityDecimal": round(float(p_spike), 3), "whenPeriod": {"start": start, "end": end}},
            {"outcome": {"coding": [{"system": LOCAL, "code": "hypoglycaemia-70-15min", "display": "Glucose <70 mg/dL for >=15 min"}]},
             "probabilityDecimal": round(float(p_hypo), 3), "whenPeriod": {"start": start, "end": end}},
        ],
        "note": [{"text": "Predicted median glucose (mg/dL) every 5 min for 2 h: " + ", ".join(str(int(round(v))) for v in q50 if v is not None)}]
        + ([{"text": "Main drivers: " + "; ".join(reasons)}] if reasons else []),
    }
    RiskAssessment.model_validate(ra)
    return ra
