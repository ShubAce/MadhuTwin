# Privacy and data protection: DPDP Act 2023 and ABDM alignment

MadhuTwin handles health data, the most sensitive category of personal data. This note maps the design to India's Digital Personal Data Protection Act 2023 (DPDP) and the Ayushman Bharat Digital Mission (ABDM) Health Data Management Policy. The proof of concept itself uses **no real Indian patient data**: synthetic patients plus de-identified open research datasets.

| Obligation | How MadhuTwin addresses it |
|---|---|
| Lawful processing and consent (DPDP §4–6) | Every virtual patient carries a consent artefact: purposes (treatment, care coordination), scope and expiry, shown in the dashboard. In deployment, EHR records are pulled only through ABDM's consent manager flow (ABHA-linked, purpose-bound, time-bound). |
| Purpose limitation | The twin uses data only to forecast and explain glycaemic risk for the treating team. The API audit trail records the declared purpose of each access (`x-purpose` header). |
| Data minimisation | Prediction needs CGM, activity, sleep and a short EHR summary; names and addresses are not model inputs. Real-world records are shown by research ID only. |
| Accuracy | Labs and wearable data keep provenance (source system, LOINC code, timestamp). The model card documents sensor bias and other known error sources. |
| Storage limitation | Designed so raw wearable streams can be summarised and discarded after a retention window; the twin keeps parameters and summaries, not raw history. |
| Security safeguards (DPDP §8(5)) | In-country hosting, TLS, role-based access, and an append-only audit log (prototype: in-memory, surfaced at `/api/audit`). No patient data leaves the deployment for inference: the optional LLM assistant is off by default and is enabled only by explicit configuration. |
| Rights of the data principal (§11–14) | A FHIR export lets a patient's data, including the twin's risk assessment, be shared or ported. Erasure removes the patient's bundle and twin parameters. |
| Accountability and breach notification | The audit trail supports investigation; deployment would add Data Protection Board notification procedures. |
| Children's data (§9) | Out of scope: adults only. |

## Interoperability

- **FHIR R4** bundles validated against the FHIR schema (R4B models): Patient, Condition (SNOMED CT), Observation (LOINC; CGM as `SampledData`, LOINC 99504-3), MedicationStatement (WHO ATC), FamilyMemberHistory, genomic Observation (LOINC 69548-6, dbSNP rs7903146), and **RiskAssessment** for the twin's live forecast.
- Terminology follows the ABDM / NRCeS stack (SNOMED CT, LOINC, UCUM). Daily activity and sleep summaries map naturally onto ABDM's WellnessRecord content.

## Open datasets and licences

- CGMacros: PhysioNet, CC BY-NC-SA 4.0. Not redistributed in raw form; downloaded by `scripts/download_data.py`. The two CGMacros-derived demo bundles in `artifacts/demo/` carry the same licence.
- ShanghaiT2DM: Figshare, CC BY 4.0, with attribution.
