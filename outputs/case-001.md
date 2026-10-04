# Case case-001

**Alert level:** HIGH  
**Sepsis risk:** 0.996 (alert threshold 0.693)  
**Proposed action (needs clinician decision):** Clinician review for possible sepsis

**Mortality risk (septic-cohort model):** 0.081  
**Data-driven phenotype cluster:** 0

## Top drivers of the sepsis score

- respiratory rate (6-hour mean) = 23.2 (raises risk, contribution +0.690)
- systolic blood pressure (6-hour mean) = 84.8 (raises risk, contribution +0.516)
- mean arterial pressure (6-hour mean) = 71.8 (raises risk, contribution +0.481)
- lactate = 3.03 (raises risk, contribution +0.434)
- total bilirubin = 1.8 (raises risk, contribution +0.268)

## Reasoning trace

1. **Score** (SepsisPredictionAgent): Model risk 0.996 is at or above the validation-tuned threshold 0.693.
2. **Drivers** (TreeSHAP): Largest contributions: respiratory rate (6-hour mean) raises risk; systolic blood pressure (6-hour mean) raises risk; mean arterial pressure (6-hour mean) raises risk.
3. **Outcome and phenotype** (DEKD): Mortality model (trained on septic patients only) gives 0.081; patient falls in SHAP-space cluster 0. Leading mortality drivers: Fibrinogen (stay min), Hgb (stay mean), end-tidal CO2 (stay mean).
4. **Evidence** (EvidenceRetriever): Retrieved 3 knowledge-base entries: Choudhury et al., ICETCI 2025, Peng et al., IEEE JBHI 2023, Liu et al., IEEE JBHI 2022
5. **Verification** (deletion test): Explanation passed the deletion test (ratio 108.1).
6. **Gate** (ApprovalGate): Proposal is held for clinician review; nothing is actioned automatically.

## Supporting literature (retrieved, verbatim KB entries)

- [Choudhury et al., ICETCI 2025] (match 0.13) XGBoost with a focal weighted binary cross-entropy loss predicts sepsis from ICU vitals and labs on PhysioNet 2019 data. SHAP showed the important features spanned cardiovascular, respiratory, coagulation, liver, renal and biochemical groups. Temperature, ICU length of stay and the missingness of FiO2 were influential.
- [Peng et al., IEEE JBHI 2023] (match 0.05) In preterm infants, heart rate variability, respiration and motion features from bedside monitors predicted late-onset sepsis (AUC 0.88). Motion changes appeared earlier than cardiorespiratory changes.
- [Liu et al., IEEE JBHI 2022] (match 0.05) Time-dependent features (previous values, differences between time points, counts) improve XGBoost sepsis prediction. Top features included lactate, temperature, neutrophils, potassium, glasgow coma score, age and weight.

## Explanation check

Removing the top-5 features changed the score 108.1x more than removing 5 random features -> faithful.

## Caveats

- Research prototype. Trained on whatever data it was given (the bundled demo uses synthetic data), not validated prospectively and not a medical device.
- The mortality model is trained on septic patients only and is not meaningful for patients without sepsis.
- No treatment model is configured. Treatment decisions remain entirely with the treating clinician.
- Some key labs were not measured this hour; their values are carried forward or imputed.
