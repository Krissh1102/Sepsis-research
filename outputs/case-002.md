# Case case-002

**Alert level:** HIGH  
**Sepsis risk:** 0.981 (alert threshold 0.693)  
**Proposed action (needs clinician decision):** Clinician review for possible sepsis

**Mortality risk (septic-cohort model):** 0.135  
**Data-driven phenotype cluster:** 2

## Top drivers of the sepsis score

- mean arterial pressure (6-hour mean) = 71.3 (raises risk, contribution +0.809)
- lactate = 3.17 (raises risk, contribution +0.706)
- respiratory rate (6-hour mean) = 18.2 (lowers risk, contribution -0.482)
- total bilirubin = 1.22 (raises risk, contribution +0.319)
- creatinine = 2.27 (raises risk, contribution +0.263)

## Reasoning trace

1. **Score** (SepsisPredictionAgent): Model risk 0.981 is at or above the validation-tuned threshold 0.693.
2. **Drivers** (TreeSHAP): Largest contributions: mean arterial pressure (6-hour mean) raises risk; lactate raises risk; respiratory rate (6-hour mean) lowers risk.
3. **Outcome and phenotype** (DEKD): Mortality model (trained on septic patients only) gives 0.135; patient falls in SHAP-space cluster 2. Leading mortality drivers: Hgb (stay mean), Fibrinogen (stay min), age.
4. **Evidence** (EvidenceRetriever): Retrieved 3 knowledge-base entries: Choudhury et al., ICETCI 2025, Liu et al., IEEE JBHI 2022, Peng et al., IEEE JBHI 2023
5. **Verification** (deletion test): Explanation passed the deletion test (ratio 44.9).
6. **Gate** (ApprovalGate): Proposal is held for clinician review; nothing is actioned automatically.

## Supporting literature (retrieved, verbatim KB entries)

- [Choudhury et al., ICETCI 2025] (match 0.13) XGBoost with a focal weighted binary cross-entropy loss predicts sepsis from ICU vitals and labs on PhysioNet 2019 data. SHAP showed the important features spanned cardiovascular, respiratory, coagulation, liver, renal and biochemical groups. Temperature, ICU length of stay and the missingness of FiO2 were influential.
- [Liu et al., IEEE JBHI 2022] (match 0.10) Time-dependent features (previous values, differences between time points, counts) improve XGBoost sepsis prediction. Top features included lactate, temperature, neutrophils, potassium, glasgow coma score, age and weight.
- [Peng et al., IEEE JBHI 2023] (match 0.06) In preterm infants, heart rate variability, respiration and motion features from bedside monitors predicted late-onset sepsis (AUC 0.88). Motion changes appeared earlier than cardiorespiratory changes.

## Explanation check

Removing the top-5 features changed the score 44.9x more than removing 5 random features -> faithful.

## Caveats

- Research prototype. Trained on whatever data it was given (the bundled demo uses synthetic data), not validated prospectively and not a medical device.
- The mortality model is trained on septic patients only and is not meaningful for patients without sepsis.
- No treatment model is configured. Treatment decisions remain entirely with the treating clinician.
- Some key labs were not measured this hour; their values are carried forward or imputed.
