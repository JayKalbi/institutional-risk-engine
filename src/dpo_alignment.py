"""
Direct Preference Optimization (DPO) Alignment Dataset Builder
Generates pairwise (chosen vs rejected) credit memorandum narratives for H100 alignment fine-tuning.
"""
import json
import os

class DPOAlignmentBuilder:
    def build_preference_pair(self, prompt, quantitative_summary):
        """
        Creates a chosen (preferred) vs rejected (dispreferred) response pair.
        """
        chosen = f"""**Capacity:** The applicant demonstrates solid debt service capacity with a verified DTI of {quantitative_summary.get('dti', 38)}%. Monthly cash flows adequately support the proposed mortgage obligations under CFPB Regulation Z guidelines.

**Capital:** Down payment provides a suitable equity cushion, though LTV ratio of {quantitative_summary.get('ltv', 85)}% warrants Private Mortgage Insurance (PMI) coverage.

**Collateral:** Property valuation provides sufficient asset backing for the requested loan amount.

**Character:** Continuous employment history and absence of adverse credit events confirm satisfactory willingness to repay.

**Conditions:** Proposed loan term and interest rate structure align with current market standards.

**Proposed Covenants/Mitigants:** Mandate PMI coverage and escrow reserve requirements prior to funding."""

        rejected = f"""The applicant wants a loan of ${quantitative_summary.get('loan_amount', 350000)}. Everything looks mostly fine. We should probably approve this application because their income seems okay and they have a job. The property is good collateral. Approval recommended without any specific conditions."""

        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected
        }

if __name__ == "__main__":
    builder = DPOAlignmentBuilder()
    pair = builder.build_preference_pair("Evaluate loan application #1042", {"dti": 38.5, "ltv": 85.0, "loan_amount": 350000})
    print("DPO Pair Generated Successfully.")
    print("Chosen length:", len(pair["chosen"]), "| Rejected length:", len(pair["rejected"]))
