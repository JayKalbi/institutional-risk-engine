"""
Document Intelligence & Income Fraud Cross-Verification Module
Compares verified document values (W-2s, Tax Returns, Paystubs) against self-reported loan application data.
"""

class DocumentIntelligenceEngine:
    def verify_income_documents(self, application_income, document_income_w2, document_income_tax):
        """
        Cross-reference self-reported income against W-2 and Tax Return extractions.
        """
        app_inc = float(application_income)
        w2_inc = float(document_income_w2)
        tax_inc = float(document_income_tax)
        
        # Average verified income
        verified_avg = (w2_inc + tax_inc) / 2.0
        discrepancy_pct = abs(app_inc - verified_avg) / (verified_avg + 1.0) * 100.0
        
        if discrepancy_pct < 5.0:
            status = "VERIFIED_VERIFIED"
            risk_level = "LOW FRAUD RISK"
            flag = False
        elif discrepancy_pct < 15.0:
            status = "MODERATE_DISCREPANCY"
            risk_level = "ELEVATED REVIEW REQUIRED"
            flag = False
        else:
            status = "HIGH_DISCREPANCY_FLAG"
            risk_level = "HIGH INCOME FRAUD RISK"
            flag = True
            
        return {
            "application_income": app_inc,
            "verified_document_income": verified_avg,
            "discrepancy_pct": float(discrepancy_pct),
            "fraud_risk_level": risk_level,
            "discrepancy_flag": flag,
            "audit_note": f"Self-reported income (${app_inc:,.2f}) has a {discrepancy_pct:.1f}% variance against verified W-2/Tax extractions (${verified_avg:,.2f})."
        }

if __name__ == "__main__":
    doc_engine = DocumentIntelligenceEngine()
    res = doc_engine.verify_income_documents(120000, 115000, 118000)
    print(res)
