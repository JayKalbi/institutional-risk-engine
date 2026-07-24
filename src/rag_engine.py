"""
Regulatory Retrieval-Augmented Generation (RAG) Engine
Indexes CFPB Regulation Z (12 CFR § 1026.43), ECOA (12 CFR Part 1002), and Basel III Guidelines
to retrieve statutory citations for credit underwriting justifications.
"""
import re

REGULATORY_KNOWLEDGEBASE = [
    {
        "citation": "12 CFR § 1026.43(c) - CFPB Ability-to-Repay Rule",
        "topic": "Debt-to-Income (DTI) & Cash Flow Capacity",
        "text": "A creditor shall not make a covered mortgage loan unless the creditor makes a reasonable and good faith determination based on documented and verified information that the consumer has a reasonable ability to repay the loan according to its terms. The debt-to-income ratio generally should not exceed 43% for standard Qualified Mortgages.",
        "keywords": ["dti", "income", "capacity", "payment", "burden", "debt"]
    },
    {
        "citation": "12 CFR § 1002.9 - ECOA Adverse Action Notification",
        "topic": "Fair Lending & Adverse Action",
        "text": "Under the Equal Credit Opportunity Act, a creditor must notify an applicant of action taken within 30 days after receiving a completed application. If denied or approved with mitigants, the creditor must provide specific principal reasons for the adverse action.",
        "keywords": ["adverse", "deny", "notification", "ecoa", "fairness", "race", "sex", "age"]
    },
    {
        "citation": "Basel III Framework - Minimum Capital Requirements",
        "topic": "Loss Given Default & Collateral LTV Cushion",
        "text": "Banks must hold sufficient Tier 1 Capital against Expected Credit Loss (ECL = EAD * PD * LGD). For residential real estate, Loan-to-Value (LTV) ratios exceeding 80% require credit enhancement (Private Mortgage Insurance or additional capital reserves).",
        "keywords": ["ltv", "collateral", "ecl", "lgd", "property", "capital", "basel"]
    },
    {
        "citation": "12 CFR § 1026.43(e)(2) - Qualified Mortgage (QM) Standards",
        "topic": "Loan Term & Amortization Conditions",
        "text": "A Qualified Mortgage cannot provide for negative amortization, interest-only payments, or balloon payments, and the loan term must not exceed 30 years (360 months).",
        "keywords": ["term", "conditions", "amortization", "balloon", "qm", "interest"]
    }
]

class RegulatoryRAGEngine:
    def __init__(self):
        self.kb = REGULATORY_KNOWLEDGEBASE

    def retrieve_citations(self, query_text, top_k=2):
        """
        Retrieve relevant statutory citations for a given query or risk context.
        """
        words = re.findall(r'\w+', query_text.lower())
        scored = []
        
        for item in self.kb:
            score = 0
            for kw in item["keywords"]:
                if kw in words or any(kw in w for w in words):
                    score += 1
            scored.append((score, item))
            
        scored.sort(key=lambda x: x[0], reverse=True)
        results = [item for score, item in scored[:top_k] if score > 0]
        
        if not results:
            results = [self.kb[0]] # Fallback to Ability-to-Repay Rule
            
        return results

    def format_citation_prompt(self, citations):
        """
        Format retrieved legal citations into prompt context for LLM narrative synthesis.
        """
        formatted = "Regulatory Statutory Framework Citations:\n"
        for idx, c in enumerate(citations, 1):
            formatted += f"[{idx}] {c['citation']}\n    Rule Summary: {c['text']}\n"
        return formatted

if __name__ == "__main__":
    rag = RegulatoryRAGEngine()
    cits = rag.retrieve_citations("high loan to value ratio and dti debt burden")
    print(rag.format_citation_prompt(cits))
