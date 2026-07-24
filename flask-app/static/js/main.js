document.addEventListener('DOMContentLoaded', () => {
    
    // Sync Groq API key inputs between topbar and form
    const navKeyInput = document.getElementById('hf_token');
    const formKeyInput = document.getElementById('form_groq_key');

    if (navKeyInput && formKeyInput) {
        navKeyInput.addEventListener('input', (e) => formKeyInput.value = e.target.value);
        formKeyInput.addEventListener('input', (e) => navKeyInput.value = e.target.value);
    }

    // Markdown renderer helper function
    function renderMarkdownToHTML(markdownText) {
        if (!markdownText) return '';
        let html = markdownText
            .replace(/^### (.*$)/gim, '<h4 class="memo-h4">$1</h4>')
            .replace(/^## (.*$)/gim, '<h3 class="memo-h3">$1</h3>')
            .replace(/^# (.*$)/gim, '<h2 class="memo-h2">$1</h2>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/^\s*[\-\*]\s+(.*$)/gim, '<li class="memo-li">$1</li>')
            .replace(/\n\n/g, '<br><br>')
            .replace(/\n/g, '<br>');
        return html;
    }

    // =========================================================================
    // NAVIGATION TAB SWITCHING
    // =========================================================================
    const navLinks = document.querySelectorAll('.nav-link');
    const pages = document.querySelectorAll('.page');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const targetId = link.getAttribute('data-target');

            navLinks.forEach(l => l.classList.remove('active'));
            pages.forEach(p => p.classList.remove('active'));

            link.classList.add('active');
            const targetPage = document.getElementById(targetId);
            if (targetPage) {
                targetPage.classList.add('active');
            }
        });
    });

    // Range Sliders Live Updates
    const dtiInput = document.getElementById('debt_to_income_ratio');
    const dtiVal = document.getElementById('dti-val');
    if (dtiInput && dtiVal) {
        dtiInput.addEventListener('input', (e) => dtiVal.textContent = e.target.value);
    }

    const ltvInput = document.getElementById('loan_to_value_ratio');
    const ltvVal = document.getElementById('ltv-val');
    if (ltvInput && ltvVal) {
        ltvInput.addEventListener('input', (e) => ltvVal.textContent = e.target.value);
    }

    // Macro Sliders Live Updates
    const macroIrSlider = document.getElementById('macro-ir-slider');
    const macroIrVal = document.getElementById('macro-ir-val');
    if (macroIrSlider && macroIrVal) {
        macroIrSlider.addEventListener('input', (e) => macroIrVal.textContent = `+${e.target.value}`);
    }

    const macroUnempSlider = document.getElementById('macro-unemp-slider');
    const macroUnempVal = document.getElementById('macro-unemp-val');
    if (macroUnempSlider && macroUnempVal) {
        macroUnempSlider.addEventListener('input', (e) => macroUnempVal.textContent = `+${e.target.value}`);
    }

    const macroHpiSlider = document.getElementById('macro-hpi-slider');
    const macroHpiVal = document.getElementById('macro-hpi-val');
    if (macroHpiSlider && macroHpiVal) {
        macroHpiSlider.addEventListener('input', (e) => macroHpiVal.textContent = `-${e.target.value}.0`);
    }

    // =========================================================================
    // UNDERWRITING FORM SUBMISSION
    // =========================================================================
    const riskForm = document.getElementById('risk-form');
    let lastPredictionData = null;

    if (riskForm) {
        riskForm.addEventListener('submit', async (e) => {
            e.preventDefault();

            const formData = {
                loan_amount: document.getElementById('loan_amount').value,
                income: document.getElementById('income').value,
                property_value: document.getElementById('property_value').value,
                debt_to_income_ratio: document.getElementById('debt_to_income_ratio').value,
                loan_to_value_ratio: document.getElementById('loan_to_value_ratio').value,
                loan_term: document.getElementById('loan_term').value,
                age: document.getElementById('age').value,
                sex: document.getElementById('sex').value,
                race: document.getElementById('race').value,
                underwriter_notes: document.getElementById('underwriter_notes').value
            };

            const decisionText = document.getElementById('decision-text');
            const gradeText = document.getElementById('grade-text');
            const pdVal = document.getElementById('pd-val');
            const lgdVal = document.getElementById('lgd-val');
            const eclVal = document.getElementById('ecl-val');
            const shapContainer = document.getElementById('shap-container');
            const narrativeOutput = document.getElementById('narrative-output');

            decisionText.textContent = "COMPUTING FUSION RISK...";
            decisionText.style.color = "#3b82f6";
            
            try {
                // 1. Fetch Quantitative Predict
                const res = await fetch('/api/predict', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(formData)
                });
                const pred = await res.json();
                lastPredictionData = pred;

                // Update Decision Board
                decisionText.textContent = pred.decision;
                gradeText.textContent = pred.grade_text;
                
                if (pred.css_class === "approve") decisionText.style.color = "#10b981";
                else if (pred.css_class === "review") decisionText.style.color = "#f59e0b";
                else decisionText.style.color = "#ef4444";

                pdVal.textContent = `${(pred.pd * 100).toFixed(2)}%`;
                lgdVal.textContent = `${(pred.lgd * 100).toFixed(2)}%`;
                eclVal.textContent = `$${Math.round(pred.ecl).toLocaleString()}`;

                // Update SHAP List
                if (pred.top_factors && pred.top_factors.length > 0) {
                    shapContainer.innerHTML = pred.top_factors.map(f => {
                        const isPos = f.impact > 0;
                        const sign = isPos ? '▲' : '▼';
                        const colorClass = isPos ? 'text-red' : 'text-green';
                        return `<div class="shap-item">
                            <span>${sign} ${f.feature}</span>
                            <span class="${colorClass}">${f.impact > 0 ? '+' : ''}${f.impact.toFixed(4)}</span>
                        </div>`;
                    }).join('');
                }

                // 2. Fetch Qualitative Narrative
                narrativeOutput.innerHTML = '<p class="placeholder-text">Synthesizing Credit Memorandum (Groq AI / Mistral-7B / RAG)...</p>';
                
                const navKey = document.getElementById('hf_token') ? document.getElementById('hf_token').value : '';
                const formKey = document.getElementById('form_groq_key') ? document.getElementById('form_groq_key').value : '';
                const apiKey = (navKey || formKey || '').trim();

                const narrativeRes = await fetch('/api/narrative', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ ...formData, api_key: apiKey })
                });
                const nar = await narrativeRes.json();
                
                const formattedNarrative = renderMarkdownToHTML(nar.narrative);

                const sourceTag = nar.source ? `<div style="font-size: 0.72rem; color: #06b6d4; font-weight: 700; margin-bottom: 0.75rem; letter-spacing: 0.05em; text-transform: uppercase;">⚡ Engine Source: ${nar.source}</div>` : '';
                
                narrativeOutput.innerHTML = `${sourceTag}<div class="narrative-text">${formattedNarrative}</div>`;

            } catch (err) {
                console.error("Prediction Error:", err);
                decisionText.textContent = "ERROR COMPUTING RISK";
                decisionText.style.color = "#ef4444";
            }
        });
    }

    // =========================================================================
    // MULTI-AGENT SWARM SESSION
    // =========================================================================
    const runAgentBtn = document.getElementById('run-agent-swarm-btn');
    if (runAgentBtn) {
        runAgentBtn.addEventListener('click', async () => {
            const qOut = document.getElementById('agent-quant-output');
            const mOut = document.getElementById('agent-macro-output');
            const cOut = document.getElementById('agent-compliance-output');
            const croOut = document.getElementById('agent-cro-output');

            qOut.innerHTML = '<p class="placeholder-text">Quant Agent evaluating statistical PD...</p>';
            mOut.innerHTML = '<p class="placeholder-text">Macro Agent evaluating CCAR shock...</p>';
            cOut.innerHTML = '<p class="placeholder-text">Compliance Agent auditing ECOA ratios...</p>';
            croOut.innerHTML = '<p class="placeholder-text">CRO synthesizing committee consensus...</p>';

            const pd = lastPredictionData ? lastPredictionData.pd : 0.18;
            const ecl = lastPredictionData ? lastPredictionData.ecl : 24500.0;
            const top_factors = lastPredictionData ? lastPredictionData.top_factors : [{"feature": "Debt To Income Ratio"}];

            try {
                const res = await fetch('/api/multi_agent_committee', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ pd, ecl, top_factors, scenario: 'severely_adverse' })
                });
                const data = await res.json();
                const transcript = data.agent_transcript;

                if (transcript && transcript.length >= 4) {
                    qOut.innerHTML = `<div class="verdict-tag">${transcript[0].verdict}</div><p>${renderMarkdownToHTML(transcript[0].analysis)}</p>`;
                    mOut.innerHTML = `<div class="verdict-tag">${transcript[1].verdict}</div><p>${renderMarkdownToHTML(transcript[1].analysis)}</p>`;
                    cOut.innerHTML = `<div class="verdict-tag">${transcript[2].verdict}</div><p>${renderMarkdownToHTML(transcript[2].analysis)}</p>`;
                    croOut.innerHTML = `<div class="verdict-tag cro">${transcript[3].final_decision}</div><p>${renderMarkdownToHTML(transcript[3].executive_summary)}</p>`;
                }
            } catch (err) {
                console.error("Multi-Agent Error:", err);
            }
        });
    }

    // =========================================================================
    // CCAR MACRO SIMULATOR
    // =========================================================================
    const runMacroBtn = document.getElementById('run-macro-stress-btn');
    if (runMacroBtn) {
        runMacroBtn.addEventListener('click', async () => {
            const ir = parseFloat(document.getElementById('macro-ir-slider').value);
            const unemp = parseFloat(document.getElementById('macro-unemp-slider').value);
            const hpi = parseFloat(document.getElementById('macro-hpi-slider').value);
            const basePd = lastPredictionData ? lastPredictionData.pd : 0.18;

            try {
                const res = await fetch('/api/macro_stress', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        baseline_pd: basePd,
                        custom_shocks: {
                            interest_rate_hike_bps: ir,
                            unemployment_spike_pct: unemp,
                            hpi_drop_pct: hpi
                        }
                    })
                });
                const data = await res.json();

                document.getElementById('macro-base-pd').textContent = `${(data.baseline_pd * 100).toFixed(1)}%`;
                document.getElementById('macro-stressed-pd').textContent = `${(data.stressed_pd * 100).toFixed(1)}%`;
                
                const shiftSign = data.pd_delta_pct > 0 ? '+' : '';
                document.getElementById('macro-pd-shift').textContent = `${shiftSign}${data.pd_delta_pct.toFixed(1)}%`;
                document.getElementById('macro-scenario-name').textContent = `Scenario: ${data.scenario_name}`;
                document.getElementById('macro-analysis-text').textContent = `Vasicek systematic risk factor Z calculated at ${data.systematic_factor_z.toFixed(2)}. Fed Rate shock of +${ir}bps and unemployment shock of +${unemp}% increases conditional default risk.`;

            } catch (err) {
                console.error("Macro Stress Error:", err);
            }
        });
    }

    // =========================================================================
    // DOCUMENT FRAUD VERIFIER
    // =========================================================================
    const runDocBtn = document.getElementById('run-doc-verify-btn');
    if (runDocBtn) {
        runDocBtn.addEventListener('click', async () => {
            const appInc = document.getElementById('doc-app-income').value;
            const w2Inc = document.getElementById('doc-w2-income').value;
            const taxInc = document.getElementById('doc-tax-income').value;
            const outDiv = document.getElementById('doc-audit-output');

            outDiv.innerHTML = '<p class="placeholder-text">Auditing W-2 and Tax Return extractions...</p>';

            try {
                const res = await fetch('/api/verify_documents', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ income: appInc, w2_income: w2Inc, tax_income: taxInc })
                });
                const data = await res.json();

                const badgeClass = data.discrepancy_flag ? 'fraud-badge-red' : 'fraud-badge-green';
                outDiv.innerHTML = `
                    <div class="fraud-status-badge ${badgeClass}">${data.fraud_risk_level}</div>
                    <div class="audit-details">
                        <p><strong>Self-Reported Income:</strong> $${data.application_income.toLocaleString()}</p>
                        <p><strong>Verified Document Average:</strong> $${data.verified_document_income.toLocaleString()}</p>
                        <p><strong>Income Variance:</strong> ${data.discrepancy_pct.toFixed(1)}%</p>
                        <p class="audit-note">${data.audit_note}</p>
                    </div>
                `;
            } catch (err) {
                console.error("Document Verifier Error:", err);
            }
        });
    }
});
