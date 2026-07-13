document.addEventListener('DOMContentLoaded', () => {

    // --- Navigation Logic ---
    const navLinks = document.querySelectorAll('.nav-links a');
    const pages = document.querySelectorAll('.page');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            // Remove active classes
            navLinks.forEach(l => l.classList.remove('active'));
            pages.forEach(p => p.classList.add('hidden'));
            pages.forEach(p => p.classList.remove('active'));
            
            // Add active class to clicked
            link.classList.add('active');
            const targetId = link.getAttribute('data-target');
            const targetPage = document.getElementById(targetId);
            targetPage.classList.remove('hidden');
            // Small timeout to allow display:block to apply before animating opacity
            setTimeout(() => targetPage.classList.add('active'), 10);
        });
    });

    // --- Form Submission Logic ---
    const form = document.getElementById('underwriting-form');
    const runBtn = document.getElementById('run-btn');
    
    const standbyScreen = document.getElementById('standby-screen');
    const resultsScreen = document.getElementById('results-screen');
    
    // Output elements
    const decisionBox = document.getElementById('decision-box');
    const valPd = document.getElementById('val-pd');
    const valLgd = document.getElementById('val-lgd');
    const valEcl = document.getElementById('val-ecl');
    const shapContainer = document.getElementById('shap-container');
    
    const memoContent = document.getElementById('memo-content');
    const memoLoader = document.getElementById('memo-loader');
    
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // UI Loading State
        runBtn.disabled = true;
        runBtn.textContent = '⏳ RUNNING ANALYSIS...';
        standbyScreen.classList.add('hidden');
        resultsScreen.classList.remove('hidden');
        
        // Reset outputs
        decisionBox.className = 'decision-box';
        decisionBox.innerHTML = 'Computing...';
        valPd.textContent = '--';
        valLgd.textContent = '--';
        valEcl.textContent = '--';
        shapContainer.innerHTML = '';
        memoContent.innerHTML = '';
        memoLoader.classList.remove('hidden');

        // Gather Data
        const payload = {
            loan_amount: document.getElementById('loan_amount').value,
            income: document.getElementById('income').value,
            property_value: document.getElementById('property_value').value,
            debt_to_income_ratio: document.getElementById('dti').value,
            loan_to_value_ratio: document.getElementById('ltv').value,
            loan_term: document.getElementById('term').value,
            age: document.getElementById('age').value,
            sex: document.getElementById('sex').value,
            race: document.getElementById('race').value,
            loan_officer_notes: document.getElementById('narrative_notes').value,
            hf_token: document.getElementById('hf_token').value
        };

        try {
            // 1. Call Quant Model
            const quantRes = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const quantData = await quantRes.json();

            if (quantRes.ok) {
                // Populate Quant Board
                decisionBox.classList.add(quantData.css_class);
                decisionBox.innerHTML = `${quantData.decision} <span>${quantData.grade_text}</span>`;
                
                const pdPercent = (quantData.pd * 100).toFixed(2);
                valPd.textContent = `${pdPercent}%`;
                if (quantData.pd > 0.35) valPd.classList.add('alert');
                else valPd.classList.remove('alert');
                
                valLgd.textContent = `${(quantData.lgd * 100).toFixed(2)}%`;
                valEcl.textContent = `$${quantData.ecl.toLocaleString(undefined, {maximumFractionDigits:0})}`;

                // Populate SHAP
                quantData.top_factors.forEach(factor => {
                    const div = document.createElement('div');
                    if (factor.impact > 0) {
                        div.className = 'shap-alert high';
                        div.innerHTML = `<span>▲ <b>${factor.feature}</b> elevates default risk.</span> <span>Impact: ${factor.impact.toFixed(4)}</span>`;
                    } else {
                        div.className = 'shap-alert low';
                        div.innerHTML = `<span>▼ <b>${factor.feature}</b> mitigates default risk.</span> <span>Impact: ${factor.impact.toFixed(4)}</span>`;
                    }
                    shapContainer.appendChild(div);
                });
            } else {
                decisionBox.classList.add('deny');
                decisionBox.innerHTML = `ERROR <span>${quantData.error}</span>`;
            }

            // 2. Call LLM Narrative Generator
            const narrativeRes = await fetch('/api/narrative', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const narrativeData = await narrativeRes.json();

            memoLoader.classList.add('hidden');
            
            if (narrativeRes.ok) {
                // Basic markdown to HTML (bold and lists)
                let htmlMemo = narrativeData.narrative
                    .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>') // bold
                    .replace(/\n\n/g, '<br><br>') // paragraphs
                    .replace(/\n- /g, '<br>• '); // bullet points
                memoContent.innerHTML = htmlMemo;
            } else {
                memoContent.innerHTML = `<span style="color:var(--danger)">Error generating narrative: ${narrativeData.error}</span>`;
            }

        } catch (error) {
            console.error(error);
            decisionBox.classList.add('deny');
            decisionBox.innerHTML = `SYSTEM ERROR <span>Check console for details</span>`;
            memoLoader.classList.add('hidden');
        } finally {
            runBtn.disabled = false;
            runBtn.textContent = '▶ RUN HYBRID FUSION ANALYSIS';
        }
    });

});
