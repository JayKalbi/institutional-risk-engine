document.addEventListener('DOMContentLoaded', () => {

    // --- Navigation Logic ---
    const navLinks = document.querySelectorAll('.nav-links a');
    const pages = document.querySelectorAll('.page');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            
            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            const targetId = link.getAttribute('data-target');
            
            // Fade out current
            pages.forEach(p => {
                if(p.classList.contains('active')) {
                    p.style.opacity = 0;
                    setTimeout(() => {
                        p.classList.remove('active');
                        p.classList.add('hidden');
                        
                        // Fade in target
                        const targetPage = document.getElementById(targetId);
                        targetPage.classList.remove('hidden');
                        setTimeout(() => {
                            targetPage.classList.add('active');
                            targetPage.style.opacity = 1;
                            
                            // Trigger counters if home
                            if(targetId === 'home') runCounters();
                        }, 50);
                    }, 400); // match css transition
                }
            });
        });
    });

    // --- Counters Animation (Landing Page) ---
    function runCounters() {
        const counters = document.querySelectorAll('.counter');
        counters.forEach(counter => {
            counter.innerText = '0';
            const target = +counter.getAttribute('data-target');
            const speed = 200; // lower = faster
            
            const updateCount = () => {
                const cur = +counter.innerText;
                const inc = target / speed;
                
                if (cur < target) {
                    counter.innerText = (cur + inc).toFixed(target % 1 === 0 ? 0 : 1);
                    setTimeout(updateCount, 10);
                } else {
                    counter.innerText = target;
                }
            };
            updateCount();
        });
    }
    // Run once on initial load
    setTimeout(runCounters, 500);

    // --- Live Terminal API Logic ---
    const form = document.getElementById('underwriting-form');
    const runBtn = document.getElementById('run-btn');
    const btnText = runBtn.querySelector('.btn-text');
    const spinner = runBtn.querySelector('.spinner');
    
    const standbyScreen = document.getElementById('standby-screen');
    const resultsScreen = document.getElementById('results-screen');
    
    // Outputs
    const decisionBox = document.getElementById('decision-box');
    const valPd = document.getElementById('val-pd');
    const valLgd = document.getElementById('val-lgd');
    const valEcl = document.getElementById('val-ecl');
    const shapContainer = document.getElementById('shap-container');
    const memoContent = document.getElementById('memo-content');
    const memoLoader = document.getElementById('memo-loader');
    const navStatus = document.getElementById('nav-status');

    // Basic Token Check
    document.getElementById('hf_token').addEventListener('input', (e) => {
        if(e.target.value.length > 5) {
            navStatus.innerHTML = '🟢 Token Connected';
        } else {
            navStatus.innerHTML = '🔴 Token Missing';
        }
    });

    // LLM Typing Effect
    async function typeText(element, htmlContent) {
        element.innerHTML = '';
        element.classList.add('typing-cursor');
        
        // Very basic implementation: parse HTML so we type text but render tags instantly.
        // For simplicity in this demo, we'll strip complex tags and type character by character,
        // or just type the raw HTML string if we want it fast.
        // To make it look like ChatGPT, we'll simulate a stream.
        
        const chunks = htmlContent.split(/(<[^>]*>)/g);
        
        for (let chunk of chunks) {
            if (chunk.startsWith('<')) {
                // It's an HTML tag, render immediately
                element.insertAdjacentHTML('beforeend', chunk);
            } else {
                // It's text, type it out
                for (let i = 0; i < chunk.length; i++) {
                    element.innerHTML += chunk.charAt(i);
                    await new Promise(r => setTimeout(r, 10)); // 10ms per char
                }
            }
        }
        element.classList.remove('typing-cursor');
    }

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // UI Loading State
        runBtn.disabled = true;
        btnText.textContent = 'EXECUTING NEURAL INFERENCE...';
        spinner.classList.remove('hidden');
        
        standbyScreen.classList.add('hidden');
        resultsScreen.classList.remove('hidden');
        
        decisionBox.className = 'decision-card';
        decisionBox.innerHTML = 'Computing Risk Matrix...';
        valPd.textContent = '--';
        valLgd.textContent = '--';
        valEcl.textContent = '--';
        shapContainer.innerHTML = '';
        memoContent.innerHTML = '';
        memoLoader.classList.remove('hidden');

        // Payload
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
            // 1. Quant Model
            const quantRes = await fetch('/api/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const quantData = await quantRes.json();

            if (quantRes.ok) {
                decisionBox.classList.add(quantData.css_class);
                decisionBox.innerHTML = `${quantData.decision} <span>${quantData.grade_text}</span>`;
                
                const pdPercent = (quantData.pd * 100).toFixed(2);
                valPd.textContent = `${pdPercent}%`;
                if (quantData.pd > 0.35) valPd.classList.add('alert');
                
                valLgd.textContent = `${(quantData.lgd * 100).toFixed(2)}%`;
                valEcl.textContent = `$${quantData.ecl.toLocaleString(undefined, {maximumFractionDigits:0})}`;

                quantData.top_factors.forEach(factor => {
                    const div = document.createElement('div');
                    if (factor.impact > 0) {
                        div.className = 'shap-alert high hover-lift';
                        div.innerHTML = `<span>▲ <b>${factor.feature}</b> elevates default risk.</span> <span>Impact: ${factor.impact.toFixed(4)}</span>`;
                    } else {
                        div.className = 'shap-alert low hover-lift';
                        div.innerHTML = `<span>▼ <b>${factor.feature}</b> mitigates default risk.</span> <span>Impact: ${factor.impact.toFixed(4)}</span>`;
                    }
                    shapContainer.appendChild(div);
                });
            }

            // 2. LLM Narrative
            const narrativeRes = await fetch('/api/narrative', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const narrativeData = await narrativeRes.json();

            memoLoader.classList.add('hidden');
            
            if (narrativeRes.ok) {
                let htmlMemo = narrativeData.narrative
                    .replace(/\*\*(.*?)\*\*/g, '<b>$1</b>')
                    .replace(/\n\n/g, '<br><br>')
                    .replace(/\n- /g, '<br>• ');
                
                // Trigger typing effect
                typeText(memoContent, htmlMemo);
            } else {
                memoContent.innerHTML = `<span style="color:var(--danger)">Error: ${narrativeData.error}</span>`;
            }

        } catch (error) {
            console.error(error);
            decisionBox.classList.add('deny');
            decisionBox.innerHTML = `SYSTEM ERROR <span>Check console for details</span>`;
            memoLoader.classList.add('hidden');
        } finally {
            runBtn.disabled = false;
            btnText.textContent = '▶ RUN HYBRID FUSION ANALYSIS';
            spinner.classList.add('hidden');
        }
    });
});
