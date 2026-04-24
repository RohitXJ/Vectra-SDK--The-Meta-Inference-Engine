const API_BASE = window.location.origin;

const state = {
    sessionToken: null,
    classes: [],
    currentView: 1,
    maxView: 4,
    retriesLeft: 3
};

const dom = {
    // Nav
    btnNext: document.getElementById('btnNext'),
    btnBack: document.getElementById('btnBack'),
    indicators: [1,2,3,4].map(i => document.getElementById(`indicator-${i}`)),
    panels: [1,2,3,4].map(i => document.getElementById(`view-${i}`)),
    
    // View 1
    numClasses: document.getElementById('numClasses'),
    incClasses: document.getElementById('incClasses'),
    decClasses: document.getElementById('decClasses'),
    
    // View 2
    classesGrid: document.getElementById('classesGrid'),
    classCardTemplate: document.getElementById('classCardTemplate'),
    
    // View 3
    trainingStatus: document.getElementById('trainingStatus'),
    statusText: document.getElementById('statusText'),
    statusSubtext: document.getElementById('statusSubtext'),
    progressBar: document.getElementById('progressBar'),
    modelOptions: document.querySelectorAll('input[name="backbone"]'),
    useUnknown: document.getElementById('useUnknown'),
    
    // View 4
    resAccuracy: document.getElementById('resAccuracy'),
    resBackbone: document.getElementById('resBackbone'),
    resClassesCount: document.getElementById('resClassesCount'),
    predictionsTableBody: document.getElementById('predictionsTableBody'),
    downloadBtn: document.getElementById('downloadBtn'),
    resetBtn: document.getElementById('resetBtn'),
    retrainBtn: document.getElementById('retrainBtn'),
    retriesLeftText: document.getElementById('retriesLeftText'),

    // Live Eval
    evalDropZone: document.getElementById('evalDropZone'),
    evalInput: document.getElementById('evalInput'),
    evalPreview: document.getElementById('evalPreview'),
    evalResult: document.getElementById('evalResult'),
    evalLabel: document.getElementById('evalLabel')
};

// INITIALIZATION
async function init() {
    createToastContainer();
    
    try {
        const response = await fetch(`${API_BASE}/session/new`, { method: 'POST' });
        if (!response.ok) throw new Error();
        const data = await response.json();
        state.sessionToken = data.token;
        console.log("SPA Engine Initialized:", state.sessionToken);
    } catch {
        showToast("Backend connection failed.", "error");
    }

    setupEventListeners();
    let initialMin = document.getElementById('useUnknown').checked ? 1 : 2;
    dom.numClasses.value = initialMin;
    updateClassCards(initialMin); 
    
    if (window.lucide) {
        lucide.createIcons();
    }
}

function createToastContainer() {
    const c = document.createElement('div');
    c.id = 'toast-container';
    document.body.appendChild(c);
}

// EVENT LISTENERS
function setupEventListeners() {
    // V1 Counters
    dom.incClasses.addEventListener('click', () => {
        let v = parseInt(dom.numClasses.value);
        if(v < 10) { dom.numClasses.value = v + 1; updateClassCards(v + 1); }
    });
    // Checkbox toggle logic
    dom.useUnknown = document.getElementById('useUnknown');
    if(dom.useUnknown) {
        dom.useUnknown.addEventListener('change', (e) => {
            let minC = e.target.checked ? 1 : 2;
            dom.numClasses.min = minC;
            if(state.classes.length < minC) {
                dom.numClasses.value = minC;
                updateClassCards(minC);
            }
        });
    }

    dom.decClasses.addEventListener('click', () => {
        let v = parseInt(dom.numClasses.value);
        let minC = dom.useUnknown.checked ? 1 : 2;
        if(v > minC) { dom.numClasses.value = v - 1; updateClassCards(v - 1); }
    });

    // Navigation
    dom.btnNext.addEventListener('click', handleNext);
    dom.btnBack.addEventListener('click', handleBack);
    
    // V4 Download
    dom.downloadBtn.addEventListener('click', () => {
        if(state.sessionToken) window.open(`${API_BASE}/download/${state.sessionToken}`, '_blank');
    });

    // Retrain Flow
    dom.retrainBtn.addEventListener('click', () => {
        if(state.retriesLeft > 0) {
            state.retriesLeft -= 1;
            dom.retriesLeftText.textContent = state.retriesLeft;
            
            // Navigate back to architecture selection
            navigateTo(3);
            
            // Reset Training UI state
            dom.btnNext.disabled = false;
            dom.btnBack.disabled = false;
            dom.btnNext.innerHTML = 'Initialize Fast-Train <i data-lucide="zap" class="w-5 h-5"></i>';
            if (window.lucide) lucide.createIcons();
            dom.trainingStatus.classList.add('hidden');
            document.querySelector('#view-3 .grid').classList.remove('opacity-30', 'pointer-events-none');
            
        } else {
            const confirmed = window.confirm("Max retries reached. To retrain, you must start a new session and re-upload your images. Proceed to Home?");
            if(confirmed) {
                dom.resetBtn.click();
            }
        }
    });

    // Live Eval
    dom.evalInput.addEventListener('change', e => {
        if(e.target.files && e.target.files[0]) runLiveEval(e.target.files[0]);
    });

    // Reset Flow
    dom.resetBtn.addEventListener('click', async () => {
        if(state.sessionToken) {
            try {
                await fetch(`${API_BASE}/session/${state.sessionToken}`, { method: 'DELETE' });
            } catch(e) { console.warn("Failed to delete session", e) }
            state.sessionToken = null; // prevents beacon on unload
        }
        window.location.reload();
    });

    window.addEventListener('beforeunload', () => {
        if (state.sessionToken) navigator.sendBeacon(`${API_BASE}/session/${state.sessionToken}`);
    });
}

// VIEW ROUTING
function navigateTo(targetView) {
    if(targetView < 1 || targetView > state.maxView) return;
    
    const isForward = targetView > state.currentView;
    
    // Hide current
    const currentPanel = dom.panels[state.currentView - 1];
    currentPanel.classList.remove('active');
    currentPanel.classList.add(isForward ? 'exit' : 'exit-reverse');
    
    // Prep target
    const targetPanel = dom.panels[targetView - 1];
    targetPanel.classList.remove('exit', 'exit-reverse', 'enter-reverse');
    if(!isForward) targetPanel.classList.add('enter-reverse');
    
    // Execute transition
    setTimeout(() => {
        currentPanel.classList.remove('exit', 'exit-reverse');
        targetPanel.classList.remove('enter-reverse');
        targetPanel.classList.add('active');
    }, 50); // slight delay to allow CSS to snap starting positions

    state.currentView = targetView;
    updateNavUI();
}

function updateNavUI() {
    // Buttons
    dom.btnNext.classList.remove('hidden');
    
    if(state.currentView === 1) {
        dom.btnBack.classList.add('hidden');
        dom.btnNext.innerHTML = 'Define Classes <i data-lucide="chevron-right" class="w-5 h-5"></i>';
    } else if(state.currentView === 2) {
        dom.btnBack.classList.remove('hidden');
        dom.btnNext.innerHTML = 'Proceed to Model <i data-lucide="chevron-right" class="w-5 h-5"></i>';
    } else if(state.currentView === 3) {
        dom.btnBack.classList.remove('hidden');
        dom.btnNext.innerHTML = 'Initialize Fast-Train <i data-lucide="zap" class="w-5 h-5"></i>';
    } else {
        dom.btnBack.classList.add('hidden');
        dom.btnNext.classList.add('hidden'); // No next on results.
    }

    if (window.lucide) lucide.createIcons();

    // Indicators
    dom.indicators.forEach((ind, i) => {
        const viewIdx = i + 1;
        ind.classList.remove('active', 'completed');
        if(viewIdx === state.currentView) ind.classList.add('active');
        else if(viewIdx < state.currentView) ind.classList.add('completed');
    });
}

async function handleNext() {
    // Validation before moving forward
    if(state.currentView === 1) {
        navigateTo(2);
    } 
    else if(state.currentView === 2) {
        // Validate files
        let sCount = -1, qCount = -1;
        for (const cls of state.classes) {
            if (!cls.name.trim()) { showToast("All classes need names.", "error"); return; }
            if (cls.supportFiles.length === 0 || cls.queryFiles.length === 0) { showToast(`Missing images in ${cls.name}`, "error"); return; }
            if (sCount === -1) sCount = cls.supportFiles.length;
            if (qCount === -1) qCount = cls.queryFiles.length;
            if (cls.supportFiles.length !== sCount || cls.queryFiles.length !== qCount) { showToast("Unbalanced datasets detected.", "error"); return; }
        }
        navigateTo(3);
    }
    else if(state.currentView === 3) {
        // Run Train
        runTrainingPipeline();
    }
}

function handleBack() {
    navigateTo(state.currentView - 1);
}

// CLASS DOM LOGIC (INGESTION VIEW)
function updateClassCards(targetCount) {
    const cur = state.classes.length;
    if (targetCount > cur) {
        for (let i = cur; i < targetCount; i++) {
            const classObj = { id: Math.random().toString(36).substr(2,9), name: `Class-${i + 1}`, supportFiles: [], queryFiles: [] };
            state.classes.push(classObj);
            renderClassCard(classObj, i);
        }
    } else if (targetCount < cur) {
        const removed = state.classes.splice(targetCount, cur - targetCount);
        removed.forEach(c => document.getElementById(`card-${c.id}`)?.remove());
    }
    
    // Update indices
    setTimeout(() => {
        document.querySelectorAll('.class-index').forEach((el, i) => el.textContent = i + 1);
        if (window.lucide) lucide.createIcons();
    }, 10);
}

function renderClassCard(classObj, index) {
    const tpl = dom.classCardTemplate.content.cloneNode(true);
    const card = tpl.querySelector('.class-card');
    card.id = `card-${classObj.id}`;
    
    const nameInput = card.querySelector('.class-name-input');
    nameInput.value = classObj.name;
    nameInput.addEventListener('input', e => classObj.name = e.target.value);

    // File zones
    setupZone(card.querySelector('.support-zone'), classObj, 'support', card.querySelector('.count-support'));
    setupZone(card.querySelector('.query-zone'), classObj, 'query', card.querySelector('.count-query'));
    
    // Remove Btn
    card.querySelector('.remove-class-btn').addEventListener('click', () => {
        let minC = document.getElementById('useUnknown').checked ? 1 : 2;
        if(state.classes.length <= minC) { showToast(`Min ${minC} classes.`, "error"); return; }
        state.classes = state.classes.filter(c => c.id !== classObj.id);
        dom.numClasses.value = state.classes.length;
        card.remove();
        document.querySelectorAll('.class-index').forEach((el, i) => el.textContent = i + 1);
        if (window.lucide) lucide.createIcons();
    });

    dom.classesGrid.appendChild(card);
}

function setupZone(el, obj, type, countEl) {
    const input = el.querySelector('.file-input');
    ['dragenter', 'dragover'].forEach(e => el.addEventListener(e, ev => { ev.preventDefault(); el.classList.add('drag-over'); }));
    ['dragleave', 'drop'].forEach(e => el.addEventListener(e, ev => { ev.preventDefault(); el.classList.remove('drag-over'); }));
    
    el.addEventListener('drop', e => handle(e.dataTransfer.files));
    input.addEventListener('change', e => handle(e.target.files));

    function handle(files) {
        const valid = Array.from(files).filter(f => f.type.startsWith('image/'));
        if(valid.length < files.length) showToast("Images only", "error");
        
        if (type === 'support') obj.supportFiles.push(...valid);
        else obj.queryFiles.push(...valid);
        
        countEl.textContent = `${type==='support'?obj.supportFiles.length:obj.queryFiles.length} items added`;
        countEl.classList.add('text-primary-600', 'font-bold'); // highlight
    }
}

// TRAINING PIPELINE
async function runTrainingPipeline() {
    dom.btnNext.disabled = true;
    dom.btnBack.disabled = true;
    dom.btnNext.innerHTML = '<span class="loader-accent w-5 h-5 border-2 inline-block"></span> Processing...';
    
    // UI prep
    dom.trainingStatus.classList.remove('hidden');
    // Hide model options to focus on loading
    document.querySelector('#view-3 .grid').classList.add('opacity-30', 'pointer-events-none');

    try {
        const totalReqs = state.classes.length * 2;
        let cReqs = 0;

        // 1. Upload
        for (const cls of state.classes) {
            updateUploadStatus(`Ingesting ${cls.name} Support Data`, (cReqs/totalReqs)*50);
            await uploadData(cls.name, 'support', cls.supportFiles);
            cReqs++;

            updateUploadStatus(`Ingesting ${cls.name} Query Data`, (cReqs/totalReqs)*50);
            await uploadData(cls.name, 'query', cls.queryFiles);
            cReqs++;
        }

        // 2. Train
        updateUploadStatus("Compiling Graph & Training...", 75);
        const backbone = document.querySelector('input[name="backbone"]:checked').value;
        const useUnknown = dom.useUnknown.checked;

        const fd = new FormData();
        fd.append("token", state.sessionToken);
        fd.append("backbone_name", backbone);
        fd.append("use_unknown", useUnknown);

        const res = await fetch(`${API_BASE}/train`, { method: 'POST', body: fd });
        if(!res.ok) throw new Error("Training API failure");
        const results = await res.json();
        
        updateUploadStatus("Finalizing Weights...", 100);
        
        // 3. Render Results & Navigate
        setTimeout(() => {
            renderResults(results, backbone);
            navigateTo(4);
        }, 800);

    } catch(err) {
        showToast(err.message, "error");
        document.querySelector('#view-3 .grid').classList.remove('opacity-30', 'pointer-events-none');
        dom.trainingStatus.classList.add('hidden');
        dom.btnNext.disabled = false;
        dom.btnBack.disabled = false;
        dom.btnNext.innerHTML = 'Initialize Fast-Train <i data-lucide="zap" class="w-5 h-5"></i>';
        if (window.lucide) lucide.createIcons();
    }
}

async function uploadData(cName, cat, files) {
    const fd = new FormData();
    fd.append("token", state.sessionToken);
    fd.append("category", cat);
    fd.append("class_name", cName);
    files.forEach(f => fd.append("files", f));
    const r = await fetch(`${API_BASE}/upload`, { method:'POST', body: fd });
    if(!r.ok) throw new Error(`Upload failed for ${cName}`);
}

function updateUploadStatus(txt, prog) {
    dom.statusText.textContent = txt;
    dom.progressBar.style.width = `${prog}%`;
}

// LIVE EVALUATION
async function runLiveEval(file) {
    // UI Update
    dom.evalResult.classList.remove('hidden');
    dom.evalLabel.textContent = "Processing...";
    dom.evalLabel.className = "text-xl font-display font-bold text-slate-400";
    
    // Preview
    const reader = new FileReader();
    reader.onload = e => {
        dom.evalPreview.classList.remove('hidden');
        dom.evalPreview.querySelector('img').src = e.target.result;
    };
    reader.readAsDataURL(file);

    try {
        const fd = new FormData();
        fd.append("token", state.sessionToken);
        fd.append("file", file);

        const res = await fetch(`${API_BASE}/eval`, { method: 'POST', body: fd });
        if(!res.ok) throw new Error("Eval failed");
        const data = await res.json();

        dom.evalLabel.textContent = data.prediction;
        if(data.prediction === "Unknown") {
            dom.evalLabel.className = "text-xl font-display font-bold text-red-600";
        } else {
            dom.evalLabel.className = "text-xl font-display font-bold text-primary-600";
        }
    } catch(err) {
        dom.evalLabel.textContent = "Error";
        showToast("Live evaluation failed.", "error");
    }
}

// RESULTS DOM
function renderResults(data, backbone) {
    dom.resAccuracy.textContent = data.accuracy;
    dom.resClassesCount.textContent = data.labels.length;
    dom.resBackbone.textContent = backbone.replace('_', ' ');

    const accGauge = document.getElementById('accuracyGaugeCircle');
    if(accGauge) {
        const numAccuracy = parseFloat(data.accuracy);
        setTimeout(() => {
            accGauge.setAttribute('stroke-dasharray', `${numAccuracy}, 100`);
        }, 300);
    }

    dom.predictionsTableBody.innerHTML = '';
    const actuals = data.true_labels;
    const preds = data.predicted_labels;
    const lMap = data.labels;

    // Reset Live Eval UI
    dom.evalResult.classList.add('hidden');
    dom.evalPreview.classList.add('hidden');
    dom.evalInput.value = "";

    for (let i = 0; i < actuals.length; i++) {
        const tr = document.createElement('tr');
        const isC = actuals[i] === preds[i];
        const aName = lMap[actuals[i]] || actuals[i];
        
        let pName;
        if (preds[i] >= lMap.length) {
            pName = "Unknown";
        } else {
            pName = lMap[preds[i]] || preds[i];
        }

        tr.innerHTML = `
            <td class="py-3 px-2 text-slate-500">${i + 1}</td>
            <td class="py-3 px-2 font-medium">${aName}</td>
            <td class="py-3 px-2 font-bold ${isC ? 'text-emerald-600' : 'text-red-600'}">${pName}</td>
            <td class="py-3 px-2 text-right">
                ${isC ? `<span class="bg-emerald-100 text-emerald-700 font-bold px-2 py-1 rounded text-xs">PASS</span>` 
                      : `<span class="bg-red-100 text-red-700 font-bold px-2 py-1 rounded text-xs">FAIL</span>`}
            </td>
        `;
        dom.predictionsTableBody.appendChild(tr);
    }
}

// UTILS
function showToast(msg, type="success") {
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.innerHTML = `<span>${msg}</span>`;
    document.getElementById('toast-container').appendChild(t);
    requestAnimationFrame(() => t.classList.add('show'));
    setTimeout(() => { t.classList.remove('show'); setTimeout(()=>t.remove(), 400); }, 3000);
}

init();
