const API_BASE = 'http://127.0.0.1:8000';

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
    
    // View 4
    resAccuracy: document.getElementById('resAccuracy'),
    resBackbone: document.getElementById('resBackbone'),
    resClassesCount: document.getElementById('resClassesCount'),
    predictionsTableBody: document.getElementById('predictionsTableBody'),
    downloadBtn: document.getElementById('downloadBtn'),
    resetBtn: document.getElementById('resetBtn'),
    retrainBtn: document.getElementById('retrainBtn'),
    retriesLeftText: document.getElementById('retriesLeftText')
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
    updateClassCards(2); 
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
    dom.decClasses.addEventListener('click', () => {
        let v = parseInt(dom.numClasses.value);
        if(v > 2) { dom.numClasses.value = v - 1; updateClassCards(v - 1); }
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
            dom.btnNext.innerHTML = 'Initialize Fast-Train <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
            dom.trainingStatus.classList.add('hidden');
            document.querySelector('#view-3 .grid').classList.remove('opacity-30', 'pointer-events-none');
            
        } else {
            const confirmed = window.confirm("Max retries reached. To retrain, you must start a new session and re-upload your images. Proceed to Home?");
            if(confirmed) {
                dom.resetBtn.click();
            }
        }
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
        dom.btnNext.innerHTML = 'Define Classes <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>';
    } else if(state.currentView === 2) {
        dom.btnBack.classList.remove('hidden');
        dom.btnNext.innerHTML = 'Proceed to Model <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>';
    } else if(state.currentView === 3) {
        dom.btnBack.classList.remove('hidden');
        dom.btnNext.innerHTML = 'Initialize Fast-Train <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
    } else {
        dom.btnBack.classList.add('hidden');
        dom.btnNext.classList.add('hidden'); // No next on results.
    }

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
        if(state.classes.length <= 2) { showToast("Min 2 classes.", "error"); return; }
        state.classes = state.classes.filter(c => c.id !== classObj.id);
        dom.numClasses.value = state.classes.length;
        card.remove();
        document.querySelectorAll('.class-index').forEach((el, i) => el.textContent = i + 1);
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
        countEl.classList.add('text-accent-400', 'font-bold'); // highlight
    }
}

// TRAINING PIPELINE
async function runTrainingPipeline() {
    dom.btnNext.disabled = true;
    dom.btnBack.disabled = true;
    dom.btnNext.innerHTML = '<span class="loader-accent w-5 h-5 border-2"></span> Processing...';
    
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
        const fd = new FormData();
        fd.append("token", state.sessionToken);
        fd.append("backbone_name", backbone);

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
        dom.btnNext.innerHTML = 'Initialize Fast-Train <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
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

// RESULTS DOM
function renderResults(data, backbone) {
    dom.resAccuracy.textContent = data.accuracy;
    dom.resClassesCount.textContent = data.labels.length;
    dom.resBackbone.textContent = backbone.replace('_', ' ');

    dom.predictionsTableBody.innerHTML = '';
    const actuals = data.true_labels;
    const preds = data.predicted_labels;
    const lMap = data.labels;

    for (let i = 0; i < actuals.length; i++) {
        const tr = document.createElement('tr');
        const isC = actuals[i] === preds[i];
        const aName = lMap[actuals[i]] || actuals[i];
        const pName = lMap[preds[i]] || preds[i];

        tr.innerHTML = `
            <td class="py-3 px-2 text-white/70">${i + 1}</td>
            <td class="py-3 px-2 font-medium">${aName}</td>
            <td class="py-3 px-2 font-bold ${isC ? 'text-green-400' : 'text-pink-500'}">${pName}</td>
            <td class="py-3 px-2 text-right">
                ${isC ? `<span class="bg-green-500/20 text-green-300 px-2 py-1 rounded text-xs">PASS</span>` 
                      : `<span class="bg-pink-500/20 text-pink-300 px-2 py-1 rounded text-xs">FAIL</span>`}
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
