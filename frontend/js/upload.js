/**
 * upload.js — PlantDoc
 *
 * Handles image upload UX, calls API.predict + API.recommendations,
 * and renders the result panel.
 */

/* ── DOM references ──────────────────────────────────────────────────────── */
const uploadZone        = document.getElementById('uploadZone');
const fileInput         = document.getElementById('fileInput');
const previewBox        = document.getElementById('previewBox');
const previewImg        = document.getElementById('previewImg');
const removeBtn         = document.getElementById('removeImageBtn');
const predictBtn        = document.getElementById('predictBtn');
const loadingState      = document.getElementById('loadingState');
const resultPlaceholder = document.getElementById('resultPlaceholder');
const resultContent     = document.getElementById('resultContent');

const ALLOWED_TYPES = new Set(['image/jpeg', 'image/jpg', 'image/png', 'image/webp']);
const MAX_SIZE_MB   = 10;

let selectedFile = null;

/* ── File selection ──────────────────────────────────────────────────────── */

if (uploadZone) {
  uploadZone.addEventListener('click',   () => fileInput && fileInput.click());
  uploadZone.addEventListener('keydown', (e) => { if (e.key === 'Enter' || e.key === ' ') fileInput && fileInput.click(); });
  uploadZone.addEventListener('dragover',  (e) => { e.preventDefault(); uploadZone.classList.add('drag-over'); });
  uploadZone.addEventListener('dragleave', ()  => uploadZone.classList.remove('drag-over'));
  uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFileSelect(file);
  });
}

if (fileInput) {
  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) handleFileSelect(fileInput.files[0]);
  });
}

function handleFileSelect(file) {
  if (!ALLOWED_TYPES.has(file.type)) {
    showToast('Please upload a JPG, PNG, or WebP image.', 'error');
    return;
  }
  if (file.size > MAX_SIZE_MB * 1024 * 1024) {
    showToast(`Image must be smaller than ${MAX_SIZE_MB} MB.`, 'error');
    return;
  }

  selectedFile = file;

  const reader = new FileReader();
  reader.onload = (e) => {
    if (previewImg) previewImg.src = e.target.result;
    if (previewBox) previewBox.style.display = 'block';
    if (uploadZone) uploadZone.style.display = 'none';
    if (predictBtn) predictBtn.disabled = false;
  };
  reader.readAsDataURL(file);
  _showPlaceholder();
}

if (removeBtn) {
  removeBtn.addEventListener('click', () => {
    selectedFile = null;
    if (previewImg) previewImg.src = '';
    if (previewBox) previewBox.style.display = 'none';
    if (uploadZone) uploadZone.style.display = 'block';
    if (predictBtn) predictBtn.disabled = true;
    if (fileInput)  fileInput.value = '';
    _showPlaceholder();
  });
}

/* ── Predict ─────────────────────────────────────────────────────────────── */

if (predictBtn) predictBtn.addEventListener('click', runPrediction);

async function runPrediction() {
  if (!selectedFile) { showToast('Please select an image first.', 'error'); return; }

  _showLoading(true);

  try {
    const prediction = await API.predict(selectedFile);

    // Non-leaf rejection (HTTP 400 with status=rejected)
    if (prediction.status === 'rejected') {
      _showLoading(false);
      showToast(i18n.t('notLeafToast'), 'error');
      _renderRejected(prediction.message || i18n.t('notLeafMsg'));
      return;
    }

    const disease    = prediction.disease    || 'Unknown';
    const confidence = prediction.confidence ?? 0;

    // Clean display name: replace ___ and _ with spaces, title-case
    const displayDisease = disease
      .replace(/___/g, ' — ')
      .replace(/_/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase());

    // Fetch recommendations with current language (non-fatal if it fails)
    const lang = (typeof i18n !== 'undefined') ? i18n.currentLang() : 'en';
    let reco = null;
    try { reco = await API.recommendations(disease, lang); } catch { /* fallback handled below */ }

    _renderResult(displayDisease, confidence, reco);

  } catch (err) {
    _showLoading(false);
    showToast(err.message, 'error');
    _renderError(err.message);
  }
}

/* ── Render: successful prediction ───────────────────────────────────────── */

function _renderResult(disease, confidence, reco) {
  _showLoading(false);
  _restoreSections();

  const isHealthy = disease.toLowerCase().includes('healthy');
  const nameEl    = document.getElementById('diseaseName');
  if (nameEl) {
    nameEl.textContent = disease;
    nameEl.parentElement.className = `disease-badge ${isHealthy ? 'healthy' : 'detected'}`;
  }

  _setConfidenceBar(confidence);

  if (reco) {
    _renderTreatment('pesticideContent',  reco.pesticide  || reco.pesticide_recommendation);
    _renderTreatment('organicContent',    reco.organic    || reco.organic_treatment);
    _renderTreatment('preventionContent', reco.prevention || reco.prevention_methods);
    _renderYouTubeLinks(reco.youtube_links || reco.youtube || []);
  } else {
    _setHtml('pesticideContent', '<p class="text-muted">Recommendations not available.</p>');
    _setHtml('organicContent',   '');
    _setHtml('preventionContent','');
    _renderYouTubeLinks([]);
  }

  if (resultPlaceholder) resultPlaceholder.classList.add('hidden');
  if (resultContent)     resultContent.classList.add('visible');

  showToast(
    `${i18n.t('detectedToast')}: ${disease} (${confidence.toFixed(1)}%)`,
    isHealthy ? 'success' : 'error'
  );
}

/* ── Render: non-leaf rejected ───────────────────────────────────────────── */

function _renderRejected(message) {
  if (resultPlaceholder) resultPlaceholder.classList.add('hidden');
  if (!resultContent) return;
  resultContent.classList.add('visible');

  const nameEl = document.getElementById('diseaseName');
  if (nameEl) {
    nameEl.textContent = i18n.t('notLeaf');
    nameEl.parentElement.className = 'disease-badge detected';
  }

  _hideConfidenceBar();
  _hideTreatmentSections();

  _setHtml('pesticideContent', `
    <div style="text-align:center; padding:24px 0;">
      <div style="font-size:3.5rem; margin-bottom:14px;">🍂</div>
      <p style="color:var(--danger); font-size:1rem; font-weight:600; margin-bottom:10px;">${message}</p>
      <p style="color:var(--cream); font-size:0.875rem; opacity:0.75; line-height:1.6;">
        ${i18n.t('notLeafMsg')}
      </p>
    </div>
  `);
}

/* ── Render: network / server error ─────────────────────────────────────── */

function _renderError(message) {
  if (resultPlaceholder) resultPlaceholder.classList.add('hidden');
  if (!resultContent) return;
  resultContent.classList.add('visible');

  const nameEl = document.getElementById('diseaseName');
  if (nameEl) {
    nameEl.textContent = i18n.t('analysisFailed');
    nameEl.parentElement.className = 'disease-badge detected';
  }

  _hideConfidenceBar();
  _hideTreatmentSections();

  _setHtml('pesticideContent', `
    <div style="text-align:center; padding:24px 0;">
      <div style="font-size:3rem; margin-bottom:12px;">⚠️</div>
      <p style="color:var(--danger); font-size:0.95rem; font-weight:600;">${message}</p>
    </div>
  `);
}

/* ── UI helpers ──────────────────────────────────────────────────────────── */

function _setConfidenceBar(confidence) {
  const bar   = document.querySelector('.confidence-bar');
  const fill  = document.getElementById('confFill');
  const label = document.getElementById('confLabel');
  if (bar)   bar.style.display = '';
  if (label) label.textContent = `${confidence.toFixed(1)}%`;
  if (fill)  requestAnimationFrame(() => { fill.style.width = `${Math.min(confidence, 100)}%`; });
}

function _hideConfidenceBar() {
  const bar   = document.querySelector('.confidence-bar');
  const fill  = document.getElementById('confFill');
  const label = document.getElementById('confLabel');
  if (bar)   bar.style.display = 'none';
  if (fill)  fill.style.width  = '0%';
  if (label) label.textContent = '';
}

function _hideTreatmentSections() {
  if (!resultContent) return;
  const divider = resultContent.querySelector('.divider');
  if (divider) divider.style.display = 'none';
  resultContent.querySelectorAll('.treatment-section, .youtube-section')
    .forEach((el, i) => { el.style.display = i === 0 ? 'block' : 'none'; });
  const header = document.querySelector('#pesticideContent')?.previousElementSibling;
  if (header?.classList.contains('treatment-header')) header.style.display = 'none';
}

function _restoreSections() {
  if (!resultContent) return;
  const divider = resultContent.querySelector('.divider');
  if (divider) divider.style.display = '';
  resultContent.querySelectorAll('.treatment-section, .youtube-section')
    .forEach(el => el.style.display = '');
  const header = document.querySelector('#pesticideContent')?.previousElementSibling;
  if (header?.classList.contains('treatment-header')) header.style.display = '';
}

function _renderTreatment(elId, content) {
  const el = document.getElementById(elId);
  if (!el) return;
  if (!content) { el.innerHTML = '<p class="text-muted">–</p>'; return; }

  if (typeof content === 'string') {
    el.innerHTML = `<p>${content}</p>`;
    return;
  }

  let html = '';
  if (content.name || content.product)
    html += `<strong style="color:var(--cream)">${content.name || content.product}</strong><br>`;
  if (content.description)
    html += `<p>${content.description}</p>`;
  if (Array.isArray(content.steps) && content.steps.length)
    html += '<ul>' + content.steps.map(s => `<li>${s}</li>`).join('') + '</ul>';
  el.innerHTML = html || JSON.stringify(content);
}

function _renderYouTubeLinks(links) {
  const container = document.getElementById('youtubeLinks');
  if (!container) return;
  if (!links || links.length === 0) {
    container.innerHTML = '<p class="text-muted" style="font-size:13px">No video links available.</p>';
    return;
  }
  container.innerHTML = links.map(link => {
    const url   = typeof link === 'string' ? link : (link.url || '#');
    const title = typeof link === 'object'  ? (link.title || 'Watch on YouTube') : 'Watch on YouTube';
    return `<a href="${url}" target="_blank" rel="noopener noreferrer" class="yt-link">
      <span class="yt-icon">▶</span><span>${title}</span>
    </a>`;
  }).join('');
}

function _showPlaceholder() {
  if (resultPlaceholder) resultPlaceholder.classList.remove('hidden');
  if (resultContent)     resultContent.classList.remove('visible');
}

function _showLoading(show) {
  if (loadingState)      loadingState.classList.toggle('visible', show);
  if (resultPlaceholder) resultPlaceholder.classList.toggle('hidden', show);
  if (resultContent && show) resultContent.classList.remove('visible');
  if (predictBtn)        predictBtn.disabled = show;
}

function _setHtml(id, html) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = html;
}