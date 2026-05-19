/**
 * dashboard.js — PlantDoc
 *
 * Handles:
 *   - Section navigation (home / upload / about)
 *   - Language selector rendering
 *   - Toast notifications (global window.showToast)
 */

/* ── Navigation ──────────────────────────────────────────────────────────── */

const sections = document.querySelectorAll('.page-section');
const navItems  = document.querySelectorAll('.nav-item[data-section]');

const TOPBAR_KEYS = {
  home:   { title: 'topbarHomeTitle',   subtitle: 'topbarHomeSubtitle' },
  upload: { title: 'topbarUploadTitle', subtitle: 'topbarUploadSubtitle' },
  about:  { title: 'topbarAboutTitle',  subtitle: 'topbarAboutSubtitle' },
};

function activateSection(name) {
  sections.forEach(s => s.classList.remove('active'));
  navItems.forEach(n => n.classList.remove('active'));

  const target  = document.getElementById('section-' + name);
  const navItem = document.querySelector(`.nav-item[data-section="${name}"]`);
  if (target)  target.classList.add('active');
  if (navItem) navItem.classList.add('active');

  const keys       = TOPBAR_KEYS[name] || TOPBAR_KEYS.home;
  const titleEl    = document.getElementById('topbarTitle');
  const subtitleEl = document.getElementById('topbarSubtitle');
  if (titleEl)    titleEl.textContent    = i18n.t(keys.title);
  if (subtitleEl) subtitleEl.textContent = i18n.t(keys.subtitle);

  sessionStorage.setItem('activeSection', name);
}

navItems.forEach(item =>
  item.addEventListener('click', () => activateSection(item.getAttribute('data-section')))
);

document.querySelectorAll('[data-goto]').forEach(el =>
  el.addEventListener('click', () => activateSection(el.getAttribute('data-goto')))
);

/* ── Language selector ───────────────────────────────────────────────────── */

const LANGUAGES = [
  { code: 'en', label: 'English', flag: '🇬🇧' },
  { code: 'te', label: 'తెలుగు',  flag: '🇮🇳' },
  { code: 'hi', label: 'हिन्दी',   flag: '🇮🇳' },
  { code: 'ml', label: 'മലയാളം', flag: '🇮🇳' },
  { code: 'ta', label: 'தமிழ்',   flag: '🇮🇳' },
];

function buildLanguageSelector() {
  const container = document.getElementById('langSelector');
  if (!container) return;

  container.innerHTML = LANGUAGES.map(l =>
    `<button class="lang-btn${i18n.currentLang() === l.code ? ' active' : ''}"
             data-lang="${l.code}" title="${l.label}">
       ${l.flag} ${l.label}
     </button>`
  ).join('');

  container.querySelectorAll('.lang-btn').forEach(btn =>
    btn.addEventListener('click', () => {
      i18n.setLanguage(btn.getAttribute('data-lang'));
      buildLanguageSelector(); // re-render to update active state
      activateSection(sessionStorage.getItem('activeSection') || 'home');
    })
  );
}

/* ── Toast notifications ─────────────────────────────────────────────────── */

/**
 * Show a toast message.
 * @param {string} message
 * @param {'info'|'success'|'error'} type
 */
function showToast(message, type = 'info') {
  const container = document.getElementById('toastContainer');
  if (!container) return;

  const ICONS = { success: '✓', error: '✕', info: 'ℹ' };
  const toast  = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `<span>${ICONS[type] ?? 'ℹ'}</span><span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity   = '0';
    toast.style.transform = 'translateY(8px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}

window.showToast = showToast;

/* ── Init ────────────────────────────────────────────────────────────────── */

document.addEventListener('DOMContentLoaded', () => {
  buildLanguageSelector();
  i18n.applyTranslations();
  activateSection(sessionStorage.getItem('activeSection') || 'home');
});
