/**
 * api.js — PlantDoc API client
 *
 * Centralises all HTTP calls to the FastAPI backend.
 * Exposes a single global `API` object used by upload.js.
 */

const API_BASE = (() => {
  // Allow override via ?api= query param (useful for staging/prod)
  const params = new URLSearchParams(window.location.search);
  return params.get('api') || 'http://127.0.0.1:8000';
})();

/**
 * POST a leaf image file to /predict.
 * @param {File} fileBlob
 * @returns {Promise<object>} prediction response
 */
async function apiPredict(fileBlob) {
  const form = new FormData();
  form.append('file', fileBlob);

  let res, data;
  try {
    res  = await fetch(`${API_BASE}/predict`, { method: 'POST', body: form });
    data = await res.json();
  } catch {
    throw new Error('Cannot connect to server. Is the backend running at ' + API_BASE + '?');
  }

  // HTTP 400 with status=rejected → non-leaf: return as-is for upload.js to handle
  if (res.status === 400 && data.status === 'rejected') return data;

  if (!res.ok) {
    throw new Error(data.detail || data.error || `HTTP ${res.status}`);
  }

  return data;
}

/**
 * POST a disease name to /recommendations.
 * @param {string} diseaseName
 * @param {string} [language] - BCP-47 language code e.g. 'hi', 'te', 'en'
 * @returns {Promise<object>} recommendations response
 */
async function apiRecommendations(diseaseName, language) {
  let res, data;
  try {
    res = await fetch(`${API_BASE}/recommendations`, {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({
        disease_name: diseaseName,
        language: language || 'en',
      }),
    });
    data = await res.json();
  } catch {
    throw new Error('Failed to fetch recommendations.');
  }

  if (!res.ok) {
    throw new Error(data.detail || data.error || `HTTP ${res.status}`);
  }

  return data;
}

window.API = {
  base:            API_BASE,
  predict:         apiPredict,
  recommendations: apiRecommendations,
};