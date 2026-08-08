// Local dev (vite dev server) talks to the backend directly on :4000; the
// production build is served by nginx on the same origin/port that proxies
// /api/ to the backend, so it uses a relative base.
const BASE = import.meta.env.VITE_API_BASE ?? (import.meta.env.DEV ? "http://localhost:4000" : "");

async function req(path, opts) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const err = new Error(data.error || (data.errors || []).join("; ") || res.statusText);
    err.details = data;
    throw err;
  }
  return data;
}

export const api = {
  schema: () => req("/api/schema"),
  comments: (annotator) => req(`/api/comments?annotator=${encodeURIComponent(annotator)}`),
  comment: (id, annotator) => req(`/api/comments/${id}?annotator=${encodeURIComponent(annotator)}`),
  saveLabel: (id, body) => req(`/api/labels/${id}`, { method: "PUT", body: JSON.stringify(body) }),
  progress: () => req("/api/progress"),

  adjudicationList: () => req("/api/adjudication"),
  adjudicationProgress: () => req("/api/adjudication/progress"),
  adjudicationItem: (id) => req(`/api/adjudication/${id}`),
  saveAdjudication: (id, body) =>
    req(`/api/adjudication/${id}`, { method: "PUT", body: JSON.stringify(body) }),
  postMessage: (id, annotator, body) =>
    req(`/api/adjudication/${id}/messages`, { method: "POST", body: JSON.stringify({ annotator, body }) }),
  messages: (id) => req(`/api/adjudication/${id}/messages`),
  exportGold: () => req("/api/adjudication/export", { method: "POST" }),
};
