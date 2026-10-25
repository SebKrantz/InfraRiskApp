// Behind the Google sign-in, Caddy answers /api/* with a plain 401 once the
// sign-in cookie has expired (pages get the redirect to Google instead). Reload
// the page once so it goes through sign-in and comes back; if the reloaded page
// still gets a 401, announce it instead of reloading again. In single-user mode
// the API never answers 401, so none of this runs.

const RELOADED = 'aei-signin-reloaded'
export const SESSION_EXPIRED_EVENT = 'aei-session-expired'

let expired = false

function isApi(input: RequestInfo | URL): boolean {
  const raw = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
  const url = new URL(raw, window.location.href)
  return url.origin === window.location.origin && url.pathname.startsWith('/api/')
}

function onUnauthorized() {
  if (expired) return
  expired = true
  let first = false
  try {
    first = sessionStorage.getItem(RELOADED) !== '1'
    if (first) sessionStorage.setItem(RELOADED, '1')
  } catch {
    first = false // no storage, no guard: never reload
  }
  if (first) window.location.reload()
  else window.dispatchEvent(new Event(SESSION_EXPIRED_EVENT))
}

export function installSessionGuard() {
  const original = window.fetch.bind(window)
  window.fetch = async (input, init) => {
    const res = await original(input, init)
    if (isApi(input)) {
      if (res.status === 401) onUnauthorized()
      else if (res.ok) {
        try {
          sessionStorage.removeItem(RELOADED) // signed in again: a later expiry may reload
        } catch {
          // ignore
        }
      }
    }
    return res
  }
}
