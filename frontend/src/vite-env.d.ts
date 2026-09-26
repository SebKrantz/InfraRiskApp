/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_CARTO_API_KEY?: string
}

interface Window {
  /** CARTO basemap API key, injected into index.html by the backend. */
  __CARTO_API_KEY__?: string
}
