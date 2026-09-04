"""The in-app AI assistant: agent loop, tool registry, deliverables.

Server-side tools call the analysis layer directly (app.utils.geospatial,
app.api.hazards, app.api.export), so the assistant's numbers and figures are
the app's own. Client-side `ui_*` tools are executed by the browser against
App.tsx's state setters.
"""
