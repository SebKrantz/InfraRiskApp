"""Browser-executed tools: they drive the React app's own state setters.

Schema-only declarations — the executor lives in the frontend
(`frontend/src/lib/assistantTools.ts`), which validates each call and applies
the App.tsx bindings. UI tools SHOW things to the user; the server tools return
the authoritative numbers. Results come back as short confirmation strings.

Setting the dataset, hazard, threshold or vulnerability inputs makes the app
re-run its own analysis ~300 ms later; the user sees the map and chart update
without any further call.
"""

from __future__ import annotations

from . import client_tool

client_tool(
    "ui_read_app_state",
    "Snapshot of what the user is currently looking at: the loaded dataset, the "
    "selected hazard, the threshold and its valid range, vulnerability settings, "
    "display options and a summary of the on-screen result. Call this first "
    "whenever the user says 'this', 'here', or refers to the screen.",
    {"type": "object", "properties": {}},
)

client_tool(
    "ui_show_dataset",
    "Put a dataset on the user's map and make it the app's active dataset. Pass "
    "the file_id from load_infrastructure (or list_datasets). The map fits to "
    "its extent automatically, and if a hazard is already selected the app "
    "re-runs its analysis. ALWAYS call this after load_infrastructure so the "
    "user can see what you are working with.",
    {
        "type": "object",
        "properties": {
            "file_id": {"type": "string", "description": "From load_infrastructure."},
        },
        "required": ["file_id"],
    },
)

client_tool(
    "ui_select_hazard",
    "Select a hazard layer in the app: the raster is drawn on the map and, if a "
    "dataset is loaded, the analysis re-runs. Selecting a layer resets the "
    "threshold to that layer's minimum, so call ui_set_threshold AFTER this, "
    "never before. Returns the layer's intensity range.",
    {
        "type": "object",
        "properties": {
            "hazard": {
                "type": "string",
                "description": "hazard_id or layer name. Empty string clears the selection.",
            },
        },
        "required": ["hazard"],
    },
)

client_tool(
    "ui_set_threshold",
    "Set the hazard intensity threshold, in the layer's own units (mm, km/h, "
    "cm/s², class). Assets at or above it count as affected. The app re-runs "
    "its analysis and repaints the map. Requires a hazard to be selected first.",
    {
        "type": "object",
        "properties": {
            "threshold": {"type": "number", "description": "In the layer's units."},
        },
        "required": ["threshold"],
    },
)

client_tool(
    "ui_set_vulnerability",
    "Turn the app's vulnerability (damage-cost) mode on or off and supply its "
    "inputs. The curve must already be loaded with load_vulnerability_curve; "
    "name it here and the app receives the same file. Both the curve and a "
    "positive replacement value are needed before the app will compute "
    "anything, so pass them together. Replacement value is per feature for "
    "point datasets and per metre for line datasets.",
    {
        "type": "object",
        "properties": {
            "enabled": {"type": "boolean"},
            "curve": {
                "type": "string",
                "description": "Curve name from load_vulnerability_curve.",
            },
            "replacement_value": {"type": "number", "description": "Must be > 0."},
        },
        "required": ["enabled"],
    },
)

client_tool(
    "ui_set_display",
    "Change how the map looks, without touching the analysis: the hazard colour "
    "palette, its opacity, and the basemap.",
    {
        "type": "object",
        "properties": {
            "palette": {
                "type": "string",
                "enum": ["viridis", "magma", "inferno", "plasma", "cividis", "turbo"],
            },
            "opacity": {"type": "number", "description": "Hazard layer opacity, 0-100."},
            "basemap": {
                "type": "string",
                "description": "positron, dark-matter, osm, topo, esri-street, "
                "esri-topo, esri-terrain, esri-ocean, esri-imagery, google-maps, "
                "google-terrain, google-hybrid, google-satellite.",
            },
        },
    },
)

client_tool(
    "ui_fit_map",
    "Move the map camera to a bounding box [west, south, east, north] in "
    "degrees. Use it to zoom to a country or to a cluster of affected assets; "
    "the map already fits a newly shown dataset by itself.",
    {
        "type": "object",
        "properties": {
            "bbox": {
                "type": "array",
                "items": {"type": "number"},
                "description": "[west, south, east, north] in degrees.",
            },
        },
        "required": ["bbox"],
    },
)

client_tool(
    "ui_set_sidebar",
    "Open or collapse the control sidebar — collapse it to give the map room "
    "when showing the user a result.",
    {
        "type": "object",
        "properties": {"open": {"type": "boolean"}},
        "required": ["open"],
    },
)

client_tool(
    "ui_clear_data",
    "Clear the loaded dataset and its results from the app. Ask the user before "
    "doing this; it discards what they were looking at.",
    {"type": "object", "properties": {}},
)
