# Hazard-Infrastructure Analyzer - Frontend

React + TypeScript frontend for the Hazard-Infrastructure Analyzer application.

## Setup

### Prerequisites

- Node.js 18+ and npm/yarn/pnpm

### Installation

1. Install dependencies:

```bash
npm install
# or
yarn install
# or
pnpm install
```

### Development

Start the development server:

```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

The app will be available at `http://localhost:5173`

The frontend is configured to proxy API requests to `http://localhost:8000` (backend).

## Features (Step 2)

This is the Step 2 implementation with UI components and mock data. Features include:

- ✅ Collapsible sidebar
- ✅ File upload UI (mock for now)
- ✅ Hazard selector
- ✅ Info popup with metadata (80% opacity)
- ✅ Color palette selector
- ✅ Hazard intensity slider
- ✅ Bar chart (Recharts)
- ✅ Map with selectable basemaps
- ✅ Placeholder overlays for data visualization

## Project Structure

```
frontend/
├── src/
│   ├── components/
│   │   ├── ui/           # Reusable UI components (shadcn-style)
│   │   ├── Sidebar.tsx   # Main sidebar component
│   │   ├── MapView.tsx   # Map component
│   │   └── BarChart.tsx  # Chart component
│   ├── types/            # TypeScript type definitions
│   ├── lib/              # Utility functions
│   ├── App.tsx           # Main app component
│   └── main.tsx          # Entry point
├── package.json
├── vite.config.ts
└── tailwind.config.js
```

## Next Steps (Step 3)

- Connect to backend API
- Implement actual file upload
- Load real hazard layers
- Perform analysis and visualize results
- Add loading and error states

## Build

To build for production:

```bash
npm run build
```

The built files will be in the `dist` directory.

