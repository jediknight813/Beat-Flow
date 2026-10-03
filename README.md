
<p align="center"><img src=".github/assets/banner.png" alt="BeatFlow" width="100%"></p>

# BeatFlow

Beat Saber maps generated from a song, entirely in the browser.

```
npm install
npm run dev
```

### Dev tools

Add `VITE_DEV_TOOLS=true` to `.env.local` to unlock local-only shortcuts in `npm run dev` (never in production builds):

- **Ctrl+S**: run a fake generation with a made-up song, cover and map, through progress, export and history.
- **Ctrl+X**: delete downloaded models and song history, then reload.

Deploys to GitHub Pages manually: run the **pages** workflow from the Actions tab (`.github/workflows/pages.yml`).
