// Local developer shortcuts (Ctrl+X reset, Ctrl+S fake generation). Requires both
// the dev server and VITE_DEV_TOOLS=true, so production builds never include them.
export const DEV_TOOLS = import.meta.env.DEV && import.meta.env.VITE_DEV_TOOLS === 'true'
