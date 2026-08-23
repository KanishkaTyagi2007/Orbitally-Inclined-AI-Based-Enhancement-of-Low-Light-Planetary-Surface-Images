import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The Flask app serves the built SPA out of ../static, so `npm run build`
// writes straight there and no extra copy step is needed. `base: "./"` keeps
// asset URLs relative so the same bundle works however Flask mounts it.
//
// In dev (`npm run dev`, port 5173) the API is proxied to the Flask server on
// 5000, which keeps the browser on a single origin and avoids CORS entirely.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: {
    outDir: "../static",
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://127.0.0.1:5000", changeOrigin: true },
    },
  },
});
