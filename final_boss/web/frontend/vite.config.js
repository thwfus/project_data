import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import FullReload from 'vite-plugin-full-reload'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": "http://localhost:3000",
    },
  },
});
