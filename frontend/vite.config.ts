import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  server: {

    host: "http://192.168.10.99",
    port: 5174,
    proxy: {
      "/api": {
        target: "http://192.168.10.99:11032",
        changeOrigin: true
      },
      "/health": {
        target: "http://192.168.10.99:11032",
        changeOrigin: true
      },
      "/docs": {
        target: "http://192.168.10.99:11032",
        changeOrigin: true
      }
    }
  }
});
