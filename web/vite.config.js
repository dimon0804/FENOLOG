import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// В сборке интерфейс отдаётся тем же процессом FastAPI, что и API, поэтому все
// запросы идут на тот же адрес и относительными путями. В разработке фронтенд
// живёт на своём порту, и обращения к API проксируются на бэкенд — иначе
// пришлось бы прописывать адрес сервера в коде и не забыть убрать его перед
// сборкой.
const API = process.env.FENOLOG_API || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: API, changeOrigin: true },
      '/health': { target: API, changeOrigin: true },
    },
  },
  build: {
    // Каталог совпадает с FENOLOG_WEB_DIST в настройках сервиса.
    outDir: 'dist',
    // MapLibre и Recharts вместе весят прилично, предупреждение о размере чанка
    // здесь ни на что не указывает и только зашумляет вывод сборки.
    chunkSizeWarningLimit: 1600,
  },
})
