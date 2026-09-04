import { copyFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs'
import path from 'node:path'

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// В сборке интерфейс отдаётся тем же процессом FastAPI, что и API, поэтому все
// запросы идут на тот же адрес и относительными путями. В разработке фронтенд
// живёт на своём порту, и обращения к API проксируются на бэкенд — иначе
// пришлось бы прописывать адрес сервера в коде и не забыть убрать его перед
// сборкой.
const API = process.env.FENOLOG_API || 'http://127.0.0.1:8000'

const MAPLIBRE_DIST = path.resolve('node_modules/maplibre-gl/dist')
// Воркер и общий с ним модуль. Именно два файла, а не один: воркер импортирует
// соседа по относительному пути, и в одиночку он бесполезен.
const WORKER_FILES = ['maplibre-gl-worker.mjs', 'maplibre-gl-shared.mjs']
const WORKER_ROUTE = '/maplibre/'

/**
 * Воркер maplibre рядом со сборкой.
 *
 * Библиотека вычисляет адрес воркера как файл, соседний с собственным модулем.
 * После сборки она лежит внутри общего бандла, соседа рядом нет — запрос уходит
 * в 404. Ошибка тихая и обманчивая: растровая подложка рисуется в главном
 * потоке как ни в чём не бывало, а все слои GeoJSON (найденные контуры, выбранное
 * поле, рисуемый полигон) разбираются в воркере и просто не появляются.
 *
 * Класть их через `?url` нельзя: vite переименовывает файлы под хеш, и
 * относительный импорт внутри воркера перестаёт находить соседа. Поэтому оба
 * файла копируются как есть, с исходными именами, в /maplibre/.
 */
function maplibreWorker() {
  return {
    name: 'maplibre-worker',

    // В сборке — копия в dist.
    writeBundle(options) {
      const out = path.join(options.dir || 'dist', 'maplibre')
      mkdirSync(out, { recursive: true })
      for (const file of WORKER_FILES) {
        copyFileSync(path.join(MAPLIBRE_DIST, file), path.join(out, file))
      }
    },

    // В разработке — отдаём прямо из node_modules по тому же адресу, чтобы
    // код приложения не знал разницы между режимами.
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (!req.url?.startsWith(WORKER_ROUTE)) return next()
        const name = path.basename(req.url.split('?')[0])
        const source = path.join(MAPLIBRE_DIST, name)
        if (!WORKER_FILES.includes(name) || !existsSync(source)) return next()
        res.setHeader('Content-Type', 'text/javascript')
        res.end(readFileSync(source))
      })
    },
  }
}

export default defineConfig({
  plugins: [react(), maplibreWorker()],
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
