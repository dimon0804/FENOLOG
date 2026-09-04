// Один модуль на все обращения к сервису.
//
// Пути относительные: в собранном виде интерфейс отдаётся тем же процессом, что
// и API, а в разработке запросы проксирует vite. Адреса сервера в коде нет
// нигде — иначе его пришлось бы менять перед сборкой.

async function request(path, options = {}) {
  const response = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  })
  let payload = null
  try {
    payload = await response.json()
  } catch {
    payload = null
  }
  if (!response.ok) {
    // Сервис отвечает по-русски и по делу («Контур слишком велик…»), поэтому
    // текст показывается пользователю как есть, а не подменяется общей фразой.
    const error = new Error(payload?.detail || `Ошибка ${response.status}`)
    error.status = response.status
    throw error
  }
  return payload
}

export const api = {
  health: () => request('/api/providers/health'),

  searchRegion: (q) => request(`/api/regions/search?q=${encodeURIComponent(q)}&limit=6`),

  discover: (bbox, limit = 60) =>
    request(`/api/polygons/discover?bbox=${bbox.map((v) => v.toFixed(5)).join(',')}&limit=${limit}`),

  listPolygons: () => request('/api/polygons'),

  savePolygon: (body) => request('/api/polygons', { method: 'POST', body: JSON.stringify(body) }),

  renamePolygon: (id, name) =>
    request(`/api/polygons/${id}`, { method: 'PATCH', body: JSON.stringify({ name }) }),

  deletePolygon: (id) => request(`/api/polygons/${id}`, { method: 'DELETE' }),

  analyzeGeometry: (body) => request('/api/analyze', { method: 'POST', body: JSON.stringify(body) }),

  analyzePolygon: (id, body) =>
    request(`/api/polygons/${id}/analyze`, { method: 'POST', body: JSON.stringify(body) }),

  task: (id) => request(`/api/tasks/${id}`),

  taskResult: (id) => request(`/api/tasks/${id}/result`),

  savedResult: (id) => request(`/api/polygons/${id}/result`),
}

// Опрос прогресса. Раз в секунду: чаще — лишняя нагрузка на ровном месте, реже —
// шкала начинает дёргаться и выглядит подвисшей.
export function pollTask(taskId, onProgress) {
  return new Promise((resolve, reject) => {
    const tick = async () => {
      try {
        const task = await api.task(taskId)
        onProgress?.(task)
        if (task.status === 'done') {
          const full = await api.taskResult(taskId)
          resolve(full)
        } else if (task.status === 'failed') {
          reject(new Error(task.error || 'Анализ не удался'))
        } else {
          setTimeout(tick, 1000)
        }
      } catch (error) {
        reject(error)
      }
    }
    tick()
  })
}
