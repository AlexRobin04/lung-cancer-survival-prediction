import axios from 'axios'

const API_BASE_URL_KEY = 'API_BASE_URL'
const getInitialBaseUrl = () => {
  try {
    return localStorage.getItem(API_BASE_URL_KEY) || '/api'
  } catch {
    return '/api'
  }
}

const client = axios.create({
  baseURL: getInitialBaseUrl(),
  timeout: 300000,
})

export const setApiBaseUrl = (baseUrl) => {
  const v = (baseUrl || '').trim() || '/api'
  client.defaults.baseURL = v
  try {
    localStorage.setItem(API_BASE_URL_KEY, v)
  } catch {
    // ignore
  }
  return v
}

export const getApiBaseUrl = () => client.defaults.baseURL || '/api'

export const modelApi = {
  async listModels() {
    const { data } = await client.get('/models')
    return data
  },
}

export const healthApi = {
  async health() {
    const { data } = await client.get('/health')
    return data
  },
  async config() {
    const { data } = await client.get('/config')
    return data
  },
}

export const trainingApi = {
  async start(payload) {
    const { data } = await client.post('/training/start', payload)
    return data
  },
  async best({ cancer, modelType, mode } = {}) {
    const { data } = await client.get('/training/best', { params: { cancer, modelType, mode } })
    return data
  },
  async deleteHistory({ deleteAll = false, taskIds = [], deleteArtifacts = true } = {}) {
    const { data } = await client.post('/training/history/delete', { deleteAll, taskIds, deleteArtifacts })
    return data
  },
  async stop(taskId) {
    const { data } = await client.post('/training/stop', { taskId })
    return data
  },
  async status(taskId) {
    const { data } = await client.get(`/training/status/${taskId}`)
    return data
  },
  async history() {
    const { data } = await client.get('/training/history')
    return data
  },
  async queue() {
    const { data } = await client.get('/training/queue')
    return data
  },
  async deleteQueue({ taskIds = [], deleteAll = false, deleteArtifacts = true } = {}) {
    const { data } = await client.post('/training/queue/delete', { taskIds, deleteAll, deleteArtifacts })
    return data
  },
  async log(taskId, tail = 200) {
    const { data } = await client.get(`/training/log/${taskId}`, { params: { tail } })
    return data
  },
}

export const evaluationApi = {
  async runs() {
    const { data } = await client.get('/evaluation/runs')
    return data
  },
  async curves(taskId) {
    const { data } = await client.get(`/evaluation/curves/${taskId}`)
    return data
  },
  async km(groups) {
    const { data } = await client.post('/evaluation/km', { groups })
    return data
  },
  /** LUSC 示例：Ours vs Others（后端读取 lusc (1).csv） */
  async kmLuscDemo() {
    const { data } = await client.get('/evaluation/km/lusc-demo')
    return data
  },
}

export const dataApi = {
  async uploadFeatures(formData) {
    const { data } = await client.post('/data/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return data
  },
  async getDatasets() {
    const { data } = await client.get('/data/datasets')
    return data
  },
  async getFeatures(cancer, featureType) {
    const { data } = await client.get(`/data/features/${cancer}`, { params: { featureType } })
    return data
  },
  async deleteFeature(id) {
    const { data } = await client.delete(`/data/feature/${id}`)
    return data
  },
  /** PNG/JPEG 等 → 后端转单层 TIFF 并写入 manifest */
  async uploadRaster(formData) {
    const { data } = await client.post('/data/upload-raster', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return data
  },
}

export const clinicalApi = {
  async uploadCsv(file) {
    const formData = new FormData()
    formData.append('file', file)
    const { data } = await client.post('/clinical/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
    return data
  },
  async listCases() {
    const { data } = await client.get('/clinical/cases')
    return data
  },
  async createCase({ caseId, slideId = '', time = 0, status = 0 }) {
    const { data } = await client.post('/clinical/cases', { caseId, slideId, time, status })
    return data
  },
  async getCase(caseId) {
    const { data } = await client.get(`/clinical/cases/${caseId}`)
    return data
  },
  async getCaseFeatureMeta(caseId) {
    const { data } = await client.get(`/clinical/cases/${caseId}/feature-meta`)
    return data
  },
  async deleteCase(caseId) {
    const { data } = await client.delete(`/clinical/cases/${caseId}`)
    return data
  },
  async linkFeature({ caseId, fileId, featureType }) {
    const { data } = await client.post('/clinical/cases/link-feature', { caseId, fileId, featureType })
    return data
  },
  /** 一次性为病例关联双尺度特征：JSON 传两个 fileId，或 multipart 传文件由后端生成 H5 */
  async associateFeatures({ caseId, cancer, feature20FileId, feature10FileId, file, extractor = 'raster', mpp }) {
    if (file) {
      const fd = new FormData()
      fd.append('caseId', caseId)
      fd.append('cancer', cancer || 'LUSC')
      fd.append('file', file)
      fd.append('extractor', extractor)
      if (mpp !== undefined && mpp !== null && String(mpp).trim() !== '') fd.append('mpp', String(mpp))
      const { data } = await client.post('/clinical/cases/associate-features', fd, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 600000,
      })
      return data
    }
    const { data } = await client.post('/clinical/cases/associate-features', {
      caseId,
      cancer,
      feature20FileId,
      feature10FileId,
    })
    return data
  },
}

export const predictApi = {
  /**
   * 推理输入二选一：
   * - 仅 caseId：使用 Clinical 中已为病例登记的 20×/10× fileId
   * - feature20FileId + feature10FileId：直接按上传清单中的文件推理（可选 caseId 用于关联随访）
   */
  async predict({ caseId, taskId, saveHistory = true, feature20FileId, feature10FileId, cancer }) {
    const body = { taskId, saveHistory }
    if (feature20FileId && feature10FileId) {
      body.feature20FileId = feature20FileId
      body.feature10FileId = feature10FileId
      if (cancer) body.cancer = cancer
      if (caseId) body.caseId = caseId
    } else {
      body.caseId = caseId
    }
    const { data } = await client.post('/predict', body)
    return data
  },
  /** 上传文件，后端生成双尺度 H5 后推理（extractor=raster|trident） */
  async predictFromRaster({ file, taskId, cancer, caseId, saveHistory = true, extractor = 'raster', mpp }) {
    const fd = new FormData()
    fd.append('file', file)
    if (taskId) fd.append('taskId', taskId)
    if (cancer) fd.append('cancer', cancer)
    if (caseId) fd.append('caseId', caseId)
    fd.append('saveHistory', saveHistory ? 'true' : 'false')
    fd.append('extractor', extractor)
    if (mpp !== undefined && mpp !== null && String(mpp).trim() !== '') fd.append('mpp', String(mpp))
    const { data } = await client.post('/predict/from-raster', fd, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 600000,
    })
    return data
  },
  async listPredictions(limit = 50, { taskId } = {}) {
    const params = { limit: Math.min(500, Math.max(1, Number(limit) || 50)) }
    if (taskId) params.taskId = taskId
    const { data } = await client.get('/predictions', { params })
    return data
  },
}

export default client

