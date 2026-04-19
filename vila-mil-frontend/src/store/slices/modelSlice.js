import { createSlice } from '@reduxjs/toolkit'

const initialState = {
  models: [],
  currentModel: null,
  evaluationResults: {},
  loading: false,
  error: null
}

const modelSlice = createSlice({
  name: 'model',
  initialState,
  reducers: {
    fetchModels: (state, action) => {
      state.loading = true
      state.error = null
    },
    fetchModelsSuccess: (state, action) => {
      state.loading = false
      state.models = action.payload
    },
    fetchModelsFailure: (state, action) => {
      state.loading = false
      state.error = action.payload
    },
    selectModel: (state, action) => {
      state.currentModel = action.payload
    },
    evaluateModel: (state, action) => {
      state.loading = true
      state.error = null
    },
    evaluateModelSuccess: (state, action) => {
      state.loading = false
      const { modelId, results } = action.payload
      state.evaluationResults[modelId] = results
    },
    evaluateModelFailure: (state, action) => {
      state.loading = false
      state.error = action.payload
    },
    fetchEvaluationResults: (state, action) => {
      state.loading = true
      state.error = null
    },
    fetchEvaluationResultsSuccess: (state, action) => {
      state.loading = false
      const { modelId, results } = action.payload
      state.evaluationResults[modelId] = results
    },
    fetchEvaluationResultsFailure: (state, action) => {
      state.loading = false
      state.error = action.payload
    }
  }
})

export const { 
  fetchModels, 
  fetchModelsSuccess, 
  fetchModelsFailure, 
  selectModel, 
  evaluateModel, 
  evaluateModelSuccess, 
  evaluateModelFailure, 
  fetchEvaluationResults, 
  fetchEvaluationResultsSuccess, 
  fetchEvaluationResultsFailure 
} = modelSlice.actions

export default modelSlice.reducer
