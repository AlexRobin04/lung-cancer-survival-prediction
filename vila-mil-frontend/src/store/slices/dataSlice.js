import { createSlice } from '@reduxjs/toolkit'

const initialState = {
  featureFiles: [],
  datasets: [],
  loading: false,
  error: null
}

const dataSlice = createSlice({
  name: 'data',
  initialState,
  reducers: {
    fetchFeatureFiles: (state, action) => {
      state.loading = true
      state.error = null
    },
    fetchFeatureFilesSuccess: (state, action) => {
      state.loading = false
      state.featureFiles = action.payload
    },
    fetchFeatureFilesFailure: (state, action) => {
      state.loading = false
      state.error = action.payload
    },
    uploadFeatureFiles: (state, action) => {
      state.loading = true
      state.error = null
    },
    uploadFeatureFilesSuccess: (state, action) => {
      state.loading = false
      state.featureFiles = [...state.featureFiles, ...action.payload]
    },
    uploadFeatureFilesFailure: (state, action) => {
      state.loading = false
      state.error = action.payload
    },
    deleteFeatureFile: (state, action) => {
      state.loading = true
      state.error = null
    },
    deleteFeatureFileSuccess: (state, action) => {
      state.loading = false
      const { fileId } = action.payload
      state.featureFiles = state.featureFiles.filter(file => file.id !== fileId)
    },
    deleteFeatureFileFailure: (state, action) => {
      state.loading = false
      state.error = action.payload
    },
    fetchDatasets: (state, action) => {
      state.loading = true
      state.error = null
    },
    fetchDatasetsSuccess: (state, action) => {
      state.loading = false
      state.datasets = action.payload
    },
    fetchDatasetsFailure: (state, action) => {
      state.loading = false
      state.error = action.payload
    }
  }
})

export const { 
  fetchFeatureFiles, 
  fetchFeatureFilesSuccess, 
  fetchFeatureFilesFailure, 
  uploadFeatureFiles, 
  uploadFeatureFilesSuccess, 
  uploadFeatureFilesFailure, 
  deleteFeatureFile, 
  deleteFeatureFileSuccess, 
  deleteFeatureFileFailure, 
  fetchDatasets, 
  fetchDatasetsSuccess, 
  fetchDatasetsFailure 
} = dataSlice.actions

export default dataSlice.reducer
