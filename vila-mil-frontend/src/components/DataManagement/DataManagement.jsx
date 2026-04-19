import React, { useState, useEffect, useCallback, useRef } from 'react'
import {
  Box,
  Typography,
  Grid,
  Card,
  CardContent,
  CardHeader,
  Button,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Divider,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  IconButton,
  CircularProgress,
  Alert,
} from '@mui/material'
import { alpha } from '@mui/material/styles'
import { Delete as DeleteIcon, Refresh as RefreshIcon } from '@mui/icons-material'
import { dataApi } from '../../services/api'
import useCancerOptions from '../../hooks/useCancerOptions'
import Toast from '../common/Toast.jsx'

const sectionCardSx = (accent) => (theme) => ({
  height: '100%',
  borderRadius: 2,
  overflow: 'hidden',
  border: '1px solid',
  borderColor: 'divider',
  borderTop: '4px solid',
  borderTopColor: accent,
  bgcolor: alpha(accent, theme.palette.mode === 'dark' ? 0.12 : 0.05),
  boxShadow:
    theme.palette.mode === 'dark' ? 'none' : `0 4px 16px ${alpha(accent, 0.1)}`,
  transition: 'box-shadow 0.2s ease',
  '&:hover': {
    boxShadow:
      theme.palette.mode === 'dark' ? 'none' : `0 8px 22px ${alpha(accent, 0.14)}`,
  },
})

const DataManagement = () => {
  const { cancerOptions, cancer: selectedCancer, setCancer: setSelectedCancer } = useCancerOptions('LUSC')
  const [selectedFeatureType, setSelectedFeatureType] = useState('20')
  const [uploading, setUploading] = useState(false)
  const [files, setFiles] = useState([])
  const [fileInputKey, setFileInputKey] = useState(0)
  const fileInputRef = useRef(null)

  const [featureFiles, setFeatureFiles] = useState([])
  const [loading, setLoading] = useState(true)
  const [datasetSummary, setDatasetSummary] = useState(null)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')

  const loadFeatures = useCallback(async () => {
    setError('')
    try {
      const res = await dataApi.getFeatures(selectedCancer, selectedFeatureType)
      const list = res?.files || res?.data?.files || []
      setFeatureFiles(Array.isArray(list) ? list : [])
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '加载特征列表失败')
      setFeatureFiles([])
    }
  }, [selectedCancer, selectedFeatureType])

  const loadSummary = useCallback(async () => {
    try {
      const res = await dataApi.getDatasets()
      setDatasetSummary(res)
    } catch {
      setDatasetSummary(null)
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    const run = async () => {
      setLoading(true)
      await loadFeatures()
      await loadSummary()
      if (!cancelled) setLoading(false)
    }
    run()
    return () => {
      cancelled = true
    }
  }, [loadFeatures, loadSummary])

  const handleFileChange = (e) => {
    setFiles(e.target.files ? [...e.target.files] : [])
  }

  const handleUpload = async () => {
    const fromDom = fileInputRef.current?.files?.length
      ? [...fileInputRef.current.files]
      : []
    const toUpload = fromDom.length > 0 ? fromDom : files
    if (toUpload.length === 0) {
      setError('请先选择 .h5 文件（若已选仍失败，请换用下方「选择文件」按钮重选）')
      return
    }
    setUploading(true)
    setError('')
    setNotice('')
    try {
      const formData = new FormData()
      formData.append('cancer', selectedCancer)
      formData.append('featureType', selectedFeatureType)
      toUpload.forEach((f) => formData.append('files', f))
      await dataApi.uploadFeatures(formData)
      setNotice(`已上传 ${toUpload.length} 个文件`)
      setFiles([])
      if (fileInputRef.current) fileInputRef.current.value = ''
      setFileInputKey((k) => k + 1)
      await loadFeatures()
      await loadSummary()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '上传失败')
    } finally {
      setUploading(false)
    }
  }

  const handleDelete = async (id) => {
    setError('')
    try {
      await dataApi.deleteFeature(id)
      setNotice('已删除')
      await loadFeatures()
      await loadSummary()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '删除失败')
    }
  }

  return (
    <Box sx={{ mt: 2 }}>
      <Box
        sx={(theme) => ({
          mb: 2.5,
          p: { xs: 2, sm: 2.5 },
          borderRadius: 2,
          border: '1px solid',
          borderColor: 'divider',
          background:
            theme.palette.mode === 'dark'
              ? 'linear-gradient(115deg, rgba(2,136,209,0.22) 0%, rgba(46,125,50,0.12) 100%)'
              : 'linear-gradient(115deg, rgba(2,136,209,0.12) 0%, rgba(46,125,50,0.06) 100%)',
        })}
      >
        <Typography variant="h4" component="h1" gutterBottom sx={{ fontWeight: 700, mb: 1 }}>
          Data Management
        </Typography>
        <Typography variant="body2" color="text.secondary">
          上传的 .h5 特征会保存在后端 <code>ViLa-MIL/uploaded_features/</code>，与训练使用的数据目录相互独立；若要让训练读取此处文件，需在训练配置里把{' '}
          <code>data_root_dir</code> 指到对应路径。
        </Typography>
      </Box>

      {error && <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>{error}</Alert>}
      <Toast open={!!notice} message={notice} severity="success" onClose={() => setNotice('')} />

      {datasetSummary?.summary && (
        <Alert
          severity="info"
          sx={(theme) => ({
            mb: 2,
            borderRadius: 2,
            border: '1px solid',
            borderColor: alpha(theme.palette.info.main, 0.35),
            bgcolor: alpha(theme.palette.info.main, theme.palette.mode === 'dark' ? 0.12 : 0.06),
          })}
        >
          已登记特征文件总数: <strong>{datasetSummary.totalFiles ?? '—'}</strong>
          {Object.keys(datasetSummary.summary).length > 0 && (
            <Typography variant="caption" component="div" sx={{ mt: 1, display: 'block', lineHeight: 1.6 }}>
              按癌种: {Object.entries(datasetSummary.summary).map(([c, v]) => `${c}(10×:${v['10'] ?? 0}, 20×:${v['20'] ?? 0})`).join('；')}
            </Typography>
          )}
        </Alert>
      )}

      <Grid container spacing={3}>
        <Grid item xs={12} md={4}>
          <Card sx={sectionCardSx('#0288d1')}>
            <CardHeader
              title="Upload Features"
              titleTypographyProps={{ variant: 'subtitle1', fontWeight: 700 }}
            />
            <CardContent>
              <FormControl
                fullWidth
                sx={(theme) => ({
                  mb: 2,
                  '& .MuiOutlinedInput-root': {
                    bgcolor: theme.palette.mode === 'dark' ? 'action.hover' : 'background.paper',
                  },
                })}
              >
                <InputLabel id="cancer-label">Cancer Type</InputLabel>
                <Select
                  labelId="cancer-label"
                  value={selectedCancer}
                  label="Cancer Type"
                  onChange={(e) => setSelectedCancer(e.target.value)}
                >
                  {cancerOptions.map((opt) => (
                    <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
                  ))}
                </Select>
              </FormControl>

              <FormControl
                fullWidth
                sx={(theme) => ({
                  mb: 2,
                  '& .MuiOutlinedInput-root': {
                    bgcolor: theme.palette.mode === 'dark' ? 'action.hover' : 'background.paper',
                  },
                })}
              >
                <InputLabel id="feature-type-label">Feature Type</InputLabel>
                <Select
                  labelId="feature-type-label"
                  value={selectedFeatureType}
                  label="Feature Type"
                  onChange={(e) => setSelectedFeatureType(e.target.value)}
                >
                  <MenuItem value="20">20×（对应训练目录 features/20）</MenuItem>
                  <MenuItem value="10">10×（对应训练目录 features/10）</MenuItem>
                </Select>
              </FormControl>

              <Button
                component="label"
                variant="outlined"
                fullWidth
                sx={(theme) => ({
                  mb: 2,
                  py: 1.25,
                  borderColor: alpha('#0288d1', 0.55),
                  color: 'primary.main',
                  '&:hover': { borderColor: 'primary.main', bgcolor: alpha('#0288d1', 0.06) },
                })}
              >
                选择 .h5 文件（可多选）
                <input
                  key={fileInputKey}
                  ref={fileInputRef}
                  type="file"
                  multiple
                  accept=".h5,.hdf5"
                  hidden
                  onChange={handleFileChange}
                />
              </Button>

              <Button
                variant="contained"
                fullWidth
                onClick={handleUpload}
                disabled={uploading}
              >
                {uploading ? <CircularProgress size={20} color="inherit" /> : 'Upload'}
              </Button>

              {files.length > 0 && (
                <Box sx={{ mt: 2 }}>
                  <Typography variant="body2" color="text.secondary">
                    Selected files ({files.length}):
                  </Typography>
                  <List dense sx={{ mt: 1, maxHeight: 150, overflow: 'auto' }}>
                    {files.map((file, index) => (
                      <ListItem key={index}>
                        <ListItemText
                          primary={file.name}
                          secondary={`${(file.size / (1024 * 1024)).toFixed(2)} MB`}
                        />
                      </ListItem>
                    ))}
                  </List>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          <Card sx={sectionCardSx('#2e7d32')}>
            <CardHeader
              title="Feature Files"
              titleTypographyProps={{ variant: 'subtitle1', fontWeight: 700 }}
              action={
                <Button
                  variant="outlined"
                  size="small"
                  startIcon={<RefreshIcon />}
                  onClick={() => {
                    loadFeatures()
                    loadSummary()
                  }}
                  disabled={loading}
                  sx={{ color: '#2e7d32', borderColor: alpha('#2e7d32', 0.5), '&:hover': { borderColor: '#2e7d32' } }}
                >
                  Refresh
                </Button>
              }
            />
            <CardContent>
              <Box
                sx={(theme) => ({
                  display: 'flex',
                  flexWrap: 'wrap',
                  gap: 2,
                  mb: 2,
                  p: 1.5,
                  borderRadius: 1.5,
                  bgcolor: theme.palette.mode === 'dark' ? 'action.hover' : alpha('#2e7d32', 0.04),
                  border: '1px solid',
                  borderColor: 'divider',
                })}
              >
                <FormControl sx={{ minWidth: 160 }}>
                  <InputLabel id="filter-cancer-label">Cancer Type</InputLabel>
                  <Select
                    labelId="filter-cancer-label"
                    value={selectedCancer}
                    label="Cancer Type"
                    onChange={(e) => setSelectedCancer(e.target.value)}
                  >
                    {cancerOptions.map((opt) => (
                      <MenuItem key={opt.value} value={opt.value}>{opt.label}</MenuItem>
                    ))}
                  </Select>
                </FormControl>

                <FormControl sx={{ minWidth: 160 }}>
                  <InputLabel id="filter-feature-type-label">Feature Type</InputLabel>
                  <Select
                    labelId="filter-feature-type-label"
                    value={selectedFeatureType}
                    label="Feature Type"
                    onChange={(e) => setSelectedFeatureType(e.target.value)}
                  >
                    <MenuItem value="20">20×</MenuItem>
                    <MenuItem value="10">10×</MenuItem>
                  </Select>
                </FormControl>
              </Box>

              <Divider sx={{ mb: 2 }} />

              {loading ? (
                <Box sx={{ display: 'flex', justifyContent: 'center', py: 4 }}>
                  <CircularProgress />
                </Box>
              ) : featureFiles.length > 0 ? (
                <List disablePadding sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
                  {featureFiles.map((file) => (
                    <ListItem
                      key={file.id}
                      sx={(theme) => ({
                        borderRadius: 1.5,
                        border: '1px solid',
                        borderColor: 'divider',
                        bgcolor: theme.palette.mode === 'dark' ? 'action.hover' : 'background.paper',
                        pr: 7,
                        transition: 'box-shadow 0.15s ease',
                        '&:hover': {
                          boxShadow: `0 2px 10px ${alpha('#2e7d32', 0.12)}`,
                        },
                      })}
                    >
                      <ListItemText
                        primary={file.name}
                        primaryTypographyProps={{ variant: 'body2', sx: { fontWeight: 600, wordBreak: 'break-all' } }}
                        secondary={`Size: ${file.size} ｜ id: ${file.id}`}
                        secondaryTypographyProps={{ variant: 'caption' }}
                      />
                      <ListItemSecondaryAction>
                        <IconButton
                          edge="end"
                          aria-label="delete"
                          color="error"
                          size="small"
                          onClick={() => handleDelete(file.id)}
                        >
                          <DeleteIcon fontSize="small" />
                        </IconButton>
                      </ListItemSecondaryAction>
                    </ListItem>
                  ))}
                </List>
              ) : (
                <Box
                  sx={{
                    py: 3,
                    px: 2,
                    borderRadius: 2,
                    border: '1px dashed',
                    borderColor: 'divider',
                    bgcolor: 'action.hover',
                    textAlign: 'center',
                  }}
                >
                  <Typography variant="body1" sx={{ fontWeight: 600 }}>
                    暂无特征文件
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                    在左侧上传 .h5 特征文件，或切换癌种 / 倍率筛选
                  </Typography>
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  )
}

export default DataManagement
