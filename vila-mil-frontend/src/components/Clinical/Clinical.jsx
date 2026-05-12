import React, { useEffect, useMemo, useRef, useState } from 'react'
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Divider,
  FormControl,
  InputLabel,
  LinearProgress,
  MenuItem,
  Select,
  TextField,
  ToggleButton,
  ToggleButtonGroup,
  Typography,
} from '@mui/material'
import { clinicalApi } from '../../services/api'
import { readClinicalDemoFakeExtraction } from '../../utils/clinicalDemoStorage'
import useCancerOptions from '../../hooks/useCancerOptions'
import Toast from '../common/Toast.jsx'
import RasterPreview from './RasterPreview.jsx'

function ColumnTitle({ children }) {
  return (
    <Typography
      variant="overline"
      color="text.secondary"
      sx={{
        letterSpacing: 0.8,
        display: 'block',
        mb: 1.25,
        fontWeight: 700,
      }}
    >
      {children}
    </Typography>
  )
}

export default function Clinical() {
  const fileRef = useRef(null)
  /** 上传成功后保留预览（blob URL 由本组件 revoke） */
  const persistedRasterRef = useRef(null)
  /** WSI 缩略图预览（blob URL 由本组件 revoke） */
  const generatedPreviewRef = useRef(null)
  /** 已绑定病例预览（blob URL 由本组件 revoke） */
  const boundCasePreviewRef = useRef(null)
  const [persistedRasterPreview, setPersistedRasterPreview] = useState(null)
  const [generatedPreview, setGeneratedPreview] = useState(null)
  const [boundCasePreview, setBoundCasePreview] = useState(null)
  const [boundCasePreviewBusy, setBoundCasePreviewBusy] = useState(false)
  const [previewBusy, setPreviewBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [cases, setCases] = useState([])
  const { cancerOptions, cancer: selectedCancer, setCancer: setSelectedCancer } = useCancerOptions('LUSC')
  const [selectedCaseId, setSelectedCaseId] = useState('')
  const [newCaseId, setNewCaseId] = useState('')
  const [editTime, setEditTime] = useState('')
  const [editStatus, setEditStatus] = useState('0')
  const [genMode, setGenMode] = useState('quick') // quick | formal
  const [rasterFile, setRasterFile] = useState(null)
  const [tridentMpp, setTridentMpp] = useState('0.25')
  const [associating, setAssociating] = useState(false)
  const [associateProgress, setAssociateProgress] = useState(0)
  /** 批量 CSV：已选文件名（未点导入）、上传中、最近一次成功导入摘要 */
  const [csvPendingName, setCsvPendingName] = useState('')
  const [csvUploading, setCsvUploading] = useState(false)
  const [csvLastImport, setCsvLastImport] = useState(null)
  const caseOptions = useMemo(() => cases.map((c) => c.caseId), [cases])
  const selectedCase = useMemo(() => cases.find((c) => c.caseId === selectedCaseId) || null, [cases, selectedCaseId])
  const hasBoundFeatures = Boolean(selectedCase?.feature20FileId && selectedCase?.feature10FileId)

  useEffect(() => {
    persistedRasterRef.current = persistedRasterPreview
  }, [persistedRasterPreview])

  useEffect(() => {
    generatedPreviewRef.current = generatedPreview
  }, [generatedPreview])

  useEffect(() => {
    boundCasePreviewRef.current = boundCasePreview
  }, [boundCasePreview])

  useEffect(() => {
    return () => {
      if (persistedRasterRef.current?.url) {
        URL.revokeObjectURL(persistedRasterRef.current.url)
      }
      if (generatedPreviewRef.current?.url) {
        URL.revokeObjectURL(generatedPreviewRef.current.url)
      }
      if (boundCasePreviewRef.current?.url) {
        URL.revokeObjectURL(boundCasePreviewRef.current.url)
      }
    }
  }, [])

  useEffect(() => {
    let cancelled = false
    setBoundCasePreview((prev) => {
      if (prev?.url) URL.revokeObjectURL(prev.url)
      return null
    })
    if (!selectedCaseId) {
      setBoundCasePreviewBusy(false)
      return undefined
    }
    ;(async () => {
      try {
        setBoundCasePreviewBusy(true)
        const blob = await clinicalApi.getCasePreview(selectedCaseId)
        if (cancelled) return
        const url = URL.createObjectURL(blob)
        setBoundCasePreview({ url, name: `${selectedCaseId}（已绑定预览）` })
      } catch {
        if (cancelled) return
        setBoundCasePreview(null)
      } finally {
        if (!cancelled) setBoundCasePreviewBusy(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [selectedCaseId])

  useEffect(() => {
    let cancelled = false
    const ext = String(rasterFile?.name || '')
      .toLowerCase()
      .split('.')
      .pop()
    const wsiExts = new Set(['svs', 'ndpi', 'mrxs', 'scn'])
    const isWsiLike = Boolean(ext && wsiExts.has(ext))

    setGeneratedPreview((prev) => {
      if (prev?.url) URL.revokeObjectURL(prev.url)
      return null
    })
    setPreviewBusy(false)

    if (!rasterFile || !isWsiLike) return undefined

    ;(async () => {
      try {
        setPreviewBusy(true)
        const blob = await clinicalApi.previewWsi(rasterFile, { maxSide: 1800 })
        if (cancelled) return
        const url = URL.createObjectURL(blob)
        setGeneratedPreview({ url, name: `${rasterFile.name}（缩略预览）` })
      } catch (e) {
        if (cancelled) return
        setError(e?.response?.data?.message || e.message || 'WSI 预览生成失败')
      } finally {
        if (!cancelled) setPreviewBusy(false)
      }
    })()

    return () => {
      cancelled = true
    }
  }, [rasterFile])

  useEffect(() => {
    if (!selectedCase) {
      setEditTime('')
      setEditStatus('0')
      return
    }
    setEditTime(selectedCase.time ?? '')
    setEditStatus(String(selectedCase.status ?? 0))
  }, [selectedCase])

  const loadAll = async () => {
    setError('')
    try {
      const cRes = await clinicalApi.listCases()
      setCases(cRes?.cases || [])
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '加载失败')
    }
  }

  useEffect(() => {
    loadAll()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const uploadCsv = async () => {
    const f = fileRef.current?.files?.[0]
    if (!f) {
      setError('请先选择 CSV 文件')
      return
    }
    setError('')
    setNotice('')
    setCsvUploading(true)
    try {
      const res = await clinicalApi.uploadCsv(f)
      const count = res?.count
      setNotice(`已导入 ${count ?? '—'} 条病例`)
      setCsvLastImport({ fileName: f.name, count: count ?? null })
      setCsvPendingName('')
      await loadAll()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '导入失败')
    } finally {
      setCsvUploading(false)
      if (fileRef.current) fileRef.current.value = ''
      setCsvPendingName('')
    }
  }

  const createCase = async () => {
    const cid = String(newCaseId || '').trim()
    if (!cid) {
      setError('请先输入 caseId')
      return
    }
    setError('')
    setNotice('')
    try {
      const res = await clinicalApi.createCase({ caseId: cid })
      setNotice(res?.created ? `已新增：${cid}` : `已选中已有病例：${cid}`)
      setSelectedCaseId(cid)
      setNewCaseId('')
      await loadAll()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '创建病例失败')
    }
  }

  const saveFollowup = async () => {
    if (!selectedCaseId) {
      setError('请先在左侧选择病例')
      return
    }
    setError('')
    setNotice('')
    try {
      await clinicalApi.createCase({
        caseId: selectedCaseId,
        slideId: selectedCase?.slideId || '',
        time: editTime === '' ? 0 : Number(editTime),
        status: Number(editStatus),
      })
      setNotice(`已保存随访：${selectedCaseId}`)
      await loadAll()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '保存失败')
    }
  }

  const associateFeaturesFromUpload = async () => {
    if (!selectedCaseId) {
      setError('请先在左侧选择病例')
      return
    }
    if (!rasterFile) {
      setError('请选择病理图像文件')
      return
    }
    const extractor = genMode === 'formal' ? 'trident' : 'raster'
    const quick = genMode === 'quick'
    if (extractor === 'trident' && rasterFile && /\.(png|jpe?g)$/i.test(rasterFile.name || '') && !(Number(tridentMpp) > 0)) {
      setError('TRIDENT 处理 PNG/JPEG 需填写 mpp（如 0.25）')
      return
    }
    setError('')
    setNotice('')
    setAssociating(true)
    setAssociateProgress(2)
    const t0 = Date.now()
    const timer = setInterval(() => {
      const dt = Date.now() - t0
      const target = dt < 8000 ? 35 : dt < 20000 ? 65 : dt < 40000 ? 85 : 95
      setAssociateProgress((p) => (p < target ? p + 2 : p))
    }, 600)
    try {
      const res = await clinicalApi.associateFeatures({
        caseId: selectedCaseId,
        cancer: selectedCancer,
        file: rasterFile,
        extractor,
        quick,
        mpp: extractor === 'trident' ? tridentMpp : undefined,
        demoFakeExtraction: readClinicalDemoFakeExtraction() || undefined,
      })
      setNotice(
        res?.message ||
          (extractor === 'trident' ? '已通过 TRIDENT 生成特征并关联' : '已通过快速近似流程生成特征并关联')
      )
      setAssociateProgress(100)
      if (rasterFile) {
        setPersistedRasterPreview((prev) => {
          if (prev?.url) URL.revokeObjectURL(prev.url)
          return { url: URL.createObjectURL(rasterFile), name: rasterFile.name }
        })
      }
      setRasterFile(null)
      await loadAll()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '关联失败')
    } finally {
      clearInterval(timer)
      setAssociating(false)
      setTimeout(() => setAssociateProgress(0), 800)
    }
  }

  const deleteCase = async () => {
    if (!selectedCaseId) {
      setError('请先在左侧选择病例')
      return
    }
    const ok = window.confirm(`删除病例「${selectedCaseId}」？将清除该病例下的特征关联。`)
    if (!ok) return
    setError('')
    setNotice('')
    try {
      await clinicalApi.deleteCase(selectedCaseId)
      setNotice(`已删除：${selectedCaseId}`)
      setSelectedCaseId('')
      setPersistedRasterPreview((prev) => {
        if (prev?.url) URL.revokeObjectURL(prev.url)
        return null
      })
      setRasterFile(null)
      await loadAll()
    } catch (e) {
      setError(e?.response?.data?.message || e.message || '删除失败')
    }
  }

  return (
    <Box sx={{ mt: 2 }}>
      <Box
        sx={(theme) => ({
          mb: 3,
          p: { xs: 2, sm: 2.5 },
          borderRadius: 2,
          border: '1px solid',
          borderColor: 'divider',
          background:
            theme.palette.mode === 'dark'
              ? 'linear-gradient(115deg, rgba(25,118,210,0.20) 0%, rgba(2,136,209,0.08) 100%)'
              : 'linear-gradient(115deg, rgba(25,118,210,0.12) 0%, rgba(2,136,209,0.05) 100%)',
        })}
      >
        <Typography variant="h4" gutterBottom sx={{ fontWeight: 700, mb: 1 }}>
          Clinical
        </Typography>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}
      <Toast open={!!notice} message={notice} severity="success" onClose={() => setNotice('')} />

      <Box
        sx={{
          display: 'grid',
          gridTemplateColumns: { xs: '1fr', md: 'minmax(0, 1fr) minmax(0, 1fr)' },
          gap: { xs: 2, md: 3 },
          alignItems: 'stretch',
        }}
      >
        {/* —— 随访 —— */}
        <Card
          variant="outlined"
          sx={(theme) => ({
            borderRadius: 2,
            borderColor: 'divider',
            bgcolor: theme.palette.mode === 'dark' ? 'action.selected' : 'grey.50',
            boxShadow: theme.palette.mode === 'dark' ? 'none' : '0 6px 18px rgba(15,23,42,0.06)',
            transition: 'box-shadow .2s ease, transform .2s ease',
            '&:hover': {
              boxShadow: theme.palette.mode === 'dark' ? 'none' : '0 10px 24px rgba(15,23,42,0.10)',
              transform: 'translateY(-1px)',
            },
          })}
        >
          <CardContent sx={{ p: { xs: 2, sm: 2.5 } }}>
            <ColumnTitle>随访</ColumnTitle>
            <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 0.5 }}>
              病例与生存数据
            </Typography>

            <Divider sx={{ mb: 2 }} />

            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              批量导入
            </Typography>
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, alignItems: 'center' }}>
              <Button component="label" variant="outlined" size="small" disabled={csvUploading}>
                选择 CSV
                <input
                  ref={fileRef}
                  type="file"
                  accept=".csv"
                  hidden
                  disabled={csvUploading}
                  onChange={(e) => {
                    const f = e.target.files?.[0]
                    setCsvPendingName(f ? f.name : '')
                  }}
                />
              </Button>
              <Button
                variant="contained"
                size="small"
                onClick={uploadCsv}
                disabled={csvUploading}
                startIcon={csvUploading ? <CircularProgress size={16} color="inherit" /> : null}
              >
                {csvUploading ? '导入中…' : '导入'}
              </Button>
            </Box>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1, mt: 1, mb: 2 }}>
              {!csvUploading && csvPendingName ? (
                <Chip size="small" color="primary" variant="outlined" label={`已选择：${csvPendingName}`} />
              ) : null}
              {!csvUploading && csvLastImport ? (
                <Alert severity="success" variant="outlined" sx={{ py: 0.5 }}>
                  已成功上传并导入「{csvLastImport.fileName}」，共写入{' '}
                  <strong>{csvLastImport.count != null ? csvLastImport.count : '—'}</strong> 条病例。
                </Alert>
              ) : null}
            </Box>

            <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 1 }}>
              病例操作
            </Typography>
            <Box
              sx={{
                border: '1px solid',
                borderColor: 'divider',
                borderRadius: 1.5,
                bgcolor: 'background.paper',
                p: 1.5,
                mb: 1.5,
                boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.04)',
              }}
            >
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                A. 新增病例
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, alignItems: 'center' }}>
                <TextField
                  size="small"
                  label="新 caseId"
                  value={newCaseId}
                  onChange={(e) => setNewCaseId(e.target.value)}
                  sx={{ minWidth: 200 }}
                  placeholder="如 CASE_1001"
                />
                <Button variant="outlined" size="small" onClick={createCase}>
                  新增并选中
                </Button>
              </Box>
            </Box>

            <Box
              sx={{
                border: '1px solid',
                borderColor: 'divider',
                borderRadius: 1.5,
                bgcolor: 'background.paper',
                p: 1.5,
                boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.04)',
              }}
            >
              <Typography variant="subtitle2" sx={{ fontWeight: 600, mb: 1 }}>
                B. 已有病例管理（选择 / 修改 / 删除）
              </Typography>
              <FormControl fullWidth size="small" sx={{ mb: 1.5 }}>
                <InputLabel id="caseid-edit-label">当前病例</InputLabel>
                <Select
                  labelId="caseid-edit-label"
                  label="当前病例"
                  value={selectedCaseId}
                  onChange={(e) => setSelectedCaseId(e.target.value)}
                >
                  {caseOptions.map((cid) => (
                    <MenuItem key={cid} value={cid}>
                      {cid}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>

              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, alignItems: 'center', mb: 1.5 }}>
                <TextField
                  size="small"
                  label="time"
                  type="number"
                  value={editTime}
                  onChange={(e) => setEditTime(e.target.value)}
                  sx={{ width: 120 }}
                  helperText="随访时间"
                />
                <FormControl size="small" sx={{ minWidth: 140 }}>
                  <InputLabel id="status-edit-label">status</InputLabel>
                  <Select
                    labelId="status-edit-label"
                    label="status"
                    value={editStatus}
                    onChange={(e) => setEditStatus(e.target.value)}
                  >
                    <MenuItem value="0">0 删失</MenuItem>
                    <MenuItem value="1">1 事件</MenuItem>
                  </Select>
                </FormControl>
              </Box>

              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                <Button variant="contained" size="small" onClick={saveFollowup} disabled={!selectedCaseId}>
                  保存随访
                </Button>
                <Button variant="outlined" color="error" size="small" onClick={deleteCase} disabled={!selectedCaseId}>
                  删除病例
                </Button>
              </Box>
            </Box>

            <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
              共 {cases.length} 个病例
            </Typography>

            {selectedCaseId ? (
              <>
                <Divider sx={{ my: 2 }} />
                {hasBoundFeatures ? (
                  <>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                      当前病例特征状态
                    </Typography>
                    <Box
                      sx={(theme) => ({
                        px: 1.25,
                        py: 1,
                        borderRadius: 1.2,
                        border: '1px solid',
                        borderColor: theme.palette.mode === 'dark' ? 'divider' : 'rgba(25,118,210,0.25)',
                        bgcolor: theme.palette.mode === 'dark' ? 'action.hover' : 'rgba(25,118,210,0.04)',
                      })}
                    >
                      <Typography variant="body2" color="text.secondary" component="div" sx={{ lineHeight: 1.7 }}>
                        来源：
                        {selectedCase?.featureSource === 'raster_derived'
                          ? '图像派生'
                          : selectedCase?.featureSource === 'trident_derived'
                            ? 'TRIDENT 派生'
                            : selectedCase?.featureSource === 'h5_pair'
                              ? '已选 H5'
                              : selectedCase?.feature20FileId
                                ? '已登记'
                                : '未关联'}
                        <br />
                        20×：{selectedCase?.feature20FileId ? <code>{selectedCase.feature20FileId}</code> : '—'}
                        <br />
                        10×：{selectedCase?.feature10FileId ? <code>{selectedCase.feature10FileId}</code> : '—'}
                        {selectedCase?.rasterSourceFileName ? (
                          <>
                            <br />
                            图像文件名：{selectedCase.rasterSourceFileName}
                          </>
                        ) : null}
                      </Typography>
                    </Box>
                    <Divider sx={{ my: 2 }} />
                    <Box>
                      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.5 }}>
                        已绑定病例图片预览
                      </Typography>
                      <Typography
                        variant="body2"
                        sx={{ fontWeight: 500 }}
                        noWrap
                        title={boundCasePreview?.name ?? `${selectedCaseId}（已绑定预览）`}
                      >
                        {boundCasePreview?.name ?? `${selectedCaseId}（已绑定预览）`}
                      </Typography>
                    </Box>
                  </>
                ) : (
                  <Typography
                    variant="body2"
                    color="text.secondary"
                    sx={{
                      px: 1.25,
                      py: 1,
                      borderRadius: 1.2,
                      bgcolor: 'action.hover',
                      border: '1px dashed',
                      borderColor: 'divider',
                    }}
                  >
                    绑定成功后将在此显示当前病例的详细特征信息。
                  </Typography>
                )}
              </>
            ) : null}
          </CardContent>
        </Card>

        {/* —— 特征与推理 —— */}
        <Card
          variant="outlined"
          sx={(theme) => ({
            borderRadius: 2,
            borderColor: 'divider',
            bgcolor: theme.palette.mode === 'dark' ? 'background.paper' : '#fcfdff',
            boxShadow: theme.palette.mode === 'dark' ? 'none' : '0 6px 18px rgba(15,23,42,0.06)',
            transition: 'box-shadow .2s ease, transform .2s ease',
            '&:hover': {
              boxShadow: theme.palette.mode === 'dark' ? 'none' : '0 10px 24px rgba(15,23,42,0.10)',
              transform: 'translateY(-1px)',
            },
          })}
        >
          <CardContent sx={{ p: { xs: 2, sm: 2.5 } }}>
            <ColumnTitle>特征与推理</ColumnTitle>
            <Typography variant="subtitle1" sx={{ fontWeight: 600, mb: 0.5 }}>
              从 WSI 生成并绑定双尺度特征
            </Typography>

            <Divider sx={{ mb: 2 }} />

            {!selectedCaseId ? (
              <Alert severity="warning" sx={{ mb: 2 }}>
                请先在<strong>左栏</strong>选择或新增一个病例，再在此绑定特征或上传 WSI。
              </Alert>
            ) : (
              <Box sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
                <Typography variant="body2" color="text.secondary">
                  操作对象
                </Typography>
                <Chip size="small" label={selectedCaseId} color="primary" variant="outlined" />
              </Box>
            )}

            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1.5 }}>
                <ToggleButtonGroup
                  value={genMode}
                  exclusive
                  size="small"
                  fullWidth
                  onChange={(_, v) => v && setGenMode(v)}
                  disabled={!selectedCaseId || associating}
                >
                  <ToggleButton value="quick">快速预览（只跑近似）</ToggleButton>
                  <ToggleButton value="formal">正式预测（TRIDENT）</ToggleButton>
                </ToggleButtonGroup>
                {associating && (
                  <Box sx={{ mb: 0.5 }}>
                    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mb: 0.6 }}>
                      正在生成双尺度特征并写入病例，请稍候（{associateProgress}%）
                    </Typography>
                    <LinearProgress variant="determinate" value={Math.max(2, Math.min(100, associateProgress))} />
                  </Box>
                )}
                <FormControl fullWidth size="small">
                  <InputLabel id="raster-cancer-label">癌种</InputLabel>
                  <Select
                    labelId="raster-cancer-label"
                    label="癌种"
                    value={selectedCancer}
                    onChange={(e) => setSelectedCancer(e.target.value)}
                    disabled={associating}
                  >
                    {cancerOptions.map((opt) => (
                      <MenuItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                {genMode === 'formal' ? (
                  <TextField
                    size="small"
                    label="MPP（可选，PNG/JPEG 必填）"
                    value={tridentMpp}
                    onChange={(e) => setTridentMpp(e.target.value)}
                    placeholder="如 0.25"
                    helperText="WSI 通常可从元数据读取；普通图片无元数据时请填写"
                  />
                ) : null}
                <Button component="label" variant="outlined" disabled={!selectedCaseId || associating}>
                  选择 WSI 文件
                  <input
                    type="file"
                    hidden
                    accept=".svs,.ndpi,.mrxs,.scn,.tif,.tiff"
                    onChange={(e) => {
                      const f = e.target.files?.[0] || null
                      setPersistedRasterPreview((prev) => {
                        if (prev?.url) URL.revokeObjectURL(prev.url)
                        return null
                      })
                      setGeneratedPreview((prev) => {
                        if (prev?.url) URL.revokeObjectURL(prev.url)
                        return null
                      })
                      setRasterFile(f)
                    }}
                  />
                </Button>
                {previewBusy && (
                  <Typography variant="body2" color="text.secondary">
                    正在生成 WSI 缩略预览...
                  </Typography>
                )}
                {!rasterFile && !persistedRasterPreview && (
                  <Typography variant="body2" color="text.secondary">
                    未选择文件
                  </Typography>
                )}
                <RasterPreview
                  file={rasterFile}
                  persisted={persistedRasterPreview}
                  generated={generatedPreview}
                  suppressInlineHints
                />
                <Button
                  variant="contained"
                  onClick={associateFeaturesFromUpload}
                  disabled={!selectedCaseId || associating || !rasterFile}
                >
                  {associating ? (
                    <CircularProgress size={20} color="inherit" />
                  ) : genMode === 'formal' ? (
                    '上传并正式预测（TRIDENT）'
                  ) : (
                    '上传并快速预览'
                  )}
                </Button>

                {selectedCaseId && hasBoundFeatures ? (
                  <>
                    <Divider sx={{ my: 2 }} />
                    {boundCasePreviewBusy ? (
                      <Typography variant="body2" color="text.secondary">
                        正在加载病例预览...
                      </Typography>
                    ) : boundCasePreview ? (
                      <RasterPreview
                        file={null}
                        persisted={boundCasePreview}
                        hideDisplayName
                        suppressInlineHints
                      />
                    ) : (
                      <Typography variant="body2" color="text.secondary">
                        该病例暂无可用预览图
                      </Typography>
                    )}
                  </>
                ) : null}
            </Box>
          </CardContent>
        </Card>
      </Box>
    </Box>
  )
}
