/** 与后端 api_server._scrub_training_log_content 同类的行级过滤，避免训练页 Log tail 展示 NNPACK 噪声。 */
const TRAINING_LOG_SCRUB =
  /\bnnpack\b|could not initialize nnpack|nnpack is not supported|compiled without nnpack/i

export function sanitizeTrainingLogContent(text) {
  if (text == null || text === '') return ''
  if (typeof text !== 'string') return String(text)
  return text
    .split('\n')
    .filter((line) => !TRAINING_LOG_SCRUB.test(line))
    .join('\n')
}
