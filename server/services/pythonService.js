const axios = require('axios');
const FormData = require('form-data');

const AI_SERVER_URL = process.env.AI_SERVER_URL || 'http://127.0.0.1:8000';

/**
 * Check health status of Python FastAPI AI service.
 */
const checkAiHealth = async () => {
  try {
    const response = await axios.get(`${AI_SERVER_URL}/health`, { timeout: 3000 });
    return { online: true, data: response.data };
  } catch (error) {
    return { online: false, error: 'AI service unreachable' };
  }
};

/**
 * Forward image buffer to Python FastAPI /predict endpoint.
 */
const forwardFrameToAi = async (fileBuffer, filename = 'frame.jpg', mimetype = 'image/jpeg') => {
  const form = new FormData();
  form.append('file', fileBuffer, {
    filename,
    contentType: mimetype
  });

  const response = await axios.post(`${AI_SERVER_URL}/predict`, form, {
    headers: {
      ...form.getHeaders()
    },
    timeout: 5000
  });

  return response.data;
};

module.exports = {
  checkAiHealth,
  forwardFrameToAi
};
