const express = require('express');
const multer = require('multer');
const { forwardFrameToAi, checkAiHealth } = require('../services/pythonService');

const router = express.Router();

// Memory storage for incoming frame uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 } // 10MB limit
});

/**
 * GET /api/health
 * Returns status of Node API Gateway and Python AI Service.
 */
router.get('/health', async (req, res, next) => {
  try {
    const aiHealth = await checkAiHealth();
    res.json({
      gateway: 'online',
      aiService: aiHealth.online ? 'online' : 'offline',
      modelLoaded: aiHealth.data?.model_loaded || false
    });
  } catch (err) {
    next(err);
  }
});

/**
 * POST /api/detect
 * Accepts uploaded frame image file (or base64 buffer), forwards to Python AI, returns JSON predictions.
 */
router.post('/detect', upload.single('frame'), async (req, res, next) => {
  try {
    let frameBuffer = null;
    let mimetype = 'image/jpeg';
    let filename = 'frame.jpg';

    if (req.file) {
      frameBuffer = req.file.buffer;
      mimetype = req.file.mimetype;
      filename = req.file.originalname || 'frame.jpg';
    } else if (req.body && req.body.image) {
      // Base64 fallback handling
      const base64Data = req.body.image.replace(/^data:image\/\w+;base64,/, '');
      frameBuffer = Buffer.from(base64Data, 'base64');
    } else {
      return res.status(400).json({ error: true, message: 'No frame provided in request.' });
    }

    const aiResponse = await forwardFrameToAi(frameBuffer, filename, mimetype);
    res.json(aiResponse);
  } catch (err) {
    if (err.code === 'ECONNREFUSED' || err.message.includes('unreachable')) {
      return res.status(503).json({
        error: true,
        message: 'AI Service is currently offline. Please ensure Python FastAPI service is running.'
      });
    }
    next(err);
  }
});

module.exports = router;
