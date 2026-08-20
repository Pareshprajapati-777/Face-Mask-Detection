const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const detectionRoutes = require('./routes/detection');
const errorHandler = require('./middleware/errorHandler');

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

// Enable CORS
app.use(cors({
  origin: '*',
  methods: ['GET', 'POST', 'PUT', 'DELETE'],
  allowedHeaders: ['Content-Type', 'Authorization']
}));

// Body Parsers
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Mount API Gateway Routes
app.use('/api', detectionRoutes);

// Root Endpoint
app.get('/', (req, res) => {
  res.json({
    name: 'Face Mask Detection API Gateway',
    version: '1.0.0',
    status: 'online',
    endpoints: {
      health: 'GET /api/health',
      detect: 'POST /api/detect'
    }
  });
});

// Centralized Error Handling Middleware
app.use(errorHandler);

// Start Express HTTP Server
app.listen(PORT, () => {
  console.log(`==================================================`);
  console.log(` Node.js Express API Gateway running on port ${PORT}`);
  console.log(` Target AI Service: ${process.env.AI_SERVER_URL || 'http://127.0.0.1:8000'}`);
  console.log(`==================================================`);
});
