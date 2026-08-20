import React, { useState, useEffect, useCallback } from 'react';
import Header from './components/Header';
import Camera from './components/Camera';
import StatsPanel from './components/StatsPanel';
import { AlertCircle } from 'lucide-react';

const App = () => {
  const [isAiOnline, setIsAiOnline] = useState(false);
  const [stats, setStats] = useState({
    total_faces: 0,
    mask_count: 0,
    no_mask_count: 0,
    unknown_count: 0
  });
  const [fps, setFps] = useState(0);
  const [appError, setAppError] = useState(null);

  // Check AI & Node Gateway health status
  const checkHealth = useCallback(async () => {
    try {
      const response = await fetch('/api/health');
      if (response.ok) {
        const data = await response.json();
        setIsAiOnline(data.aiService === 'online' && data.modelLoaded);
      } else {
        setIsAiOnline(false);
      }
    } catch (err) {
      setIsAiOnline(false);
    }
  }, []);

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 5000);
    return () => clearInterval(interval);
  }, [checkHealth]);

  const handleDetectionResults = useCallback((data, currentFps) => {
    setStats({
      total_faces: data.total_faces || 0,
      mask_count: data.mask_count || 0,
      no_mask_count: data.no_mask_count || 0,
      unknown_count: data.unknown_count || 0
    });
    setFps(currentFps);
  }, []);

  const handleErrorChange = useCallback((msg) => {
    setAppError(msg);
  }, []);

  return (
    <div className="app-container">
      <Header isAiOnline={isAiOnline} />

      {appError && (
        <div className="alert-banner error">
          <AlertCircle size={20} />
          <span>{appError}</span>
        </div>
      )}

      <main className="main-layout">
        <Camera
          onDetectionResults={handleDetectionResults}
          onErrorChange={handleErrorChange}
          isAiOnline={isAiOnline}
        />
        <StatsPanel
          stats={stats}
          fps={fps}
          isAiOnline={isAiOnline}
        />
      </main>
    </div>
  );
};

export default App;
