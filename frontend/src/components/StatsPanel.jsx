import React from 'react';
import { Users, ShieldCheck, ShieldAlert, HelpCircle, Activity } from 'lucide-react';

const StatsPanel = ({ stats, fps, isAiOnline }) => {
  return (
    <div className="stats-panel">
      {/* Total Faces */}
      <div className="stat-card">
        <div className="stat-icon-wrapper stat-icon-blue">
          <Users size={24} />
        </div>
        <div className="stat-info">
          <span className="stat-label">Faces Detected</span>
          <span className="stat-value">{stats.total_faces || 0}</span>
        </div>
      </div>

      {/* Mask Count */}
      <div className="stat-card">
        <div className="stat-icon-wrapper stat-icon-green">
          <ShieldCheck size={24} />
        </div>
        <div className="stat-info">
          <span className="stat-label">😷 Mask (SAFE)</span>
          <span className="stat-value" style={{ color: '#22c55e' }}>{stats.mask_count || 0}</span>
        </div>
      </div>

      {/* No Mask Count */}
      <div className="stat-card">
        <div className="stat-icon-wrapper stat-icon-red">
          <ShieldAlert size={24} />
        </div>
        <div className="stat-info">
          <span className="stat-label">⚠️ No Mask (UNSAFE)</span>
          <span className="stat-value" style={{ color: '#ef4444' }}>{stats.no_mask_count || 0}</span>
        </div>
      </div>

      {/* Unknown Count */}
      <div className="stat-card">
        <div className="stat-icon-wrapper stat-icon-yellow">
          <HelpCircle size={24} />
        </div>
        <div className="stat-info">
          <span className="stat-label">❓ Unknown</span>
          <span className="stat-value" style={{ color: '#f59e0b' }}>{stats.unknown_count || 0}</span>
        </div>
      </div>

      {/* FPS Counter */}
      <div className="stat-card">
        <div className="stat-icon-wrapper stat-icon-blue">
          <Activity size={24} />
        </div>
        <div className="stat-info">
          <span className="stat-label">FPS (Processing Rate)</span>
          <span className="stat-value">{fps}</span>
        </div>
      </div>
    </div>
  );
};

export default StatsPanel;
