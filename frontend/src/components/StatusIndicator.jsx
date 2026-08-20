import React from 'react';

const StatusIndicator = ({ isOnline }) => {
  return (
    <div className={`status-badge ${isOnline ? 'online' : 'offline'}`}>
      <span className="status-dot"></span>
      <span>{isOnline ? 'AI ONLINE' : 'AI OFFLINE'}</span>
    </div>
  );
};

export default StatusIndicator;
