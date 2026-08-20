import React from 'react';
import StatusIndicator from './StatusIndicator';

const Header = ({ isAiOnline }) => {
  return (
    <header className="header-card">
      <div className="header-title-section">
        <div className="header-icon">😷</div>
        <div>
          <h1 className="header-title">Face Mask Detection AI</h1>
          <p className="header-subtitle">Real-Time Computer Vision & Multi-Face Classification System</p>
        </div>
      </div>
      <StatusIndicator isOnline={isAiOnline} />
    </header>
  );
};

export default Header;
