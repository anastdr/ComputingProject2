// App.jsx
import React, { useState, useEffect } from 'react';
import { Outlet } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import PrivacyModel from './components/PrivacyModel';

export default function App() {
  const [showPrivacy, setShowPrivacy] = useState(true);

  const handleOpenPrivacy = () => {
    setShowPrivacy(true);
  };

  const handleClosePrivacy = () => {
    setShowPrivacy(false);
  };

  useEffect(() => {
    const hour = new Date().getHours();
    const isLightTime = hour >= 6 && hour < 18;
  
    if (isLightTime) {
      document.body.classList.add('light-mode');
    } else {
      document.body.classList.remove('light-mode');
    }
  }, []);
  
  return (
    <div className="app-container">
      <Sidebar onPrivacyClick={handleOpenPrivacy} />
      <main className="main-content">
        {showPrivacy && <PrivacyModel onClose={handleClosePrivacy} />}
        <Outlet />
      </main>
    </div>
  );
}
