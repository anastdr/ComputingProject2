// components/PrivacyModal.jsx

import React from 'react';


export default function PrivacyModel({ onClose }) {
  return (
    <div className="privacy-overlay">
      <div className="privacy-modal">
        <button className="close-btn" onClick={onClose}>✕</button>
        <h2>Privacy Policy</h2>
        <p>
          Welcome to PlantMama 🌿. We care about your data privacy. This app may store plant-related messages and
          image uploads for the purpose of improving assistance and analytics. Your data will never be shared or sold.
          By continuing to use PlantMama, you agree to our terms of use and this privacy policy.
        </p>
        <p style={{ fontSize: '12px', color: '#888' }}>
          You can always review our full privacy policy at any time.
        </p>
        <button onClick={onClose} className="accept-btn">Accept</button>
      </div>
    </div>
  );
}
