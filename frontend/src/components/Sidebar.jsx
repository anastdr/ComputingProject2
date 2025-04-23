// Sidebar.jsx
// Sidebar.jsx
import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

export default function Sidebar() {
  const location = useLocation();
  const navigate = useNavigate();
  const currentPage = location.pathname;
  

  return (
    <aside className="sidebar">
      <div>
        <h1
          className="logo"
          onClick={() => navigate('/')}
        >
          PlantMama
        </h1>
        <ul className="nav-links">
          <li
            className={currentPage === '/' ? 'active' : ''}
            onClick={() => navigate('/')}
          >
            🏠 Home
          </li>
          <li
            className={currentPage === '/chat' ? 'active' : ''}
            onClick={() => navigate('/chat')}
          >
            💬 Chat
          </li>
          <li>🏆 Rewards</li>
          <li>🌿 Plants collection</li>
        </ul>
      </div>
      <div>
        <button className="chat-btn" 
        onClick={() => navigate('/chat')}>Chat now</button>
        <p
          className="policy-text hover:text-white cursor-pointer transition"
          
        >
          Privacy policy
        </p>
        <p className="policy-text">PlantMama. Limited. All rights reserved</p>
      </div>
    </aside>
  );
}
