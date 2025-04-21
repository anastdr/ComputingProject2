// WelcomeScreen.jsx
import React from 'react';
import { useNavigate } from 'react-router-dom';

export default function WelcomeScreen() {
  const navigate = useNavigate();

  return (
    <div className="main-screen">
      <h1 className="highlight">
        Welcome to PlantMama!
      </h1>
      <p className="description">
        Discover your plant care companion. We guide you on how to best treat your leafy friends.
      </p>
      <button
        className="start-btn bg-white text-black px-6 py-3 rounded-lg font-medium transition duration-300 hover:bg-green-400 hover:text-white hover:shadow-lg"
        onClick={() => navigate('/chat')}
      >
        Start chatting
      </button>
    </div>
  );
}
