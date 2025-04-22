import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import App from './App';
import WelcomeScreen from './components/WelcomeScreen';
import ChatbotScreen from './components/ChatbotScreen';
import './App.css';
import PrivacyModel from './components/PrivacyModel';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <Router>
      <Routes>
        <Route path="/" element={<App />}>          
          <Route index element={<WelcomeScreen />} />
          <Route path="chat" element={<ChatbotScreen />} />
          <Route path="privacy" element={<PrivacyModel />} />
        </Route>
      </Routes>
    </Router>
  </React.StrictMode>
);