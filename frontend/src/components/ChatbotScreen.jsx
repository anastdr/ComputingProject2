// components/ChatbotScreen.jsx

// src/components/ChatbotScreen.jsx
import React, { useState, useRef, useEffect } from 'react';
import { sendTextQuery, sendImageQuery } from './plantMamaAPI'; // ✅ Import from helper

export default function ChatbotScreen() {
  const [messages, setMessages] = useState([
    { from: 'bot', text: 'Hi! How can I help you with your plants today?', animate: false }
  ]);
  const [input, setInput] = useState('');
  const [image, setImage] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const messagesEndRef = useRef(null);

  const handleSend = async () => {
    if (!input.trim() && !image) return;

    setError('');
    const newMessages = [];

    if (input.trim()) {
      const userMsg = { from: 'user', text: input.trim(), animate: true };
      setMessages((prev) => [...prev, userMsg]);
      setLoading(true);
      try {
        const data = await sendTextQuery(input);
        newMessages.push({ from: 'bot', text: data.reply, animate: true });
      } catch (err) {
        setError('Failed to get response from chatbot.');
      } finally {
        setLoading(false);
      }
      setInput('');
    }

    if (image) {
      const userImgMsg = {
        from: 'user',
        text: '',
        imageUrl,
        animate: true
      };
      setMessages((prev) => [...prev, userImgMsg]);
      setLoading(true);

      try {
        const data = await sendImageQuery(image);
        const { plant, disease, chatbot_message } = data;

        newMessages.push({
          from: 'bot',
          text: `🌿 Plant: ${plant}\n🦠 Disease: ${disease}\n\n${chatbot_message}`,
          animate: true
        });
      } catch (err) {
        setError('Image upload failed. Try again.');
      } finally {
        setLoading(false);
        setImage(null);
        setImageUrl(null);
      }
    }

    if (newMessages.length > 0) {
      setMessages((prev) => [...prev, ...newMessages]);
    }
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      setImageUrl(URL.createObjectURL(file));
    }
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  return (
    <div className="chat-screen">
      <div className="messages">
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.from} ${msg.animate ? 'fade-in' : ''}`}>
            {msg.text?.split('\n').map((line, i) => (
              <div key={i}>{line}</div>
            ))}
            {msg.imageUrl && (
              <img src={msg.imageUrl} alt="Uploaded" className="image-preview" />
            )}
          </div>
        ))}
        {loading && (
          <div className="message bot fade-in">
             <ClipLoader size={25} color="#4CAF50" />
          </div>
        )}
        {error && (
          <div className="message error">
            ⚠️ {error}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      <div className="input-box">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && handleSend()}
          placeholder="Ask me anything about plants..."
        />

        <input
          type="file"
          accept="image/*"
          onChange={handleImageUpload}
          style={{ display: 'none' }}
          id="image-upload"
        />

        <label htmlFor="image-upload" className="icon-btn photo-btn">📷</label>
        <button onClick={handleSend} className="icon-btn">➤</button>
      </div>

      {image && (
        <p style={{ color: 'white', marginTop: '10px', fontSize: '14px' }}>
          📷 {image.name}
        </p>
      )}
    </div>
  );
}
