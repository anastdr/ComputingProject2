
// src/components/plantMamaAPI.js
import axios from 'axios';

const BASE_URL = 'http://127.0.0.1:8000'; // Update this to your deployed backend URL if needed

export const sendTextQuery = async (message) => {
  try {
    const res = await axios.post(`${BASE_URL}/chat`, { message });
    return res.data; // should contain { reply: "..." }
  } catch (error) {
    throw new Error('Text query failed.');
  }
};

export const sendImageQuery = async (imageFile) => {
  const formData = new FormData();
  formData.append('image', imageFile);

  try {
    const res = await axios.post(`${BASE_URL}/predict-chatbot`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return res.data; // should contain { plant, disease, chatbot_message }
  } catch (error) {
    throw new Error('Image query failed.');
  }
};
