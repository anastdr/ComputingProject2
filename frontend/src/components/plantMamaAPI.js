
// src/components/plantMamaAPI.js
import axios from 'axios';

const BASE_URL = 'https://plantmama-backend.onrender.com'; 

export const sendTextQuery = async (message) => {
  try {
    const res = await axios.post(`${BASE_URL}/chat`, { message });
    return res.data; // 
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
    return res.data; 
  } catch (error) {
    throw new Error('Image query failed.');
  }
};
