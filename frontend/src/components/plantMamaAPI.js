
// src/components/plantMamaAPI.js
import axios from 'axios';

const BASE_URL = import.meta.env.VITE_API_URL;

export const sendTextQuery = async (message) => {
    try {
      const res = await axios.post(`${BASE_URL}/chat`, { message });
      console.log('Response from backend:', res); 
      return res.data; 
    } catch (error) {
      console.error('❌ sendTextQuery failed:', error); 
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
