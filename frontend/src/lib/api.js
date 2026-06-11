import axios from "axios";

export const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
export const API = `${BACKEND_URL}/api`;

export const artifactUrl = (rel) => `${API}/${rel}`;

const api = axios.create({ baseURL: API });

export default api;
